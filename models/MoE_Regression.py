import torch
import torch.nn as nn
from models.PatchTST import Model as PatchTST_Base_Model
from layers.Transformer_EncDec import Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.MoE_layers import MoE_EncoderLayer

class Model(PatchTST_Base_Model):
    """
    Switch Transformer (MoE) for Yield Regression.
    This model inherits from the original PatchTST model and replaces its encoder
    with an MoE-based encoder.
    """
    def __init__(self, configs):
        # Initialize the base PatchTST model.
        # It handles all configurations, patching, and embedding layers.
        super().__init__(configs)

        # --- Override the Encoder with our MoE Encoder ---
        self.encoder = Encoder(
            [
                MoE_EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, getattr(configs, 'factor', 1), attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    n_experts=getattr(configs, 'n_experts', 8),
                    dropout=configs.dropout,
                    activation=configs.activation,
                    aux_loss_weight=getattr(configs, 'aux_loss_weight', 0.1)
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # --- Define the Regression Head ---
        # The base model's head is for forecasting, we need a custom one for regression.
        self.static_feat_dim = getattr(configs, 'static_feat_dim', 0)

        # Calculate the flattened dimension from the encoder output
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        patch_num = int((configs.seq_len - patch_len) / stride + 1)

        # The flattened time-series representation will be of size n_vars * patch_num * d_model
        ts_repr_dim = configs.enc_in * patch_num * configs.d_model

        head_input_dim = ts_repr_dim + self.static_feat_dim

        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, configs.head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static):
        # x_dynamic: [B, L, C]
        # x_static: [B, S]
        x_dynamic = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        B, L, C = x_dynamic.shape

        # --- Use the base model's patching and embedding ---
        x_dynamic = x_dynamic.permute(0, 2, 1)  # B, C, L
        enc_in, n_vars = self.patch_embedding(x_dynamic) # enc_in: [B * C, N, D]

        # --- Use our overridden MoE Encoder ---
        enc_out, attns = self.encoder(enc_in) # enc_out: [B * C, N, D]

        # --- Aggregate auxiliary losses from all MoE layers ---
        total_aux_loss = 0
        for layer in self.encoder.layers:
            if hasattr(layer, 'aux_loss'):
                total_aux_loss += layer.aux_loss

        # --- Prepare for Regression Head ---
        # Reshape back to [B, C, N, D]
        enc_out = enc_out.reshape(B, n_vars, -1, self.d_model)

        # Flatten time-series features
        ts_repr_flat = enc_out.reshape(B, -1) # B, C*N*D

        # Combine with static features
        combined_features = torch.cat([ts_repr_flat, x_static], dim=1)

        # Regression
        prediction = self.regression_head(combined_features)

        # We don't have a single set of expert indices to return anymore
        return prediction, total_aux_loss, None
