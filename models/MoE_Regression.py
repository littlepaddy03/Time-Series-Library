import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.MoE_layers import MoE_EncoderLayer

class Model(nn.Module):
    """
    Switch Transformer (MoE) for Yield Regression.
    Replaces the FFN layers in PatchTST with MoE layers.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.n_experts = configs.n_experts
        self.aux_loss_weight = configs.aux_loss_weight

        # Patching
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        self.padding_patch = nn.ReplicationPad1d((0, self.stride))

        # Backbone
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, configs.n_vars, self.patch_num, self.d_model))
        self.dropout = nn.Dropout(configs.dropout)

        # MoE Encoder
        self.encoder = Encoder(
            [
                MoE_EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    n_experts=self.n_experts,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    aux_loss_weight=self.aux_loss_weight
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Regression Head
        self.static_feat_dim = configs.static_feat_dim
        # The flattened time-series representation will be of size n_vars * patch_num * d_model
        head_input_dim = configs.n_vars * self.patch_num * self.d_model + self.static_feat_dim

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
        x = x_dynamic.permute(0, 2, 1) # B, C, L

        # Patching
        x = self.padding_patch(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # B, C, N, P
        x = x.permute(0, 1, 3, 2) # B, C, P, N

        # Embedding
        enc_in = self.patch_embedding(x) # B, C, N, D
        enc_in = enc_in + self.pos_embedding
        enc_in = self.dropout(enc_in)

        # Encoder
        # Reshape for encoder: [B*C, N, D]
        enc_in = enc_in.reshape(-1, self.patch_num, self.d_model)
        enc_out, attns = self.encoder(enc_in) # [B*C, N, D]

        # Reshape back: [B, C, N, D]
        enc_out = enc_out.reshape(B, C, self.patch_num, self.d_model)

        # --- Aggregate auxiliary losses from all MoE layers ---
        total_aux_loss = 0
        for layer in self.encoder.layers:
            total_aux_loss += layer.aux_loss

        # Flatten time-series features
        ts_repr_flat = enc_out.reshape(B, -1) # B, C*N*D

        # Combine with static features
        combined_features = torch.cat([ts_repr_flat, x_static], dim=1)

        # Regression
        prediction = self.regression_head(combined_features)

        return prediction, total_aux_loss, None # No single expert_indices to return
