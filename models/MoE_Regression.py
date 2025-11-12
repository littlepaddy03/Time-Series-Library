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
    with an MoE-based encoder. It aligns the regression head design with the
    successful baseline model (PatchTST_Regression) and defensively handles all
    necessary configurations. It also returns expert affinities for analysis.
    """
    def __init__(self, configs):
        super().__init__(configs)
        self.d_model = configs.d_model

        n_heads = configs.n_heads
        d_ff = configs.d_ff
        e_layers = configs.e_layers
        dropout = configs.dropout
        activation = configs.activation
        output_attention = getattr(configs, 'output_attention', False)
        factor = getattr(configs, 'factor', 1)
        self.n_experts = getattr(configs, 'n_experts', 8)
        aux_loss_weight = getattr(configs, 'aux_loss_weight', 0.1)

        self.encoder = Encoder(
            [
                MoE_EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), self.d_model, n_heads),
                    self.d_model,
                    d_ff,
                    n_experts=self.n_experts,
                    dropout=dropout,
                    activation=activation,
                    aux_loss_weight=aux_loss_weight
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        static_feat_dim = getattr(configs, 'static_feat_dim', 0)
        head_mlp_dim = getattr(configs, 'head_mlp_dim', 128)
        head_input_dim = self.d_model + static_feat_dim

        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static):
        x_dynamic = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        B, L, C = x_dynamic.shape

        x_dynamic = x_dynamic.permute(0, 2, 1)
        enc_in, n_vars = self.patch_embedding(x_dynamic)

        enc_out, attns = self.encoder(enc_in)

        # --- Aggregate auxiliary losses and gate probabilities from all MoE layers ---
        total_aux_loss = 0
        all_gate_probs = []
        for layer in self.encoder.attn_layers:
            if hasattr(layer, 'aux_loss'):
                total_aux_loss += layer.aux_loss
            if hasattr(layer, 'gate_prob') and layer.gate_prob is not None:
                all_gate_probs.append(layer.gate_prob)

        # --- Calculate Per-Layer Sample Affinity Score ---
        if len(all_gate_probs) > 0:
            # Stack gate probabilities across layers. Shape: [e_layers, B*C*N, n_experts]
            stacked_gate_probs = torch.stack(all_gate_probs, dim=0)

            # Reshape to bring Batch dimension to the front. Shape: [e_layers, B, C*N, n_experts]
            layer_token_affinity = stacked_gate_probs.reshape(
                len(all_gate_probs), B, -1, self.n_experts
            )

            # Average across all tokens (patches) for each sample, keeping the layer dimension.
            # Shape: [e_layers, B, n_experts]
            layer_sample_affinity = layer_token_affinity.mean(dim=2)

            # Permute to get the desired [B, e_layers, n_experts] shape
            sample_affinity = layer_sample_affinity.permute(1, 0, 2)
        else:
            sample_affinity = None

        # --- Pooling & Feature Combination ---
        enc_out = enc_out.reshape(B, n_vars, -1, self.d_model)
        ts_repr_pooled = enc_out.mean(dim=1)[:, -1, :]

        combined_features = torch.cat([ts_repr_pooled, x_static], dim=1)

        prediction = self.regression_head(combined_features)

        return prediction, total_aux_loss, sample_affinity
