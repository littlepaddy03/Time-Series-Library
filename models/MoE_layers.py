import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE_FFN(nn.Module):
    """
    MoE Feed-Forward Network Layer for Switch Transformer.
    This layer replaces the standard FFN in a Transformer encoder layer.
    """
    def __init__(self, d_model, d_ff, n_experts, top_k=1, dropout=0.1, activation="relu", aux_loss_weight=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # Gating network
        self.gate = nn.Linear(d_model, n_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_experts)
        ])

        self.aux_loss = 0

    def forward(self, x):
        # x: [batch_size, patch_num, d_model]
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model) # [batch_size * patch_num, d_model]

        # Gate logits
        gate_logits = self.gate(x) # [B*N, num_experts]

        # Get top-k experts
        weights, indices = torch.topk(F.softmax(gate_logits, dim=1), self.top_k, dim=1) # [B*N, top_k]

        # Create a sparse dispatch tensor
        mask = F.one_hot(indices.squeeze(), num_classes=self.n_experts) # [B*N, top_k, num_experts]

        # --- Load Balancing Loss ---
        # Calculate as per the Switch Transformer paper
        samples_per_expert = mask.float().sum(dim=0).squeeze() # [num_experts]
        fraction_samples_per_expert = samples_per_expert / samples_per_expert.sum()

        prob_per_expert = F.softmax(gate_logits, dim=1).mean(dim=0)

        load_balancing_loss = self.n_experts * torch.sum(fraction_samples_per_expert * prob_per_expert)
        self.aux_loss = self.aux_loss_weight * load_balancing_loss

        # --- Dispatch to Experts ---
        expert_outputs = []
        for i in range(self.n_experts):
            expert_outputs.append(self.experts[i](x))
        expert_outputs = torch.stack(expert_outputs, dim=1) # [B*N, num_experts, d_model]

        # Combine expert outputs with gating weights
        # Mask shape: [B*N, num_experts]
        # Weights shape: [B*N, 1]
        # Expert outputs shape: [B*N, num_experts, d_model]
        output = torch.einsum('be,bed->bd', (mask.squeeze() * weights), expert_outputs)

        output = output.reshape(batch_size, seq_len, d_model) # [batch_size, patch_num, d_model]

        return output, self.aux_loss


class MoE_EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with MoE FFN.
    """
    def __init__(self, attention, d_model, d_ff=None, n_experts=8, dropout=0.1, activation="relu", aux_loss_weight=0.1):
        super(MoE_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.moe_ffn = MoE_FFN(d_model, d_ff, n_experts, dropout=dropout, activation=activation, aux_loss_weight=aux_loss_weight)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.aux_loss = 0

    def forward(self, x, attn_mask=None, **kwargs):
        # 1. Multi-Head Attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = self.norm1(x)

        # 2. MoE FFN
        ffn_output, self.aux_loss = self.moe_ffn(y)
        y = y + self.dropout(ffn_output)

        return self.norm2(y), attn
