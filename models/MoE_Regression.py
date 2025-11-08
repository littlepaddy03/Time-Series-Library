import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST_Regression import PatchTST_backbone

class GatingNetwork(nn.Module):
    """
    门控网络: 根据动态和静态特征决定将样本路由到哪个专家。
    """
    def __init__(self, input_dim, num_experts, top_k=1):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        self.layer = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.layer(x)
        # 使用 softmax 获取路由权重
        gates = F.softmax(logits, dim=1)
        # 选择 top_k 个专家
        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=1)

        # 创建一个稀疏的路由掩码
        zeros = torch.zeros_like(gates)
        sparse_gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return sparse_gates, top_k_indices.squeeze()

class Model(nn.Module):
    """
    第4章模型: 混合专家 (MoE) 回归模型 (参考 Switch Transformer)
    - 使用共享的 Backbone 提取时序特征
    - 门控网络同时使用时序和静态特征
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_experts = configs.n_experts
        self.aux_loss_weight = configs.aux_loss_weight
        self.d_model = configs.d_model

        # 0. Shared Backbone
        self.backbone = PatchTST_backbone(configs)

        # 1. Gating Network
        # The input now includes both time-series (d_model) and static features
        gating_input_dim = configs.d_model + configs.static_feat_dim
        self.gating = GatingNetwork(gating_input_dim, self.num_experts)

        # 2. Expert Networks (Simplified to simple Linear layers)
        self.experts = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model) for _ in range(self.num_experts)])

        # 3. Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static):
        # Replace NaNs with 0
        x_dynamic = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        batch_size, _, _ = x_dynamic.shape

        # 0. Shared Backbone: Extract time-series features
        # (B, L, C) -> (B, n_vars, N, D)
        ts_repr = self.backbone(x_dynamic)
        # (B, n_vars, N, D) -> (B, D)
        # Average over n_vars and take the last patch's representation
        ts_embedding = ts_repr.mean(dim=1)[:, -1, :]

        # 1. Gating: Combine features and get routing decisions
        # We detach the time-series embedding to prevent the gating network's loss from affecting the backbone's training
        gating_input = torch.cat((ts_embedding.detach(), x_static), dim=1)
        gates, expert_indices = self.gating(gating_input) # gates: (B, N_experts), expert_indices: (B,)

        # --- 计算负载均衡损失 (Load Balancing Loss) ---
        samples_per_expert = F.one_hot(expert_indices, self.num_experts).float()
        fraction_samples_per_expert = samples_per_expert.mean(dim=0)
        prob_per_expert = gates.mean(dim=0)
        load_balancing_loss = self.num_experts * torch.sum(fraction_samples_per_expert * prob_per_expert)
        self.aux_loss = self.aux_loss_weight * load_balancing_loss

        # --- 分发到专家 (Dispatch to Experts) ---
        final_output = torch.zeros(batch_size, self.d_model).to(x_dynamic.device)

        for i in range(self.num_experts):
            idx = torch.where(expert_indices == i)[0]

            if idx.numel() > 0:
                expert_input = ts_embedding[idx]
                expert_output = self.experts[i](expert_input)

                gate_scores = gates[idx, i].unsqueeze(1)
                weighted_output = expert_output * gate_scores

                final_output.index_add_(0, idx, weighted_output)

        # 3. 回归头
        prediction = self.regression_head(final_output)

        # 返回 expert_indices 以便在 exp_yield_regression.py 中进行诊断
        return prediction, self.aux_loss, expert_indices
