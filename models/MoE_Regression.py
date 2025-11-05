import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST_Regression import PatchTST_backbone

class GatingNetwork(nn.Module):
    """
    门控网络: 根据静态特征决定将样本路由到哪个专家。
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

class Expert(nn.Module):
    """
    专家网络: 一个完整的 PatchTST backbone。
    """
    def __init__(self, configs):
        super(Expert, self).__init__()
        self.backbone = PatchTST_backbone(configs)

    def forward(self, x):
        # (B, L, C) -> (B, n_vars, N, D)
        time_series_repr = self.backbone(x)
        # (B, n_vars, N, D) -> (B, D)
        # Average over n_vars and take the last patch
        return time_series_repr.mean(dim=1)[:, -1, :]

class Model(nn.Module):
    """
    第4章模型: 混合专家 (MoE) 回归模型 (参考 Switch Transformer)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_experts = configs.n_experts
        self.aux_loss_weight = configs.aux_loss_weight

        # 1. Gating Network
        self.gating = GatingNetwork(configs.static_feat_dim, self.num_experts)

        # 2. Expert Networks
        self.experts = nn.ModuleList([Expert(configs) for _ in range(self.num_experts)])

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

        # 1. Gating: 获取路由权重和决策
        gates, expert_indices = self.gating(x_static) # gates: (B, N_experts), expert_indices: (B,)

        # --- 计算负载均衡损失 (Load Balancing Loss) ---
        # f_i: 每个专家处理的样本比例
        # P_i: 路由到每个专家的概率总和

        samples_per_expert = F.one_hot(expert_indices, self.num_experts).float() # (B, N_experts)
        fraction_samples_per_expert = samples_per_expert.mean(dim=0) # f_i

        prob_per_expert = gates.mean(dim=0) # P_i

        # L_aux = N * sum(f_i * P_i)
        load_balancing_loss = self.num_experts * torch.sum(fraction_samples_per_expert * prob_per_expert)

        self.aux_loss = self.aux_loss_weight * load_balancing_loss

        # --- 分发到专家 (Dispatch to Experts) ---
        final_output = torch.zeros(batch_size, self.experts[0].backbone.d_model).to(x_dynamic.device)

        # 将门控权重应用到专家的输出
        for i in range(self.num_experts):
            idx = torch.where(expert_indices == i)[0]

            if idx.numel() > 0:
                expert_input = x_dynamic[idx]
                expert_output = self.experts[i](expert_input)

                # 获取对应样本的门控权重
                gate_scores = gates[idx, i].unsqueeze(1)

                # 加权输出
                weighted_output = expert_output * gate_scores

                # 将结果放回原位
                final_output.index_add_(0, idx, weighted_output)

        # 3. 回归头
        prediction = self.regression_head(final_output)

        return prediction, self.aux_loss
