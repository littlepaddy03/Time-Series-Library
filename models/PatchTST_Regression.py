import torch
import torch.nn as nn
from models.PatchTST import Model as PatchTSTModel

class PatchTST_backbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # We need a dummy task_name for the original model
        class DummyArgs:
            def __init__(self):
                self.task_name = 'long_term_forecast'

        dummy_configs = configs
        dummy_configs.task_name = 'long_term_forecast'

        self.patchtst = PatchTSTModel(dummy_configs)
        self.d_model = configs.d_model

    def forward(self, x_dynamic, x_static_proj): # x_dynamic: [bs x seq_len x n_vars], x_static_proj: [bs x d_model]
        # do patching and embedding
        x_dynamic = x_dynamic.permute(0, 2, 1) # [bs x n_vars x seq_len]
        enc_out, n_vars = self.patchtst.patch_embedding(x_dynamic)

        # --- Early Fusion: Add static features to each patch ---
        # enc_out shape: [bs * n_vars, patch_num, d_model]
        # x_static_proj shape: [bs, d_model]

        patch_num = enc_out.shape[1]

        # Reshape and expand static features
        # [bs, d_model] -> [bs, 1, d_model] -> [bs * n_vars, 1, d_model]
        static_embedding = x_static_proj.unsqueeze(1).repeat(n_vars, 1, 1)

        # [bs * n_vars, 1, d_model] -> [bs * n_vars, patch_num, d_model]
        static_embedding_expanded = static_embedding.expand(-1, patch_num, -1)

        enc_out = enc_out + static_embedding_expanded

        # Encoder
        enc_out, attns = self.patchtst.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        return enc_out


class Model(nn.Module):
    """
    第3章模型: "单一全局" 时空 Transformer (Seq2One 回归)
    
    重用 TSLib 的 PatchTST_backbone, 并添加一个自定义的回归头。
    这个回归头会融合来自 Transformer 的时序表征和静态特征。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # --- 1. 初始化 TSLib 的 PatchTST 骨干网络 ---
        self.backbone = PatchTST_backbone(configs)
        self.d_model = configs.d_model
        
        # --- 2. 新增: 静态特征投影层 (Static Feature Projector) ---
        self.static_feat_dim = configs.static_feat_dim
        self.static_projector = nn.Linear(self.static_feat_dim, self.d_model)
        self.proj_norm = nn.LayerNorm(self.d_model)
        self.proj_dropout = nn.Dropout(configs.dropout)

        # --- 3. 定义回归头 (Regression Head) ---
        # 输入维度现在只有 d_model, 因为融合已在backbone中完成
        head_input_dim = configs.d_model
        
        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, configs.head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static):
        # Replace NaNs with 0 to ensure robustness
        x_dynamic = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        # --- 1. Project and normalize static features ---
        x_static_proj = self.static_projector(x_static)
        x_static_proj = self.proj_norm(x_static_proj)
        x_static_proj = self.proj_dropout(x_static_proj)

        # --- 2. Pass both dynamic and projected static features to the backbone ---
        # (B, L, C), (B, D) -> (B, n_vars, N, D)
        time_series_repr = self.backbone(x_dynamic, x_static_proj)
        
        # --- 3. Pool the output from the backbone ---
        # (B, n_vars, N, D) -> (B, D)
        time_series_repr_flat = time_series_repr.mean(dim=1)[:, -1, :]
        
        # --- 4. Pass the pooled representation to the regression head ---
        # (B, D) -> (B, 1)
        prediction = self.regression_head(time_series_repr_flat)
        
        return prediction
