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

    def forward(self, x): # x: [bs x seq_len x n_vars]
        # do patching and embedding
        x = x.permute(0, 2, 1) # [bs x n_vars x seq_len]
        enc_out, n_vars = self.patchtst.patch_embedding(x)

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
        
        # --- 2. 定义回归头 (Regression Head) ---
        self.static_feat_dim = configs.static_feat_dim
        head_input_dim = configs.d_model + configs.static_feat_dim
        
        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, configs.head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static):
        """
        前向传播
        """
        # (B, L, C) -> (B, n_vars, N, D)
        time_series_repr = self.backbone(x_dynamic) 
        
        # (B, n_vars, N, D) -> (B, D)
        time_series_repr_flat = time_series_repr.mean(dim=1)[:, -1, :]
        
        # (B, D) 和 (B, M) -> (B, D + M)
        combined_features = torch.cat([time_series_repr_flat, x_static], dim=1)
        
        # (B, D + M) -> (B, 1)
        prediction = self.regression_head(combined_features)
        
        return prediction
