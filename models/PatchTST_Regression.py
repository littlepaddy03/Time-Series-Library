import torch
import torch.nn as nn
from models.PatchTST import PatchTST_backbone

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
        # (B, L, C) -> (B, N, D)
        time_series_repr = self.backbone(x_dynamic) 
        
        # (B, N, D) -> (B, D)
        time_series_repr_flat = time_series_repr[:, -1, :]
        
        # (B, D) 和 (B, M) -> (B, D + M)
        combined_features = torch.cat([time_series_repr_flat, x_static], dim=1)
        
        # (B, D + M) -> (B, 1)
        prediction = self.regression_head(combined_features)
        
        return prediction
