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
        # 我们的回归头是一个 MLP, 它接收:
        # a) PatchTST 的输出 (configs.d_model)
        # b) 静态特征 (configs.static_feat_dim)
        
        self.static_feat_dim = configs.static_feat_dim
        
        head_input_dim = configs.d_model + configs.static_feat_dim
        
        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, configs.head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.head_mlp_dim, 1) # 最终输出1个值 (产量)
        )

    def forward(self, x_dynamic, x_static):
        """
        前向传播
        
        Args:
            x_dynamic (torch.Tensor): 动态时序数据 (B, L, C)
                                      B=batch_size, L=seq_len(365), C=n_vars(21)
            x_static (torch.Tensor):  静态特征数据 (B, M)
                                      B=batch_size, M=static_feat_dim(65)
        
        Returns:
            torch.Tensor: 预测的产量 (B, 1)
        """
        
        # 1. 通过 TSLib 骨干网络
        # x_dynamic: (B, L, C)
        # B = batch size
        # L = seq_len (e.g., 365)
        # C = n_vars (e.g., 21)
        
        # configs.seq_len 必须在 .sh 脚本中设置为 L (365)
        # configs.enc_in 必须在 .sh 脚本中设置为 C (21)
        
        # TSLib 的 PatchTST 返回 (B, N, D)
        # B = batch_size
        # N = num_patches
        # D = d_model
        # 我们只取第一个 [CLS] token (如果使用) 或者平均池化
        
        # (B, L, C) -> (B, N, D)
        time_series_repr = self.backbone(x_dynamic) 
        
        # (B, N, D) -> (B, D)
        # 我们使用最后一个 patch 的 embedding 作为序列的总结表征
        # (或者使用平均池化: time_series_repr.mean(dim=1))
        time_series_repr_flat = time_series_repr[:, -1, :]
        
        # 2. 融合静态特征
        # (B, D) 和 (B, M) -> (B, D + M)
        combined_features = torch.cat([time_series_repr_flat, x_static], dim=1)
        
        # 3. 通过回归头
        # (B, D + M) -> (B, 1)
        prediction = self.regression_head(combined_features)
        
        return prediction
