import torch
import torch.nn as nn
from models.TimeMixer import Model as TimeMixer_Base_Model

class Model(TimeMixer_Base_Model):
    """
    TimeMixer model adapted for the yield regression task.
    This version uses the refactored `encode` method from the base class.
    """
    def __init__(self, configs):
        # We need to set task_name to a forecasting one to inherit __init__
        original_task_name = configs.task_name
        configs.task_name = 'long_term_forecast'
        super().__init__(configs)
        configs.task_name = original_task_name

        self.d_model = configs.d_model
        static_feat_dim = getattr(configs, 'static_feat_dim', 0)
        head_mlp_dim = getattr(configs, 'head_mlp_dim', 128)

        head_input_dim = self.d_model + static_feat_dim

        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static):
        x_enc = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        B, L, C = x_enc.shape
        x_mark_enc = None

        # Call the refactored encode method from the base class
        enc_out_list, _ = self.encode(x_enc, x_mark_enc)

        # We take the output from the original scale (the first in the list)
        enc_out = enc_out_list[0]

        if self.channel_independence:
            # Reshape back if channel independent
            enc_out = enc_out.reshape(B, C, L).permute(0, 2, 1)

        # Pooling: Get a single vector representation for the time series.
        # Using the last time step is a common approach.
        ts_repr_pooled = enc_out[:, -1, :]

        # Fusion with static features
        combined_features = torch.cat([ts_repr_pooled, x_static], dim=1)

        # Regression
        prediction = self.regression_head(combined_features)

        return prediction, None, None
