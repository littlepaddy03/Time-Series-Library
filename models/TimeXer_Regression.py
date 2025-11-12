import torch
import torch.nn as nn
from models.TimeXer import Model as TimeXer_Base_Model

class Model(TimeXer_Base_Model):
    """
    TimeXer model adapted for the yield regression task.
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
        # TimeXer expects x_enc and x_mark_enc.
        # Our data loader provides x_dynamic for x_enc.
        # We'll create a dummy x_mark_enc as it's not used in our regression task.
        x_enc = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        B, L, C = x_enc.shape
        x_mark_enc = torch.zeros([B, L, 0]).to(x_enc.device)

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        # The original TimeXer has two embedding branches, one for the main series (en) and one for exogenous (ex).
        # In our case, x_enc holds all dynamic features.
        # We will need to decide how to partition them if we want to use the exogenous branch meaningfully.
        # For a simple baseline, we'll pass all of x_enc to the main branch (`en_embedding`)
        # and the time features (x_mark_enc) to the exogenous branch (`ex_embedding`).
        # As our x_mark_enc is empty, ex_embed will be based on positional encoding only.

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(torch.zeros_like(x_enc), x_mark_enc) # Pass zeros as placeholder for series part of ex_embed

        # Encoder
        enc_out = self.encoder(en_embed, ex_embed)

        # Reshape and Pool
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # We'll take the output of the [GLB] token for pooling
        ts_repr_pooled = enc_out[:, :, -1, :].mean(dim=1)

        # Fusion with static features
        combined_features = torch.cat([ts_repr_pooled, x_static], dim=1)

        # Regression
        prediction = self.regression_head(combined_features)

        # The model should return prediction and optionally other info like in MoE.
        # For a baseline, returning just the prediction is fine.
        # To match the experiment runner's expected tuple output for regression, we return None placeholders.
        return prediction, None, None
