import torch
import torch.nn as nn
from models.TimeXer import Model as TimeXer_Base


class Model(nn.Module):
    """
    TimeXer model adapted for the yield regression task, with support for variable sequence lengths
    and proper handling of static (exogenous) and dynamic (endogenous) features.
    """

    def __init__(self, configs):
        super().__init__()
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model

        # Instantiate the base TimeXer model
        # We need to tell the base model about the dynamic and static feature dimensions
        # Following convention, `enc_in` from args is for dynamic features.
        # `static_feat_dim` from args is for static features.
        base_configs = configs

        # BUG WORKAROUND: The original TimeXer's `ex_embedding` is incorrectly initialized with `enc_in`
        # instead of `dec_in`. To fix this, we temporarily set `enc_in` to the static feature dimension
        # before initializing the backbone, and then restore it.
        original_enc_in = base_configs.enc_in
        base_configs.enc_in = configs.static_feat_dim # Temporarily set for ex_embedding
        base_configs.dec_in = configs.static_feat_dim # Static feature dim (correct for other parts if used)
        base_configs.c_out = 1 # Not used in encoder-only, but good practice

        # Temporarily change task_name to initialize backbone, then restore it
        original_task_name = base_configs.task_name
        base_configs.task_name = 'long_term_forecast'

        self.backbone = TimeXer_Base(base_configs)

        # Restore original enc_in for consistency
        base_configs.enc_in = original_enc_in
        base_configs.task_name = original_task_name

        # Regression head
        # The output of TimeXer backbone's encoder is based on the [GLB] token, which has shape (B, d_model)
        # We will take the pooled output from the backbone and pass it to the regression head.
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, configs.head_mlp_dim),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.head_mlp_dim, 1)
        )

    def forward(self, x_dynamic, x_static, attention_mask=None):
        """
        Forward pass for the regression task.
        x_dynamic: [B, L, C_dynamic] - Padded dynamic features.
        x_static: [B, C_static] - Static features.
        attention_mask: [B, L] - Boolean mask for the dynamic features.
        """
        # Convert NaNs to zero to ensure numerical stability
        x_dynamic = torch.nan_to_num(x_dynamic)
        x_static = torch.nan_to_num(x_static)

        # TimeXer's base `forecast` method expects `x_enc` and `x_mark_enc`.
        # We map our inputs accordingly:
        # x_enc -> x_dynamic
        # x_mark_enc -> x_static (which will be treated as the exogenous 'cross' features inside)

        # The base TimeXer model does not directly accept an attention_mask.
        # However, its underlying Encoder and Attention layers DO.
        # We need to manually pass the mask to the backbone's encoder.

        # 1. Normalization (from base model)
        if self.backbone.use_norm:
            means = x_dynamic.mean(1, keepdim=True).detach()
            x_dynamic = x_dynamic - means
            stdev = torch.sqrt(torch.var(x_dynamic, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_dynamic /= stdev

        # 2. Embedding
        # The TimeXer backbone will handle embedding of dynamic and static features separately.
        en_embed, n_vars = self.backbone.en_embedding(x_dynamic.permute(0, 2, 1))
        # Pass static features to the exogenous embedding part
        ex_embed = self.backbone.ex_embedding(x_static.unsqueeze(1), None) # Unsqueeze to add a time dimension of 1

        # 3. Create patch-level attention mask
        if attention_mask is not None:
            # The attention is applied on patches, so we need to create a patch-level mask.
            # A simple way is to check if a patch contains any unmasked data.
            num_patches = en_embed.shape[1] -1 # Exclude GLB token

            # Reshape mask to be patch-wise
            mask_patches = attention_mask.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)

            # A patch is valid if it contains at least one 'True' value
            patch_mask = torch.any(mask_patches, dim=-1) # Shape: [B, num_patches]

            # Add a True for the [GLB] token at the end
            glb_mask = torch.ones(patch_mask.shape[0], 1, dtype=torch.bool, device=patch_mask.device)
            patch_mask = torch.cat([patch_mask, glb_mask], dim=1) # Shape: [B, num_patches + 1]

            # The encoder expects mask of shape [B, 1, L, L] or similar for multi-head attention
            # For our purpose, a 2D mask [B, num_patches+1] should be broadcast correctly by PyTorch attention layers.
            # We need to reshape it to be compatible with multi-head attention: [B, 1, 1, num_patches+1]
            patch_mask = patch_mask.unsqueeze(1).unsqueeze(2)

        else:
            patch_mask = None

        # 4. Encoder
        enc_out = self.backbone.encoder(en_embed, ex_embed, x_mask=patch_mask)

        # 5. Pooling
        # Reshape and extract the [GLB] token's representation
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # Average the [GLB] token output across variables/channels
        ts_repr_pooled = enc_out[:, :, -1, :].mean(dim=1)

        # 6. Regression
        prediction = self.regression_head(ts_repr_pooled)

        # Match the expected output format of the experiment runner
        return prediction, None, None
