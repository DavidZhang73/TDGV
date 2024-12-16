import math

import torch
from torch import nn


class PositionEmbeddingSineDETR(nn.Module):
    """This is a more standard version of the position embedding.

    Very similar to the one used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingSineLVTR(nn.Module):
    """This is a more standard version of the position embedding."""

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = True, scale: float = 2 * math.pi
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Positional Embedding.

        Args:
            x (torch.Tensor): (batch_size, L, d)
            mask (torch.Tensor): (batch_size, L), with 0 as valid, 1 as ignored

        Returns:
            torch.Tensor: _description_
        """
        x_embed = (~mask).cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(
            2
        )  # (bsz, L, num_pos_feats*2)
        # import ipdb; ipdb.set_trace()
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int = 5000,
        temperature: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.temperature = temperature
        index = torch.arange(self.d_model, dtype=torch.float32)
        denominator = self.temperature ** (2 * (index // 2) / self.d_model)  # 10000^(2i/d)
        pos = torch.arange(self.max_length, dtype=torch.float32).unsqueeze(1) + 1
        pe = pos / denominator
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        self.register_buffer("pe", pe)

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None):
        """Generate positional encoding for features with mask.

        Args:
            features (torch.Tensor): Batched and padded features with shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Same binary mask as `key_padding_mask` in `nn.MultiheadAttention` with shape
                (batch_size, seq_len).
                `True` indicates that the corresponding key will be ignored. If None, no mask will be applied.
                Defaults to None.

        Returns:
            Torch.Tensor: The positional encoding with shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = features.shape
        ret = self.pe.repeat(batch_size, 1, 1)
        ret = ret[:, :seq_len, :]
        if mask is None:
            return ret
        return ret * (~mask).unsqueeze(-1)
