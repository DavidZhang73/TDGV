# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

Copy-paste from DETR with modifications:
    * refine type hints
    * refine variable names
    * remove normalize_before with False as default
"""

import copy
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.attention import MultiheadAttention
from src.models.common import MLP


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_feedforward: int = 2048,
        p_dropout: float = 0.1,
        f_activation: Literal["relu", "gelu", "glu"] = "relu",
        return_intermediate_dec: bool = True,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_feedforward, p_dropout, f_activation)
        # TODO: normalize_before == False, hence no encoder_norm
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, n_encoder_layers, None)

        decoder_layer = TransformerDecoderLayer(d_model, n_head, d_feedforward, p_dropout, f_activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, n_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        )

        # self._reset_parameters()

        self.d_model = d_model
        self.n_head = n_head

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,  # N x B x D
        tgt: Tensor,  # L x B x D
        src_mask: Tensor = None,  # B x N
        tgt_mask: Tensor = None,  # B x L
        src_pos: Tensor = None,  # N x B x D
        tgt_pos: Tensor = None,  # L x B x D
    ):
        memory, _ = self.encoder(
            src,
            src_key_padding_mask=src_mask,
            src_pos=src_pos,
        )
        memory = memory[-1]
        output, att_weights, self_att_weights = self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
            tgt_pos=tgt_pos,
            memory_pos=src_pos,
        )
        return output, memory, att_weights


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        n_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, n_layers)
        self.n_layers = n_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        src_pos: Tensor | None = None,
    ):
        output = src

        intermediate = []
        self_att_weights = []

        for layer in self.layers:
            output, self_att = layer(
                output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, src_pos=src_pos
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                self_att_weights.append(self_att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(self_att_weights)

        return output.unsqueeze(0), self_att.unsqueeze(0)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        n_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, n_layers)
        self.n_layers = n_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_pos: Tensor | None = None,
        memory_pos: Tensor | None = None,
    ):
        output = tgt

        intermediate = []
        att_weights = []
        self_att_weights = []

        for layer in self.layers:
            output, att, self_att = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_pos=tgt_pos,
                memory_pos=memory_pos,
            )
            if self.return_intermediate:
                intermediate.append(output)
                att_weights.append(att)
                self_att_weights.append(self_att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(att_weights), torch.stack(self_att_weights)

        return output.unsqueeze(0), att.unsqueeze(0), self_att.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int = 2048,
        p_dropout: float = 0.1,
        f_activation: Literal["relu", "gelu", "glu"] = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=p_dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)

        self.activation = _get_activation_fn(f_activation)

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        src_pos: Tensor | None = None,
    ):
        q = k = self.with_pos_embed(src, src_pos)
        src2, self_att = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            average_attn_weights=False,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, self_att


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int = 2048,
        p_dropout: float = 0.1,
        f_activation: Literal["relu", "gelu", "glu"] = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=p_dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=p_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)
        self.dropout3 = nn.Dropout(p_dropout)

        self.activation = _get_activation_fn(f_activation)

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_pos: Tensor | None = None,
        memory_pos: Tensor | None = None,
    ):
        q = k = self.with_pos_embed(tgt, tgt_pos)
        tgt2, self_att = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att = self.multihead_attn(
            query=self.with_pos_embed(tgt, tgt_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att, self_att


class TransformerConditionalDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        n_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
        d_model: int = 768,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = _get_clones(decoder_layer, n_layers)
        self.num_layers = n_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.kaiming_uniform_(p, nonlinearity="relu")

    def _compute_sinusoidal_pe(
        self, pos_tensor: Tensor, temperature: float = 10000, scale: float = 2 * torch.pi
    ) -> Tensor:
        """Compute positional embeddings for point.

        Parameters:
        -----------
        pos_tensor: Tensor
            Coordinates of 1d points (x) normalized to (0, 1). The shape is (n_q, bs, 1).
        temperature: float, Default: 10000.
            The temperature parameter in sinusoidal functions.

        Returns:
        --------
        pos: Tensor
            Sinusoidal positional embeddings of shape (n_q, bs, d_model).
        """
        dim = self.d_model
        dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
        dim_t = temperature ** (2 * (dim_t // 2) / dim)
        embed = pos_tensor[:, :, 0] * scale
        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        memory_pos: Tensor | None = None,
        tgt_pos: Tensor | None = None,  # same as query_pos
        tgt_pos2: Tensor | None = None,
    ):
        output = tgt

        intermediate = []
        att_weights = []
        self_att_weights = []

        for layer in self.layers:
            output, att, self_att = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=memory_pos,
                tgt_pos=tgt_pos,
                tgt_pos2=tgt_pos2,
            )
            if self.norm is not None:
                output = self.norm(output)
            if self.return_intermediate:
                intermediate.append(output)
                att_weights.append(att)
                self_att_weights.append(self_att)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(att_weights),
                torch.stack(self_att_weights) if self_att_weights[0] is not None else None,
            )

        return output.unsqueeze(0), att.unsqueeze(0), self_att.unsqueeze(0)


class TransformerConditionalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 6,
        d_feedforward: int = 2048,
        p_dropout: float = 0.1,
        f_activation: Literal["relu", "gelu", "glu"] = "relu",
        sa_fuse: Literal["add", "concat"] = "concat",
        ca_fuse: Literal["add", "concat"] = "concat",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        # Decoder Self-Attention
        self.sa_q_content_proj = nn.Linear(d_model, d_model)
        self.sa_q_pos_proj = nn.Linear(d_model, d_model)
        self.sa_k_content_proj = nn.Linear(d_model, d_model)
        self.sa_k_pos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        if sa_fuse == "concat":
            _d_model = d_model * 2
        elif sa_fuse == "add":
            _d_model = d_model
        else:
            raise NotImplementedError(f"Unknown self attention fusion type: {sa_fuse}.")
        self.self_attn = MultiheadAttention(_d_model, n_head, dropout=p_dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_q_content_proj = nn.Linear(d_model, d_model)
        self.ca_q_pos_proj = nn.Linear(d_model, d_model)
        self.ca_k_content_proj = nn.Linear(d_model, d_model)
        self.ca_k_pos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        # self.ca_q_pos_sine_proj = nn.Linear(d_model, d_model)
        if ca_fuse == "concat":
            _d_model = d_model * 2
        elif ca_fuse == "add":
            _d_model = d_model
        else:
            raise NotImplementedError(f"Unknown cross attention fusion type: {ca_fuse}.")
        self.cross_attn = MultiheadAttention(_d_model, n_head, dropout=p_dropout, vdim=d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)
        self.dropout3 = nn.Dropout(p_dropout)

        self.activation = _get_activation_fn(f_activation)
        self.sa_fuse = sa_fuse
        self.ca_fuse = ca_fuse

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        memory_pos: Tensor = None,
        tgt_pos: Tensor = None,
        tgt_pos2: Tensor | None = None,
    ):
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x dim

        q_content = self.sa_q_content_proj(tgt_pos)  # target is the input of the first decoder layer.
        q_pos = self.sa_q_pos_proj(tgt)
        k_content = self.sa_k_content_proj(tgt_pos)
        k_pos = self.sa_k_pos_proj(tgt)
        v = self.sa_v_proj(tgt)

        if self.sa_fuse == "add":
            q = q_content + q_pos
            k = k_content + k_pos
        elif self.sa_fuse == "concat":
            num_queries, bs, n_model = q_content.shape
            num_keys, _, _ = k_content.shape
            q_content = q_content.view(num_queries, bs, self.n_head, n_model // self.n_head)
            q_pos = q_pos.view(num_queries, bs, self.n_head, n_model // self.n_head)
            q = torch.cat([q_content, q_pos], dim=3).view(num_queries, bs, n_model * 2)
            k_content = k_content.view(num_keys, bs, self.n_head, n_model // self.n_head)
            k_pos = k_pos.view(num_keys, bs, self.n_head, n_model // self.n_head)
            k = torch.cat([k_content, k_pos], dim=3).view(num_keys, bs, n_model * 2)

        tgt2, self_att = self.self_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x D
        # q_content = self.ca_q_content_proj(tgt)
        # q_pos = self.ca_q_pos_proj(tgt_pos)
        # swap q_content and q_pos
        q_content = self.ca_q_content_proj(tgt_pos)
        q_pos = self.ca_q_pos_proj(tgt + tgt_pos2)
        k_content = self.ca_k_content_proj(memory)
        k_pos = self.ca_k_pos_proj(memory_pos)

        # !!! important change !!!
        # v = self.ca_v_proj(memory)
        # v = self.ca_v_proj(memory + memory_pos)
        v = self.ca_v_proj(memory_pos)

        if self.ca_fuse == "add":
            q = q_content + q_pos
            k = k_content + k_pos
        elif self.ca_fuse == "concat":
            num_queries, bs, n_model = q_content.shape
            num_keys, _, _ = k_content.shape
            q_content = q_content.view(num_queries, bs, self.n_head, n_model // self.n_head)
            q_pos = q_pos.view(num_queries, bs, self.n_head, n_model // self.n_head)
            q = torch.cat([q_content, q_pos], dim=3).view(num_queries, bs, n_model * 2)
            k_content = k_content.view(num_keys, bs, self.n_head, n_model // self.n_head)
            k_pos = k_pos.view(num_keys, bs, self.n_head, n_model // self.n_head)
            k = torch.cat([k_content, k_pos], dim=3).view(num_keys, bs, n_model * 2)

        tgt2, att = self.cross_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att, self_att


def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: Literal["relu", "gelu", "glu"]):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
