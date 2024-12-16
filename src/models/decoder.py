import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=True)

from src.models.attention import MultiheadAttention


@dataclass
class DecoderOutput:
    output: Tensor
    ca_weight: Tensor
    sa_weight: Tensor
    content_output: Tensor | None = None


class DecoderBase(nn.Module, ABC):
    def __init__(
        self,
        # Decoder
        d_model: int,
        n_layers: int,
        decoder_layer: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = _get_clones(decoder_layer, n_layers)
        self.num_layers = n_layers
        self.norm = nn.LayerNorm(d_model)

    @abstractmethod
    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput: ...


class PositionalDecoder(DecoderBase):
    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput:
        output = positional_query

        outputs = []
        ca_weights = []
        sa_weights = []

        for layer in self.layers:
            decoder_layer_output = layer(
                video_features=video_features,
                diagram_features=diagram_features,
                positional_query=output,
                video_positional_encoding=video_positional_encoding,
                diagram_positional_encoding=diagram_positional_encoding,
                video_key_padding_mask=video_key_padding_mask,
                diagram_query_key_padding_mask=diagram_query_key_padding_mask,
                video_attention_mask=video_attention_mask,
                diagram_query_attention_mask=diagram_query_attention_mask,
            )
            output = decoder_layer_output.output
            outputs.append(self.norm(output))
            ca_weights.append(decoder_layer_output.ca_weight)
            sa_weights.append(decoder_layer_output.sa_weight)

        return DecoderOutput(
            torch.stack(outputs),
            torch.stack(ca_weights),
            torch.stack(sa_weights),
        )


class ContentDecoder(DecoderBase):
    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput:
        output = diagram_features

        outputs = []
        ca_weights = []
        sa_weights = []

        for layer in self.layers:
            decoder_layer_output = layer(
                video_features=video_features,
                diagram_features=output,
                positional_query=positional_query,
                video_positional_encoding=video_positional_encoding,
                diagram_positional_encoding=diagram_positional_encoding,
                video_key_padding_mask=video_key_padding_mask,
                diagram_query_key_padding_mask=diagram_query_key_padding_mask,
                video_attention_mask=video_attention_mask,
                diagram_query_attention_mask=diagram_query_attention_mask,
            )
            output = decoder_layer_output.output
            outputs.append(self.norm(output))
            ca_weights.append(decoder_layer_output.ca_weight)
            sa_weights.append(decoder_layer_output.sa_weight)

        return DecoderOutput(
            torch.stack(outputs),
            torch.stack(ca_weights),
            torch.stack(sa_weights),
        )


class TwinTowerIndependentDecoder(DecoderBase):
    def __init__(
        self,
        content_decoder_layer: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.content_layers = _get_clones(content_decoder_layer, self.num_layers)
        self.content_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput:
        output = positional_query

        outputs = []
        ca_weights = []
        sa_weights = []

        for layer in self.layers:
            decoder_layer_output = layer(
                video_features=video_features,
                diagram_features=diagram_features,
                positional_query=output,
                video_positional_encoding=video_positional_encoding,
                diagram_positional_encoding=diagram_positional_encoding,
                video_key_padding_mask=video_key_padding_mask,
                diagram_query_key_padding_mask=diagram_query_key_padding_mask,
                video_attention_mask=video_attention_mask,
                diagram_query_attention_mask=diagram_query_attention_mask,
            )
            output = decoder_layer_output.output
            outputs.append(self.norm(output))
            ca_weights.append(decoder_layer_output.ca_weight)
            sa_weights.append(decoder_layer_output.sa_weight)

        output = diagram_features
        content_outputs = []
        for layer in self.content_layers:
            decoder_layer_output = layer(
                video_features=video_features,
                diagram_features=output,
                positional_query=positional_query,
                video_positional_encoding=video_positional_encoding,
                diagram_positional_encoding=diagram_positional_encoding,
                video_key_padding_mask=video_key_padding_mask,
                diagram_query_key_padding_mask=diagram_query_key_padding_mask,
                video_attention_mask=video_attention_mask,
                diagram_query_attention_mask=diagram_query_attention_mask,
            )
            output = decoder_layer_output.output
            content_outputs.append(self.content_norm(output))

        return DecoderOutput(
            torch.stack(outputs),
            torch.stack(ca_weights),
            torch.stack(sa_weights),
            torch.stack(content_outputs),
        )


class TwinTowerDecoder(DecoderBase):
    def __init__(
        self,
        content_decoder_layer: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.content_layers = _get_clones(content_decoder_layer, self.num_layers)
        self.content_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput:
        output = positional_query
        content_output = diagram_features

        outputs = []
        content_outputs = []
        ca_weights = []
        sa_weights = []

        for layer, content_layer in zip(self.layers, self.content_layers):
            decoder_layer_output = layer(
                video_features=video_features,
                diagram_features=content_output,
                positional_query=output,
                video_positional_encoding=video_positional_encoding,
                diagram_positional_encoding=diagram_positional_encoding,
                video_key_padding_mask=video_key_padding_mask,
                diagram_query_key_padding_mask=diagram_query_key_padding_mask,
                video_attention_mask=video_attention_mask,
                diagram_query_attention_mask=diagram_query_attention_mask,
            )

            output = decoder_layer_output.output
            outputs.append(self.norm(output))
            ca_weights.append(decoder_layer_output.ca_weight)
            sa_weights.append(decoder_layer_output.sa_weight)

            content_decoder_layer_output = content_layer(
                video_features=video_features,
                diagram_features=content_output,
                positional_query=output,
                video_positional_encoding=video_positional_encoding,
                diagram_positional_encoding=diagram_positional_encoding,
                video_key_padding_mask=video_key_padding_mask,
                diagram_query_key_padding_mask=diagram_query_key_padding_mask,
                video_attention_mask=video_attention_mask,
                diagram_query_attention_mask=diagram_query_attention_mask,
            )
            content_output = content_decoder_layer_output.output
            content_outputs.append(self.content_norm(content_output))

        return DecoderOutput(
            torch.stack(outputs),
            torch.stack(ca_weights),
            torch.stack(sa_weights),
            torch.stack(content_outputs),
        )


class DecoderLayerBase(nn.Module, ABC):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        p_dropout: float,
        f_activation: Literal["relu", "gelu", "glu"],
        sa_fuse: Literal["add", "concat"],
        ca_fuse: Literal["add", "concat"],
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_feedforward = d_feedforward
        self.p_dropout = p_dropout
        self.sa_fuse = sa_fuse
        self.ca_fuse = ca_fuse

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

    @abstractmethod
    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput: ...


class PositionalDecoderLayer(DecoderLayerBase):
    def __init__(
        self,
        # exp parameters
        diagram_pe_fuse_type: Literal[
            "none",  # do not use the diagram PE
            "add",  # add diagram PE to the positional query in cross attention
            "concat",  # concatenate diagram PE to the positional query in cross attention, then repeat the video PE
            "concat_porj",  # concatenate diagram PE to the positional query in cross attention, then project
            "sa_concat_proj",  # concatenate diagram PE to the positional query in self attention, then project
        ] = "none",
        sa_v: Literal[
            "positional_query",  # use the positional query as the value in self attention
            "add",  # add the positional query to the diagram features as the value in self attention
            "concat_proj",  # concatenate the positional query and the diagram features, then project
        ] = "positional_query",
        ca_v: Literal[
            "video_pe",  # use the video PE as the value in cross attention
            "add",  # add the video PE to the video features as the value in cross attention
            "concat_proj",  # concatenate the video PE and the video features, then project
        ] = "video_pe",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diagram_pe_fuse_type = diagram_pe_fuse_type
        if diagram_pe_fuse_type == "concat":
            if self.ca_fuse == "add":
                raise NotImplementedError
            self.ca_k_pos2_proj = nn.Linear(self.d_model, self.d_model)
            self.ca_q_pos2_proj = nn.Linear(self.d_model, self.d_model)
            self.cross_attn = MultiheadAttention(
                self.d_model * 3, self.n_head, dropout=self.p_dropout, vdim=self.d_model
            )
        elif diagram_pe_fuse_type == "concat_proj":
            self.concat_proj = nn.Linear(self.d_model * 2, self.d_model)
        elif diagram_pe_fuse_type == "sa_concat_proj":
            self.sa_query_diagram_pe_concat_proj = nn.Linear(self.d_model * 2, self.d_model)

        self.sa_v = sa_v
        self.ca_v = ca_v
        if sa_v == "concat_proj":
            self.sa_v_proj = nn.Linear(self.d_model * 2, self.d_model)
        if ca_v == "concat_proj":
            self.ca_v_proj = nn.Linear(self.d_model * 2, self.d_model)

    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput:
        # ========== Begin of Self-Attention =============
        if self.diagram_pe_fuse_type == "sa_concat_proj":
            positional_query = self.sa_query_diagram_pe_concat_proj(
                torch.cat([positional_query, diagram_positional_encoding], dim=2)
            )

        q_content = self.sa_q_content_proj(diagram_features)
        q_pos = self.sa_q_pos_proj(positional_query)
        k_content = self.sa_k_content_proj(diagram_features)
        k_pos = self.sa_k_pos_proj(positional_query)
        if self.sa_v == "positional_query":
            v = self.sa_v_proj(positional_query)
        elif self.sa_v == "add":
            v = self.sa_v_proj(positional_query + diagram_features)
        elif self.sa_v == "concat_proj":
            v = self.sa_v_proj(torch.cat([positional_query, diagram_features], dim=2))
        else:
            raise NotImplementedError(f"Unknown sa_v type: {self.sa_v}.")

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

        positional_query2, self_att = self.self_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=diagram_query_attention_mask,
            key_padding_mask=diagram_query_key_padding_mask,
        )
        # ========== End of Self-Attention =============
        positional_query = positional_query + self.dropout1(positional_query2)
        positional_query = self.norm1(positional_query)

        # ========== Begin of Cross-Attention =============
        q_content = self.ca_q_content_proj(diagram_features)
        if self.diagram_pe_fuse_type == "add":
            q_pos = self.ca_q_pos_proj(positional_query + diagram_positional_encoding)
        elif self.diagram_pe_fuse_type == "concat_proj":
            q_pos = self.ca_q_pos_proj(
                self.concat_proj(torch.cat([positional_query, diagram_positional_encoding], dim=2))
            )
        else:
            q_pos = self.ca_q_pos_proj(positional_query)
        k_content = self.ca_k_content_proj(video_features)
        k_pos = self.ca_k_pos_proj(video_positional_encoding)

        if self.diagram_pe_fuse_type == "concat":
            q_pos2 = self.ca_q_pos2_proj(positional_query)
            k_pos2 = self.ca_k_pos2_proj(video_positional_encoding)

        if self.ca_v == "video_pe":
            v = self.ca_v_proj(video_positional_encoding)
        elif self.ca_v == "add":
            v = self.ca_v_proj(video_positional_encoding + video_features)
        elif self.ca_v == "concat_proj":
            v = self.ca_v_proj(torch.cat([video_positional_encoding, video_features], dim=2))

        if self.ca_fuse == "add":
            q = q_content + q_pos
            k = k_content + k_pos
        elif self.ca_fuse == "concat":
            num_queries, bs, n_model = q_content.shape
            num_keys, _, _ = k_content.shape
            q_content = q_content.view(num_queries, bs, self.n_head, n_model // self.n_head)
            q_pos = q_pos.view(num_queries, bs, self.n_head, n_model // self.n_head)

            if self.diagram_pe_fuse_type == "concat":
                q_pos2 = q_pos2.view(num_queries, bs, self.n_head, n_model // self.n_head)
                q = torch.cat([q_content, q_pos, q_pos2], dim=3).view(num_queries, bs, n_model * 3)
            else:
                q = torch.cat([q_content, q_pos], dim=3).view(num_queries, bs, n_model * 2)

            k_content = k_content.view(num_keys, bs, self.n_head, n_model // self.n_head)
            k_pos = k_pos.view(num_keys, bs, self.n_head, n_model // self.n_head)

            if self.diagram_pe_fuse_type == "concat":
                k_pos2 = k_pos2.view(num_keys, bs, self.n_head, n_model // self.n_head)
                k = torch.cat([k_content, k_pos, k_pos2], dim=3).view(num_keys, bs, n_model * 3)
            else:
                k = torch.cat([k_content, k_pos], dim=3).view(num_keys, bs, n_model * 2)

        positional_query2, cross_att = self.cross_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=video_attention_mask,
            key_padding_mask=video_key_padding_mask,
        )
        # ========== End of Cross-Attention =============

        positional_query = positional_query + self.dropout2(positional_query2)
        positional_query = self.norm2(positional_query)
        positional_query2 = self.linear2(self.dropout(self.activation(self.linear1(positional_query))))
        positional_query = positional_query + self.dropout3(positional_query2)
        positional_query = self.norm3(positional_query)
        return DecoderOutput(positional_query, cross_att, self_att)


class ContentDecoderLayer(DecoderLayerBase):
    def __init__(
        self,
        # exp parameters
        diagram_pe_fuse_type: Literal[
            "none",  # do not use the diagram PE
            "add",  # add diagram PE to the positional query in cross attention
            "concat_porj",  # concatenate diagram PE to the positional query in cross attention, then project
            "sa_concat_proj",  # concatenate diagram PE to the positional query in self attention, then project
        ] = "concat_porj",
        sa_v: Literal[
            "diagram_feature",  # use the diagram feature as the value in self attention
            "add",  # add the positional query to the diagram features as the value in self attention
            "concat_proj",  # concatenate the positional query and the diagram features, then project
        ] = "diagram_feature",
        ca_v: Literal[
            "video_feature",  # use the video features as the value in cross attention
            "add",  # add the video PE to the video features as the value in cross attention
            "concat_proj",  # concatenate the video PE and the video features, then project
        ] = "video_feature",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diagram_pe_fuse_type = diagram_pe_fuse_type

        if diagram_pe_fuse_type == "concat_proj":
            self.concat_proj = nn.Linear(self.d_model * 2, self.d_model)
        elif diagram_pe_fuse_type == "sa_concat_proj":
            self.sa_diagram_feature_pe_concat_proj = nn.Linear(self.d_model * 2, self.d_model)

        self.sa_v = sa_v
        self.ca_v = ca_v
        if sa_v == "concat_proj":
            self.sa_v_proj = nn.Linear(self.d_model * 2, self.d_model)
        if ca_v == "concat_proj":
            self.ca_v_proj = nn.Linear(self.d_model * 2, self.d_model)

    def forward(
        self,
        video_features: Tensor,
        diagram_features: Tensor,
        positional_query: Tensor,
        video_positional_encoding: Tensor,
        diagram_positional_encoding: Tensor,
        video_key_padding_mask: Tensor | None = None,
        diagram_query_key_padding_mask: Tensor | None = None,
        video_attention_mask: Tensor | None = None,
        diagram_query_attention_mask: Tensor | None = None,
    ) -> DecoderOutput:
        # ========== Begin of Self-Attention =============
        if self.diagram_pe_fuse_type == "sa_concat_proj":
            diagram_features = self.sa_diagram_feature_pe_concat_proj(
                torch.cat([diagram_features, diagram_positional_encoding], dim=2)
            )

        q_content = self.sa_q_content_proj(diagram_features)
        q_pos = self.sa_q_pos_proj(positional_query)
        k_content = self.sa_k_content_proj(diagram_features)
        k_pos = self.sa_k_pos_proj(positional_query)
        if self.sa_v == "diagram_feature":
            v = self.sa_v_proj(diagram_features)
        elif self.sa_v == "add":
            v = self.sa_v_proj(positional_query + diagram_features)
        elif self.sa_v == "concat_proj":
            v = self.sa_v_proj(torch.cat([positional_query, diagram_features], dim=2))

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

        diagram_features2, self_att = self.self_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=diagram_query_attention_mask,
            key_padding_mask=diagram_query_key_padding_mask,
        )
        # ========== End of Self-Attention =============

        diagram_features = diagram_features + self.dropout1(diagram_features2)
        diagram_features = self.norm1(diagram_features)

        # ========== Begin of Cross-Attention =============
        if self.diagram_pe_fuse_type == "add":
            positional_query = positional_query + diagram_positional_encoding
        elif self.diagram_pe_fuse_type == "concat_proj":
            positional_query = self.concat_proj(torch.cat([positional_query, diagram_positional_encoding], dim=2))

        q_content = self.ca_q_content_proj(diagram_features)
        q_pos = self.ca_q_pos_proj(positional_query)
        k_content = self.ca_k_content_proj(video_features)
        k_pos = self.ca_k_pos_proj(video_positional_encoding)

        if self.ca_v == "video_feature":
            v = self.ca_v_proj(video_features)
        elif self.ca_v == "add":
            v = self.ca_v_proj(video_features + video_positional_encoding)
        elif self.ca_v == "concat_proj":
            v = self.ca_v_proj(torch.cat([video_features, video_positional_encoding], dim=2))

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

        video_features2, cross_att = self.cross_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=video_attention_mask,
            key_padding_mask=video_key_padding_mask,
        )
        # ========== End of Cross-Attention =============

        video_features = diagram_features + self.dropout2(video_features2)
        video_features = self.norm2(video_features)
        video_features2 = self.linear2(self.dropout(self.activation(self.linear1(video_features))))
        video_features = video_features + self.dropout3(video_features2)
        video_features = self.norm3(video_features)
        return DecoderOutput(video_features, cross_att, self_att)


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
