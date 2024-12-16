import logging
import os
import random
from typing import Literal

import PIL
import pytorch_lightning as pl
import torch
import wandb
from filelock import FileLock
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import MetricCollection

from src.metrics import RecallAtIoUSegmentMap
from src.models.common import MLP, Projection
from src.models.decoder import (
    DecoderOutput,
    PositionalDecoder,
    PositionalDecoderLayer,
)
from src.models.positional_encoding import PositionEmbeddingSineLVTR
from src.models.transformer import TransformerEncoder, TransformerEncoderLayer
from src.utils.span_utils import generalized_span_iou, span_cw_to_xx, span_xx_to_cw, temporal_iou
from src.utils.training_visualization import plot_attention
from src.utils.video_length import sample_to_fixed_length, sample_to_fixed_length_list

GT_CONTINUOS = list[dict[int, list[tuple[float, float]]]]


class TDGVLightningModule(pl.LightningModule):
    def __init__(
        self,
        # Diagram
        diagram_pt_pathname: str,
        # Model
        ## Projection
        d_video_feature: int = 768,
        d_diagram_feature: int = 768,
        d_projection_hidden: int = 768,
        p_dropout: float = 0.1,
        video_fixed_length: int = 256,  # -1 means variable length
        video_fixed_length_before: bool = True,
        ## Transformer
        d_model: int = 768,
        n_head: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_feedforward: int = 2048,
        f_activation: Literal["relu", "gelu", "glu"] = "relu",
        decoder_sa_fuse: Literal["add", "concat"] = "concat",
        decoder_ca_fuse: Literal["add", "concat"] = "concat",
        decoder_diagram_pe_fuse_type: Literal["none", "add", "concat", "concat_proj", "sa_concat_proj"] = "concat_proj",
        decoder_sa_v: Literal["positional_query", "add", "concat_proj"] = "concat_proj",
        decoder_ca_v: Literal["video_pe", "add", "concat_proj"] = "concat_proj",
        decoder_sa_mask_type: Literal[
            "KeyPadding",  # Type A via key padding
            "PaddingMask",  # Type A via attention mask
            "BlockDiagonalMask",  # Type B
            "SubDiagonalMask",  # Type C
            "BlockAndSubDiagonalMask",  # Type D
        ] = "BlockAndSubDiagonalMask",
        use_encoder: bool = False,
        ## TDGV
        n_queries: int = 3,
        freeze_queries: bool = False,
        # Criterion
        losses: dict[str, float] = {
            "span/l1": 1,
            "span/giou": 1,
            "background": 1,
        },
        background_losses: dict[str, float] = {
            "CE": 1.0,
        },
        use_background_loss_temperature: bool = True,
        background_loss_temperature_init: float = 0.07,
        cost_class: float = 1,
        cost_span: float = 1,
        cost_giou: float = 1,
        use_aux: bool = True,
        # Metrics
        log_layer_loss: bool = False,
        log_enc_self_attention: bool = True,
        log_dec_self_attention: bool = True,
        log_dec_cross_attention: bool = True,
        log_attention_all_layer: bool = False,
        log_attention_key_list: list[str] = [],
        # Persistence
        save_training_results: bool = False,
        save_validation_results: bool = True,
        save_testing_results: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Diagrams
        self.diagrams_map = torch.load(diagram_pt_pathname, map_location="cpu")

        # Model
        ## Projection
        self.video_projection = Projection(
            d_input=d_video_feature,
            d_output=d_model,
            d_hidden=d_projection_hidden,
            p_dropout=p_dropout,
        )
        self.diagram_projection = Projection(
            d_input=d_diagram_feature,
            d_output=d_model,
            d_hidden=d_projection_hidden,
            p_dropout=p_dropout,
        )

        ## Encoder
        if use_encoder:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                n_head,
                d_feedforward,
                p_dropout,
                f_activation,
            )
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(
                encoder_layer,
                n_encoder_layers,
                encoder_norm,
                return_intermediate=False,
            )

        ## Positional Decoder
        position_decoder_layer = PositionalDecoderLayer(
            d_model=d_model,
            n_head=n_head,
            d_feedforward=d_feedforward,
            p_dropout=p_dropout,
            f_activation=f_activation,
            sa_fuse=decoder_sa_fuse,
            ca_fuse=decoder_ca_fuse,
            diagram_pe_fuse_type=decoder_diagram_pe_fuse_type,
            sa_v=decoder_sa_v,
            ca_v=decoder_ca_v,
        )
        self.decoder = PositionalDecoder(
            d_model,
            n_decoder_layers,
            position_decoder_layer,
        )

        ## Positional Embedding
        self.query_positional_embedding = nn.Parameter(torch.randn(n_queries, d_model))  # q_p
        if freeze_queries:
            self.query_positional_embedding.requires_grad_(False)

        ## Video PE
        self.video_positional_encoding = PositionEmbeddingSineLVTR(
            num_pos_feats=d_model, normalize=True, temperature=10000, scale=2 * torch.pi
        )
        ## Diagram PE
        self.diagram_positional_encoding = PositionEmbeddingSineLVTR(
            num_pos_feats=d_model, normalize=True, temperature=10000, scale=2 * torch.pi
        )

        ## Prediction Head
        self.span_prediction_head = MLP(input_dim=d_model, hidden_dim=d_model, output_dim=2, num_layers=3)
        self.background_prediction_head = MLP(input_dim=d_model, hidden_dim=d_model, output_dim=1, num_layers=3)

        # Loss
        if use_background_loss_temperature:
            self.background_loss_temperature = nn.Parameter(torch.tensor(background_loss_temperature_init))

        # Metrics
        metrics = MetricCollection(
            {
                "R@1_IoU@m": RecallAtIoUSegmentMap(
                    k=[1, 3],
                    iou_threshold=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    return_mIoU=True,
                    return_avgIoU=True,
                ),
            },
            compute_groups=False,
        )
        self.training_metrics = metrics.clone(prefix="train/")
        self.validation_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Persistence
        self.training_results = {}
        self.inference_results = {}

    def _get_diagram_features_list(self, key_list: list[str]) -> list[Tensor]:
        """Get diagram features by key list.

        Args:
            key_list (list[str]): Batched key list.

        Returns:
            list[Tensor]: Batched diagram features.
        """
        return [self.diagrams_map[key.split("_")[0]].to(self.device) for key in key_list]

    def _filter_diagrams_features_list(
        self, diagram_features_list: list[Tensor], ground_truth: GT_CONTINUOS
    ) -> tuple[list[Tensor], GT_CONTINUOS]:
        """Filter diagram features and ground truth by slicing them into the same window of diagram ids.

        Args:
            diagram_features_list (list[Tensor]): Batched diagram features.
            ground_truth (GT_CONTINUOS): Batched ground truth.

        Returns:
            tuple[list[Tensor], GT_CONTINUOS]: Filtered diagram features and ground truth.
        """
        batch_size = len(ground_truth)
        filtered_diagram_features_list = []
        filtered_ground_truth = []
        for i in range(batch_size):
            sample_diagram_features = diagram_features_list[i]
            sample_ground_truth = ground_truth[i]

            # Find the window of diagram ids
            sample_diagram_ids_sorted = sorted(list(sample_ground_truth.keys()))
            sample_diagram_ids_min = sample_diagram_ids_sorted[0]
            sample_diagram_ids_max = sample_diagram_ids_sorted[-1]

            # Filter diagram features
            new_sample_diagram_features = sample_diagram_features[sample_diagram_ids_min : sample_diagram_ids_max + 1]
            filtered_diagram_features_list.append(new_sample_diagram_features)

            # Filter ground truth
            new_sample_ground_truth = {}
            for diagram_id in sample_diagram_ids_sorted:
                new_sample_ground_truth[diagram_id - sample_diagram_ids_min] = sample_ground_truth[diagram_id]
            filtered_ground_truth.append(new_sample_ground_truth)

        return filtered_diagram_features_list, filtered_ground_truth

    @torch.no_grad()
    def _match(
        self,
        background_logits: Tensor,  #  B x (K x M) x 1
        span_predictions: Tensor,  # B x (K x M) x 2
        ground_truth: GT_CONTINUOS,
    ):
        batch_size = len(ground_truth)
        match_results = []
        K = self.hparams.n_queries
        for sample_i in range(batch_size):
            sample_span_predictions = span_predictions[sample_i]  # (K x M) x 2
            sample_background_logits = background_logits[sample_i]  # (K x M) x 1
            sample_span_gt_dict = ground_truth[sample_i]
            sample_match_result = []
            for i, (diagram_id, spans) in enumerate(sample_span_gt_dict.items()):
                _logits = sample_background_logits[i * K : (i + 1) * K, :]  # K x 1
                _span_pred = sample_span_predictions[i * K : (i + 1) * K, :]  # K x 2
                _span_gt = torch.tensor(spans, device=self.device)  # n_target_spans x 2
                indices = self._hungarian_matcher(_logits, _span_pred, _span_gt)
                sample_match_result.append(
                    dict(
                        diagram_id=diagram_id,
                        pred_i=indices[0],
                        gt_i=indices[1],
                    )
                )
            match_results.append(sample_match_result)
        return match_results

    @torch.no_grad()
    def _hungarian_matcher(self, logits: Tensor, span_pred: Tensor, span_tgt: Tensor):
        # Compute the class cost
        cost_class = -logits.softmax(dim=0)

        # Compute the L1 cost between spans
        cost_span = torch.cdist(span_pred, span_tgt, p=1)

        # Compute the giou cost between spans
        cost_giou = -generalized_span_iou(span_pred, span_tgt)

        # Final cost matrix
        cost = (
            self.hparams.cost_class * cost_class
            + self.hparams.cost_span * cost_span
            + self.hparams.cost_giou * cost_giou
        )

        # linear_sum_assignment
        indices = linear_sum_assignment(cost.cpu())

        return indices

    def compute_loss(
        self,
        batch_size: int,
        background_logits: Tensor,
        span_predictions: Tensor,
        ground_truth: GT_CONTINUOS,
    ):
        num_layer = background_logits.shape[0]
        K = self.hparams.n_queries
        # Losses
        loss = torch.tensor(0.0, device=self.device)

        # center l1 loss and width l2 loss: this is for logging only, not used for training
        self.hparams.losses["center/l1"] = 0
        self.hparams.losses["width/l1"] = 0

        loss_scale = 1.0 / sum(self.hparams.losses.values())

        for layer_i in range(num_layer - 1, 0 - 1, -1):
            layer_losses = {k: torch.tensor(0.0, device=self.device) for k in self.hparams.losses.keys()}
            layer_background_logits = background_logits[layer_i]  # B x (K x M) x 1
            layer_span_predictions = span_predictions[layer_i]  # B x (K x M) x 2
            layer_match_results = self._match(layer_background_logits, layer_span_predictions, ground_truth)
            span_loss_count = 0
            for sample_i in range(batch_size):
                sample_background_logits = layer_background_logits[sample_i]  # (K x M) x 1

                if "span/l1_all" in self.hparams.losses or "span/giou_all" in self.hparams.losses:
                    for diagram_index, gt in ground_truth[sample_i].items():
                        pred_spans = layer_span_predictions[sample_i, diagram_index * K : (diagram_index + 1) * K]
                        gt_span = torch.tensor(random.choice(gt), device=self.device).repeat(pred_spans.shape[0], 1)
                        if "span/l1_all" in self.hparams.losses:
                            layer_losses["span/l1_all"] += F.l1_loss(pred_spans, gt_span)
                        if "span/giou_all" in self.hparams.losses:
                            layer_losses["span/giou_all"] += (
                                1 - torch.diag(generalized_span_iou(pred_spans, gt_span)).mean()
                            )

                    if "span/l1_all" in self.hparams.losses:
                        layer_losses["span/l1_all"] /= len(ground_truth[sample_i])

                    if "span/giou_all" in self.hparams.losses:
                        layer_losses["span/giou_all"] /= len(ground_truth[sample_i])

                for match_result in layer_match_results[sample_i]:
                    # NOTE: Below two are lists of length of gt, for most of the time, the length would be just one.
                    pred_i = match_result["pred_i"]
                    gt_i = match_result["gt_i"]
                    diagram_id = match_result["diagram_id"]

                    pred_span = layer_span_predictions[sample_i, diagram_id * K + pred_i]
                    gt_span = torch.tensor(
                        [ground_truth[sample_i][diagram_id][_gt_i] for _gt_i in gt_i],
                        device=self.device,
                    )

                    # span loss
                    span_loss_count += 1
                    if "span/l1" in self.hparams.losses:
                        layer_losses["span/l1"] += F.l1_loss(pred_span, gt_span)
                    if "span/giou" in self.hparams.losses:
                        layer_losses["span/giou"] += 1 - torch.diag(generalized_span_iou(pred_span, gt_span)).mean()
                    with torch.no_grad():
                        # the following losses are for logging only, not used for training, because they are essentially
                        # the same as span/l1
                        _pred_span = span_xx_to_cw(pred_span)
                        _gt_span = span_xx_to_cw(gt_span)
                        layer_losses["center/l1"] += F.l1_loss(_pred_span[..., 0], _gt_span[..., 0])
                        layer_losses["width/l1"] += F.l1_loss(_pred_span[..., 1], _gt_span[..., 1])

                    # background loss, actually this loss's name should be **score** loss
                    sample_background_logits_per_query = sample_background_logits[
                        diagram_id * K : (diagram_id + 1) * K
                    ].unsqueeze(0)  # 1 x K x 1
                    if "background" in self.hparams.losses and "CE" in self.hparams.background_losses:
                        # Note: might be more than one non-background, here we randomly sample one
                        _pred_i = random.choice(pred_i)
                        _target = torch.tensor([[_pred_i]], dtype=torch.long, device=self.device)
                        _input = sample_background_logits_per_query
                        if self.hparams.use_background_loss_temperature:
                            _input = _input * self.background_loss_temperature
                        layer_losses["background"] += (
                            F.cross_entropy(input=_input, target=_target) * self.hparams.background_losses["CE"]
                        )
                    else:
                        raise ValueError(f"Unknown background loss: {self.hparams.background_losses}")

            if "span/l1" in self.hparams.losses:
                layer_losses["span/l1"] /= span_loss_count
            if "span/giou" in self.hparams.losses:
                layer_losses["span/giou"] /= span_loss_count
            if "background" in self.hparams.losses:
                layer_losses["background"] /= span_loss_count
            with torch.no_grad():
                layer_losses["center/l1"] /= span_loss_count
                layer_losses["width/l1"] /= span_loss_count

            # Loss
            layer_loss = torch.tensor(0.0, device=self.device)
            if layer_i == num_layer - 1:
                for name, value in layer_losses.items():
                    self.log(f"{self.stage}/loss/{name}", value, sync_dist=True)
                    _value = value * self.hparams.losses[name] * loss_scale
                    layer_loss += _value
                self.log(f"{self.stage}/loss", layer_loss, sync_dist=True)
                loss += layer_loss
                if not self.hparams.use_aux:
                    break
            if layer_i < num_layer - 1:
                for name, value in layer_losses.items():
                    if self.hparams.log_layer_loss:
                        self.log(f"{self.stage}/layer_{layer_i}/loss/{name}", value, sync_dist=True)
                    _value = value * self.hparams.losses[name] * loss_scale
                    layer_loss += _value
                if self.hparams.log_layer_loss:
                    self.log(f"{self.stage}/layer_{layer_i}/loss", layer_loss, sync_dist=True)
                loss += layer_loss

            if self.hparams.use_background_loss_temperature:
                self.log(f"{self.stage}/background_loss_temperature", self.background_loss_temperature, sync_dist=True)

        return loss

    @torch.no_grad()
    def evaluate(
        self,
        key: list[str],
        span_predictions: Tensor,
        background_logits: Tensor,
        ground_truth: GT_CONTINUOS,
    ):
        metrics = dict(train=self.training_metrics, val=self.validation_metrics, test=self.test_metrics)[self.stage]
        K = self.hparams.n_queries
        for sample_i in range(len(ground_truth)):
            sample_key = key[sample_i]
            sample_span_predictions = span_predictions[sample_i]  # (K x M) x 2
            sample_background_logits = background_logits[sample_i]  # (K x M) x 1
            sample_span_gt_dict = ground_truth[sample_i]
            sample_span_pred_dict = {}
            sample_background_logits_dict = {}
            sample_query_ids = {}
            for diagram_id, span_gt in sample_span_gt_dict.items():
                _span_pred = sample_span_predictions[diagram_id * K : (diagram_id + 1) * K, :]
                _background_logits = sample_background_logits[diagram_id * K : (diagram_id + 1) * K, :].squeeze(-1)
                sorted_idx = _background_logits.argsort(descending=True).long()
                sample_span_pred_dict[diagram_id] = _span_pred[sorted_idx].clip(0, 1).cpu().numpy().tolist()
                sample_background_logits_dict[diagram_id] = _background_logits[sorted_idx].cpu().numpy().tolist()
                sample_query_ids[diagram_id] = sorted_idx.cpu().numpy().tolist()
            metrics.update(sample_span_pred_dict, sample_span_gt_dict)
            if (self.hparams.save_validation_results and self.stage == "val") or (
                self.hparams.save_testing_results and self.stage == "test"
            ):
                if sample_key in self.inference_results:
                    # Merge prediction to adapt one diagram a time inference
                    existing_prediction = self.inference_results[sample_key]["prediction"]
                    for diagram_id, pred in sample_span_pred_dict.items():
                        if diagram_id in existing_prediction:
                            existing_prediction[diagram_id] = pred
                            logging.warning(f"Overwrite prediction: {sample_key} {diagram_id}")
                        else:
                            existing_prediction.update({diagram_id: pred})
                    # Merge ground truth to adapt one diagram a time inference
                    existing_logits = self.inference_results[sample_key]["logits"]
                    for diagram_id, logits in sample_background_logits_dict.items():
                        if diagram_id in existing_logits:
                            existing_logits[diagram_id] = logits
                            logging.warning(f"Overwrite logits: {sample_key} {diagram_id}")
                        else:
                            existing_logits.update({diagram_id: logits})
                    # Merge ground truth to adapt one diagram a time inference
                    existing_ground_truth = self.inference_results[sample_key]["ground_truth"]
                    for diagram_id, gt in sample_span_gt_dict.items():
                        if diagram_id in existing_ground_truth:
                            existing_ground_truth[diagram_id] = gt
                            logging.warning(f"Overwrite ground truth: {sample_key} {diagram_id}")
                        else:
                            existing_ground_truth.update({diagram_id: gt})
                    # Merge query ids to adapt one diagram a time inference
                    existing_query_ids = self.inference_results[sample_key]["query_ids"]
                    for diagram_id, query_ids in sample_query_ids.items():
                        if diagram_id in existing_query_ids:
                            existing_query_ids[diagram_id] = query_ids
                            logging.warning(f"Overwrite query ids: {sample_key} {diagram_id}")
                        else:
                            existing_query_ids.update({diagram_id: query_ids})
                else:
                    # If not exist, create a new entry
                    self.inference_results[sample_key] = dict(
                        prediction=sample_span_pred_dict,
                        ground_truth=sample_span_gt_dict,
                        logits=sample_background_logits_dict,
                        query_ids=sample_query_ids,
                    )
            elif self.hparams.save_training_results and self.stage == "train":
                i = 0
                while f"{sample_key}_{i}" in self.training_results:
                    i += 1
                self.training_results[f"{sample_key}_{i}"] = dict(
                    prediction=sample_span_pred_dict,
                    ground_truth=sample_span_gt_dict,
                    logits=sample_background_logits_dict,
                    query_ids=sample_query_ids,
                )

    def forward(
        self,
        video_features_list: list[Tensor],
        diagram_features_list: list[Tensor],
    ) -> Tensor:
        batch_size = len(video_features_list)

        # Video
        if self.hparams.video_fixed_length > 0 and self.hparams.video_fixed_length_before:
            video_features_list = sample_to_fixed_length_list(video_features_list, self.hparams.video_fixed_length)
        video_features_list = [self.video_projection(f) for f in video_features_list]

        video_features = nn.utils.rnn.pad_sequence(video_features_list, batch_first=True)  # B x N x D
        N = video_features.shape[1]  # N
        video_mask = torch.zeros(batch_size, N, device=self.device).bool()
        for i, video_feature in enumerate(video_features_list):
            video_mask[i, video_feature.shape[0] :] = True
        video_positional_encoding = self.video_positional_encoding(video_features, video_mask)  # B x N x D

        if self.hparams.video_fixed_length > 0 and not self.hparams.video_fixed_length_before:
            video_features = sample_to_fixed_length(video_features, video_mask, self.hparams.video_fixed_length)
            N = video_features.shape[1]  # N
            video_mask = torch.zeros(batch_size, N, device=self.device).bool()
            video_positional_encoding = self.video_positional_encoding(video_features, video_mask)  # new pe

        # Diagram
        diagram_features_list = [self.diagram_projection(f) for f in diagram_features_list]
        diagram_features = nn.utils.rnn.pad_sequence(diagram_features_list, batch_first=True)  # B x M x D
        M = diagram_features.shape[1]  # M
        diagram_mask = torch.zeros(batch_size, M, device=self.device).bool()
        for i, diagram_feature in enumerate(diagram_features_list):
            diagram_mask[i, diagram_feature.shape[0] :] = True
        diagram_positional_encoding = self.diagram_positional_encoding(diagram_features, diagram_mask)  # B x M x D

        ## Decoder
        K = self.hparams.n_queries
        video_features = video_features.permute(1, 0, 2)  # N x B x D
        video_key_padding_mask = video_mask  # B x N
        video_positional_encoding = video_positional_encoding.permute(1, 0, 2)  # N x B x D
        diagram_features = diagram_features.permute(1, 0, 2).repeat_interleave(repeats=K, dim=0)  # (M x K) x B x D
        diagram_positional_encoding = diagram_positional_encoding.permute(1, 0, 2).repeat_interleave(
            repeats=K, dim=0
        )  # (M x K) x B x D
        positional_query = self.query_positional_embedding.unsqueeze(1).repeat(M, batch_size, 1)  # (K x M) x B x D
        diagram_query_key_padding_mask = diagram_mask.repeat_interleave(repeats=K, dim=1)  # B x (M x K)
        diagram_query_attention_mask = None  # To be built later
        if self.hparams.decoder_sa_mask_type == "KeyPadding":
            pass
        elif self.hparams.decoder_sa_mask_type == "PaddingMask":  # Should be same as "KeyPadding", Type A
            diagram_query_attention_mask = (
                diagram_query_key_padding_mask.unsqueeze(1)  # B x 1 x (K x M)
                .repeat_interleave(M * K, dim=1)  # B x (K x M) x (K x M)
                .repeat_interleave(self.hparams.n_head, dim=0)
            )  # (B x num_heads) x (K x M) x (K x M)
            diagram_query_key_padding_mask = None
        elif self.hparams.decoder_sa_mask_type == "BlockDiagonalMask":  # Type B
            blocked_diagonal_mask = torch.ones(
                (M * K, M * K), device=self.device, dtype=torch.bool
            )  # All True means ignoring all
            for i in range(M):
                blocked_diagonal_mask[i * K : (i + 1) * K, i * K : (i + 1) * K] = False
            padding_mask = diagram_query_key_padding_mask.unsqueeze(1).repeat_interleave(
                M * K, dim=1
            )  # B x (K x M) x (K x M)
            diagram_query_attention_mask = (
                padding_mask | blocked_diagonal_mask
            )  # Mask out according to the key padding mask
            diagram_query_attention_mask = (
                padding_mask.transpose(1, 2) ^ diagram_query_attention_mask
            )  # Mask back the bottom to avoid NaN
            diagram_query_attention_mask = diagram_query_attention_mask.repeat_interleave(
                self.hparams.n_head, dim=0
            )  # Repeat for each head
        elif self.hparams.decoder_sa_mask_type == "SubDiagonalMask":  # Type C
            diagram_query_attention_mask = torch.ones((M * K, M * K), device=self.device, dtype=torch.bool)
            for i in range(M * K):
                for j in range(i % K, M * K, K):
                    diagram_query_attention_mask[i, j] = False
        elif self.hparams.decoder_sa_mask_type == "BlockAndSubDiagonalMask":  # Type D
            diagram_query_attention_mask = torch.ones((M * K, M * K), device=self.device, dtype=torch.bool)
            for i in range(M):
                diagram_query_attention_mask[i * K : (i + 1) * K, i * K : (i + 1) * K] = False
            for i in range(M * K):
                for j in range(i % K, M * K, K):
                    if i != j:
                        diagram_query_attention_mask[i, j] = False
        else:
            raise ValueError(f"Unknown decoder_sa_mask_type: {self.hparams.decoder_sa_mask_type}")

        enc_self_attention_weights = None
        if self.hparams.use_encoder:
            # concate video and diagram features
            src = torch.cat([video_features, diagram_features], dim=0)
            src_key_padding_mask = torch.cat([video_key_padding_mask, diagram_query_key_padding_mask], dim=1)
            src_pos = torch.cat([video_positional_encoding, diagram_positional_encoding], dim=0)
            enc_output, enc_self_attention_weights = self.encoder(
                src=src,
                src_key_padding_mask=src_key_padding_mask,
                src_pos=src_pos,
            )
            video_features = enc_output[-1]
            video_features = video_features[:N]

        decoder_output: DecoderOutput = self.decoder(
            video_features=video_features,
            diagram_features=diagram_features,
            positional_query=positional_query,
            video_positional_encoding=video_positional_encoding,
            diagram_positional_encoding=diagram_positional_encoding,
            video_key_padding_mask=video_key_padding_mask,
            diagram_query_key_padding_mask=diagram_query_key_padding_mask,
            video_attention_mask=None,
            diagram_query_attention_mask=diagram_query_attention_mask,
        )
        output = decoder_output.output
        attention_weights = decoder_output.ca_weight
        self_attention_weights = decoder_output.sa_weight

        # Span prediction
        span_predictions = self.span_prediction_head(output).sigmoid()  # n_layers x (K x M) x B x 2

        # Background prediction
        background_logits = self.background_prediction_head(output)  # n_layers x (K x M) x B x 1

        # Convert back to batch first
        return dict(
            span_predictions=span_predictions.permute(0, 2, 1, 3),  # n_layers x B x (K x M) x 2
            background_logits=background_logits.permute(0, 2, 1, 3),  # n_layers x B x (K x M) x 1
            output=output.permute(0, 2, 1, 3),  # n_layers x B x (K x M) x D
            memory=video_features.permute(1, 0, 2),  # B x N x D
            enc_self_attention_weights=enc_self_attention_weights,  # n_layers x B x N x N
            attention_weights=attention_weights,  # n_layers x B x (K x M) x N
            self_attention_weights=self_attention_weights,  # n_layers x B x (K x M) x (K x M)
            video_features_list=video_features_list,
            diagram_features_list=diagram_features_list,
            video_mask=video_mask,
            diagram_mask=diagram_mask,
        )

    def training_step(self, batch, batch_id):
        # Get data
        key: list[str] = batch["key"]
        video_features_list: list[Tensor] = batch["clip_features"]
        ground_truth: GT_CONTINUOS = batch["gt_continuous"]
        diagram_features_list: list[Tensor] = self._get_diagram_features_list(key)
        diagram_features_list, ground_truth = self._filter_diagrams_features_list(diagram_features_list, ground_truth)

        # Forward
        _output: dict[str, Tensor] = self.forward(video_features_list, diagram_features_list)
        span_predictions = _output["span_predictions"]  # n_layers x B x (K x M) x 2
        span_predictions = span_cw_to_xx(span_predictions)  # n_layers x B x (K x M) x 2
        background_logits = _output["background_logits"]  # n_layers x B x (K x M) x 1
        video_features_list = _output["video_features_list"]
        diagram_features_list = _output["diagram_features_list"]

        # Loss
        loss = self.compute_loss(
            batch_size=len(key),
            background_logits=background_logits,
            span_predictions=span_predictions,
            ground_truth=ground_truth,
        )

        self.evaluate(key, span_predictions[-1], background_logits[-1], ground_truth)

        self.log_dict(self.training_metrics.compute(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_id):
        # Get data
        key: list[str] = batch["key"]
        video_features_list: list[Tensor] = batch["clip_features"]
        ground_truth: GT_CONTINUOS = batch["gt_continuous"]
        diagram_features_list: list[Tensor] = self._get_diagram_features_list(key)

        # Forward
        _output: dict[str, Tensor] = self.forward(video_features_list, diagram_features_list)
        span_predictions = _output["span_predictions"]  # n_layers x B x (K x M) x 2
        span_predictions = span_cw_to_xx(span_predictions)  # n_layers x B x (K x M) x 2
        background_logits = _output["background_logits"]  # n_layers x B x (K x M) x 1
        enc_self_attention_weights = _output["enc_self_attention_weights"]  # n_layers x B x n_heads x N x N
        attention_weights = _output["attention_weights"]  # n_layers x B x n_heads x (K x M) x N
        self_attention_weights = _output["self_attention_weights"]  # n_layers x B x n_heads x (K x M) x (K x M)
        video_features_list = _output["video_features_list"]
        diagram_features_list = _output["diagram_features_list"]

        # Loss
        self.compute_loss(
            batch_size=len(key),
            background_logits=background_logits,
            span_predictions=span_predictions,
            ground_truth=ground_truth,
        )

        self.evaluate(key, span_predictions[-1], background_logits[-1], ground_truth)

        self._log_attention(
            key,
            enc_self_attention_weights=enc_self_attention_weights,
            dec_self_attention_weights=self_attention_weights,
            dec_cross_attention_weights=attention_weights,
        )

    def test_step(self, batch, batch_id):
        # Get data
        key: list[str] = batch["key"]
        video_features_list: list[Tensor] = batch["clip_features"]
        ground_truth: GT_CONTINUOS = batch["gt_continuous"]
        diagram_features_list: list[Tensor] = self._get_diagram_features_list(key)

        # Forward
        _output: dict[str, Tensor] = self.forward(video_features_list, diagram_features_list)
        span_predictions = _output["span_predictions"]  # n_layers x B x (K x M) x 2
        span_predictions = span_cw_to_xx(span_predictions)  # n_layers x B x (K x M) x 2
        background_logits = _output["background_logits"]  # n_layers x B x (K x M) x 1
        enc_self_attention_weights = _output["enc_self_attention_weights"]  # n_layers x B x n_heads x N x N
        attention_weights = _output["attention_weights"]  # n_layers x B x n_heads x (K x M) x N
        self_attention_weights = _output["self_attention_weights"]  # n_layers x B x n_heads x (K x M) x (K x M)

        self.evaluate(key, span_predictions[-1], background_logits[-1], ground_truth)

        self._log_attention(
            key,
            enc_self_attention_weights=enc_self_attention_weights,
            dec_self_attention_weights=self_attention_weights,
            dec_cross_attention_weights=attention_weights,
        )

    def on_train_epoch_end(self):
        self.training_metrics.reset()
        if self.hparams.save_training_results:
            self._save_pt_file_to_log_dir("results", self.training_results)
            self.training_results = {}

    def on_validation_epoch_end(self):
        self.log_dict(self.validation_metrics.compute(), sync_dist=True)
        self.validation_metrics.reset()
        if self.hparams.save_validation_results:
            self._save_pt_file_to_log_dir("results", self.inference_results)
            self.inference_results = {}

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()
        if self.hparams.save_testing_results:
            self._save_pt_file_to_log_dir("results", self.inference_results)
            self.inference_results = {}

    # Visualization
    def _log_plot(self, key: str, image: PIL.Image):
        if self.logger and self.logger.experiment:
            self.logger.experiment.log({key: wandb.Image(image)})

    def _log_attention(
        self,
        key: list[str],
        enc_self_attention_weights: Tensor | None,  # B x n_heads x N x N
        dec_self_attention_weights: Tensor | None,  # B x n_heads x (K x M) x (K x M)
        dec_cross_attention_weights: Tensor | None,  # B x n_heads x (K x M) x N
    ):
        if self.hparams.log_attention_key_list:
            for sample_i, _key in enumerate(key):
                if _key in self.hparams.log_attention_key_list:
                    if self.hparams.log_enc_self_attention and enc_self_attention_weights is not None:
                        for layer_i in (
                            range(self.hparams.n_encoder_layers)
                            if self.hparams.log_attention_all_layer
                            else [self.hparams.n_encoder_layers - 1]
                        ):
                            for head_i in range(self.hparams.n_head):
                                self._log_plot(
                                    f"{self.stage}/enc_self_attention/{_key}/layer_{layer_i}/head_{head_i}",
                                    plot_attention(enc_self_attention_weights[layer_i][sample_i][head_i]),
                                )
                    if self.hparams.log_dec_self_attention and dec_self_attention_weights is not None:
                        for layer_i in (
                            range(self.hparams.n_decoder_layers)
                            if self.hparams.log_attention_all_layer
                            else [self.hparams.n_decoder_layers - 1]
                        ):
                            for head_i in range(self.hparams.n_head):
                                self._log_plot(
                                    f"{self.stage}/dec_self_attention/{_key}/layer_{layer_i}/head_{head_i}",
                                    plot_attention(dec_self_attention_weights[layer_i][sample_i][head_i]),
                                )
                    if self.hparams.log_dec_cross_attention and dec_cross_attention_weights is not None:
                        for layer_i in (
                            range(self.hparams.n_decoder_layers)
                            if self.hparams.log_attention_all_layer
                            else [self.hparams.n_decoder_layers - 1]
                        ):
                            for head_i in range(self.hparams.n_head):
                                self._log_plot(
                                    f"{self.stage}/dec_cross_attention/{_key}/layer_{layer_i}/head_{head_i}",
                                    plot_attention(
                                        dec_cross_attention_weights[layer_i][sample_i][head_i]
                                        .reshape(
                                            -1,  # M
                                            self.hparams.n_queries,  # K
                                            dec_cross_attention_weights.shape[-1],  # N
                                        )
                                        .permute(1, 0, 2)
                                        .reshape(
                                            -1,  # K x M
                                            dec_cross_attention_weights.shape[-1],  # N
                                        ),
                                    ),
                                )

    # Persistence
    def _save_pt_file_to_log_dir(self, name: str, data: dict):
        log_dir = self.trainer.log_dir
        if (
            self.trainer.logger is not None
            and self.trainer.logger.name is not None
            and self.trainer.logger.version is not None
        ):
            log_dir = os.path.join(log_dir, self.trainer.logger.name, str(self.trainer.logger.version))
        elif self.trainer.is_global_zero:
            logging.warning(f"Could not save results to specific run folder, use default {log_dir} instead.")
        log_dir = self.trainer.strategy.broadcast(log_dir)
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step
        stage = self.stage
        file_name = f"{name}_{stage}_epoch_{epoch}_step_{step}.pt"
        file_pathname = os.path.join(log_dir, file_name)
        with FileLock(file_pathname + ".lock"):
            if os.path.exists(file_pathname):
                with open(file_pathname, "rb") as f:
                    exsiting_data = torch.load(f)
            else:
                exsiting_data = {}
            data.update(exsiting_data)
            torch.save(data, file_pathname)

    # Misc
    @property
    def stage(self):
        """Return current stage and map `validate` to `val`.

        Returns:
            str|None: current stage
        """
        return "val" if self.trainer.state.stage == "validate" else self.trainer.state.stage
