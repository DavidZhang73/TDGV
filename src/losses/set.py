import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=True)

from src.losses.hungarian_matcher import HungarianMatcher
from src.utils.span_utils import generalized_span_iou


class SetCriterion(nn.Module):
    def __init__(
        self,
        # HungarianMatcher
        cost_class: float = 1,
        cost_span: float = 1,
        cost_giou: float = 1,
        use_aux: bool = True,
    ):
        super().__init__()
        self.use_aux = use_aux
        self.matcher = HungarianMatcher(cost_class=cost_class, cost_span=cost_span, cost_giou=cost_giou)

    def forward(
        self,
        span_pred: torch.Tensor,  # n_layers x B x n_queries x 2
        diagram_features_list: list[torch.Tensor],  # a list of B M_i x D
        diagram_memory_queries_similarity: torch.Tensor,  # n_layers x B x n_queries x M
        ground_truth: list[dict[int, list[tuple[float, float]]]],
    ):
        device = span_pred.device
        n_queries = span_pred.shape[2]
        losses = {}

        # convert ground_truth to span_tgt
        span_tgt_id_map = []
        span_tgt = []
        diagram_tgt = []
        for sample in ground_truth:
            sample_span_tgt = []
            sample_diagram_tgt = []
            sample_span_tgt_id_map = {}
            count = 0
            for key, value in sample.items():
                for i, span in enumerate(value):
                    sample_span_tgt.append(span)
                    sample_diagram_tgt.append(key)
                    sample_span_tgt_id_map[count] = key, i
                    count += 1
            span_tgt.append(sample_span_tgt)
            diagram_tgt.append(sample_diagram_tgt)
            span_tgt_id_map.append(sample_span_tgt_id_map)

        if not self.use_aux:
            span_pred = span_pred[-1:]
            diagram_memory_queries_similarity = diagram_memory_queries_similarity[-1:]

        n_layers = span_pred.shape[0]
        matching_result = self.matcher(diagram_memory_queries_similarity[-1], diagram_tgt, span_pred[-1], span_tgt)
        # TODO: match for each layer
        for layer_i in range(n_layers):
            _span_pred = span_pred[layer_i]  # B x n_queries x 2
            _diagram_memory_queries_similarity = diagram_memory_queries_similarity[layer_i]  # B x n_queries x D

            # diagram_memory_queries_ce_loss = torch.tensor(0.0, device=device)
            # diagram_memory_queries_ce_loss_count = 1e-8

            matched_spin_pred_list = []
            matched_spin_tgt_list = []
            for sample_index, sample in enumerate(matching_result):
                index_pred, index_tgt = sample
                for i, j in zip(index_pred, index_tgt):
                    pred_span = _span_pred[sample_index, i]
                    tgt_span_key, tgt_span_index = span_tgt_id_map[sample_index][j]
                    tgt_span = ground_truth[sample_index][tgt_span_key][tgt_span_index]
                    matched_spin_pred_list.append(pred_span)
                    matched_spin_tgt_list.append(torch.tensor(tgt_span, device=device))

                    # Calculate diagram_memory_queries_ce_loss
                    # gt = torch.tensor(tgt_span_key, dtype=torch.long, device=device)
                    # diagram_memory_queries_ce_loss = diagram_memory_queries_ce_loss + F.cross_entropy(
                    #     _diagram_memory_queries_similarity[sample_index, i], gt
                    # )
                    # diagram_memory_queries_ce_loss_count += 1

            matched_spin_pred = torch.stack(matched_spin_pred_list)
            matched_spin_tgt = torch.stack(matched_spin_tgt_list)

            # Calculate set guidance loss
            set_guidance_loss = torch.tensor(0.0, device=device)
            diagram_lengths = [diagram_features.shape[0] for diagram_features in diagram_features_list]
            tgt = []
            for sample_index, diagram_length in enumerate(diagram_lengths):
                sample_tgt = torch.zeros(n_queries, dtype=torch.long, device=device)
                for i, chunk in enumerate(sample_tgt.chunk(diagram_length)):
                    chunk.fill_(i)
                tgt.append(sample_tgt)
            tgt = torch.stack(tgt)  # B x n_queries
            set_guidance_loss = set_guidance_loss + F.cross_entropy(
                _diagram_memory_queries_similarity.flatten(0, 1), tgt.flatten(0, 1)
            )

            layer_name = f"/layer_{layer_i}" if layer_i != n_layers - 1 else ""
            # L1 loss
            losses[f"loss/span/l1{layer_name}"] = F.l1_loss(matched_spin_pred, matched_spin_tgt)
            # GIoU loss
            losses[f"loss/span/giou{layer_name}"] = (
                1 - torch.diag(generalized_span_iou(matched_spin_pred, matched_spin_tgt)).mean()
            )
            # diagram_memory_queries_ce_loss
            # losses[f"loss/diagram_memory_queries_ce{layer_name}"] = (
            #     diagram_memory_queries_ce_loss / diagram_memory_queries_ce_loss_count
            # )
            # set guidance loss
            losses[f"loss/set_guidance{layer_name}"] = set_guidance_loss

        return losses


if __name__ == "__main__":
    span_pred = torch.tensor(
        [
            [
                [0.1, 0.3],
                [0.5, 0.6],
                [0.8, 0.9],
            ],
            [
                [0.3, 0.4],
                [0.5, 0.6],
                [0.75, 1],
            ],
            [
                [0.3, 0.4],
                [0.5, 0.6],
                [0.75, 1],
            ],
        ]
    )  # B x n_queries x 2
    ground_truth = [
        {0: [(0.15, 0.3), (0.5, 0.6)], 1: [(0.8, 0.9)]},
        {0: [(0.5, 0.6)], 1: [(0.75, 1)]},
        {0: [(0.3, 0.4)], 1: [(0.75, 1)]},
    ]
    sc = SetCriterion()
    loss = sc(span_pred.unsqueeze(0), ground_truth)
    print(loss)
