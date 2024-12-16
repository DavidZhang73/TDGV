from typing import Literal

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=True)

from src.utils.span_utils import generalized_span_iou


class HungarianMatcher(nn.Module):
    """Reproduced hungarian matcher.

    In the LVTR paper, they set the `num_class` to 4. but it is not the case for our dataset
    (for each video, we can have arbitrary number of spans), hence we do not use `cost_class`.
    Instead we use the similarity between the queries and the diagrams memory as the `cost_sim`.

    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Explore-And-Match: Bridging Proposal-Based and Proposal-Free With Transformer for Sentence Grounding in Videos
    PDF: https://arxiv.org/abs/2201.10168
    Code: https://github.com/sangminwoo/Explore-And-Match
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_span: float = 1,
        cost_giou: float = 1,
    ):
        """Creates the matcher

        Args:
            cost_class (float, optional): Relative weight of the similarity in the matching cost.
            cost_span (float, optional): Relative weight of the L1 error of the span coordinates in the matching cost.
                Defaults to 1.
            cost_giou (float, optional): Relative weight of the giou loss of the spans in the matching cost.
                Defaults to 1.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        tgt: list[list[int]],
        span_pred: torch.Tensor,
        span_tgt: list[list[tuple[float, float]]],
    ):
        """Performs the matching using `linear_sum_assignment`.

        Args:
            logits (torch.Tensor): Similarity between the queries and the diagrams memory,
                with shape B x n_queries x M.
            tgt (list[list[int]]): Flatterned list of target diagrams with B samples,
                each sample contains a list of arbitrary number of diagrams.
                The diagram is represented by an integer as index.
            span_pred (torch.Tensor): Span predictions after sigmoid with shape B x n_queries x 2.
            span_tgt (list[list[tuple[float, float]]]): Flatterned list of target spans with B samples,
                each sample contains a list of arbitrary number of spans.
                The span should be in the format of (start, end).

        Returns:
            list[tuple[Tensor, Tensor]]: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_proposals, num_target_spans)
        """
        device = span_pred.device
        batch_size, n_proposal = span_pred.shape[:2]
        sizes = [len(span_list) for span_list in span_tgt]

        # Flatten to compute the cost matrices in a batch
        logits = logits.flatten(0, 1)  # (B x n_queries) x M
        tgt = torch.cat([torch.tensor(tgt_list, device=device) for tgt_list in tgt], dim=0)  # n_target_spans

        span_pred = span_pred.flatten(0, 1)  # (B x n_queries) x 2
        span_tgt = torch.cat(
            [torch.tensor(span_list, device=device) for span_list in span_tgt], dim=0
        )  # n_target_spans x 2

        # Compute the class cost
        cost_class = -logits.softmax(dim=-1)[:, tgt]  # (B x n_queries) x n_target_spans

        # Compute the L1 cost between spans
        cost_span = torch.cdist(span_pred, span_tgt, p=1)  # (B x n_queries) x n_target_spans

        # Compute the giou cost between spans
        cost_giou = -generalized_span_iou(span_pred, span_tgt)  # (B x n_queries) x n_target_spans

        # Final cost matrix
        cost = (
            self.cost_class * cost_class + self.cost_span * cost_span + self.cost_giou * cost_giou
        )  # (B x n_queries) x n_target_spans
        cost = cost.view(batch_size, n_proposal, -1).cpu()  # B x n_queries x n_target_spans

        # linear_sum_assignment
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]

        return indices


if __name__ == "__main__":
    matcher = HungarianMatcher()
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
    span_tgt_id_map = []
    span_tgt = []
    for sample in ground_truth:
        sample_span_tgt = []
        sample_span_tgt_id_map = {}
        count = 0
        for key, value in sample.items():
            for i, span in enumerate(value):
                sample_span_tgt.append(span)
                sample_span_tgt_id_map[count] = key, i
                count += 1
        span_tgt.append(sample_span_tgt)
        span_tgt_id_map.append(sample_span_tgt_id_map)
    result = matcher(span_pred, span_tgt)
    print(span_tgt_id_map)
    print(result)
