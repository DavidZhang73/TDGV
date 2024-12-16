"""
Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
"""

import torch
import torch.nn as nn


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer from
    Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """

    def __init__(self):
        super().__init__()
        self.pairwise_distance = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, index = torch.max(dots, dim=-1)
        return index

    def forward(self, normalized_feature, eps=1e-8):
        """
        Args:
            normalized_feature (BxD): l2 normalized feature output from a backbone
        """
        index = self.pairwise_NNs_inner(normalized_feature)
        distances = self.pairwise_distance(normalized_feature, normalized_feature[index])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss
