import math

import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, tau_init: float = 0.07, tau_learnable: bool = True):
        super().__init__()
        if tau_learnable:
            self.tau = nn.Parameter(torch.log(torch.tensor([1 / tau_init])))
        else:
            self.register_buffer("tau", torch.log(torch.tensor([1 / tau_init])))

    def forward(self, similarity_matrix: torch.Tensor):
        assert (
            similarity_matrix.ndim == 2 and similarity_matrix.shape[0] == similarity_matrix.shape[1]
        ), "similarity_matrix should be a square matrix."
        ground_truth = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * self.tau
        loss_1 = torch.nn.functional.cross_entropy(similarity_matrix, ground_truth)
        loss_2 = torch.nn.functional.cross_entropy(similarity_matrix.T, ground_truth)
        loss = (loss_1 + loss_2) / 2
        return loss

    def get_tau_value(self):
        return (1 / torch.exp(self.tau)).detach().item()

    def clamp_tau(self, min: float = -math.log(100), max: float = math.log(100)):
        self.tau.data.clamp_(min, max)


if __name__ == "__main__":
    batch_size = 10
    similarity_matrix = torch.rand(10, 10)
    loss = InfoNCELoss(tau_learnable=True)
    print(loss.get_tau_value())
