import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        nn.CrossEntropyLoss()

    def forward(self, similarity_vector: torch.Tensor, ground_truth: torch.Tensor):
        return F.cross_entropy(similarity_vector, ground_truth)


if __name__ == "__main__":
    similarity_vector = torch.rand(1, 10)
    ground_truth = torch.tensor([0], dtype=torch.long)
    print(similarity_vector)
    print(ground_truth)
    print(ground_truth.shape)
    l = AlignmentLoss()
    print(l(similarity_vector, ground_truth))
