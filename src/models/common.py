import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Projection(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_hidden: int,
        p_dropout: float,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            ProjectionLayer(d_input, d_hidden, p_dropout, append_relu=True),
            ProjectionLayer(d_hidden, d_output, p_dropout, append_relu=False),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class ProjectionLayer(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        p_dropout: float,
        append_relu: bool,
    ):
        super().__init__()
        layer = [
            nn.LayerNorm(d_input),
            nn.Dropout(p_dropout),
            nn.Linear(d_input, d_output),
        ]
        if append_relu:
            layer.append(nn.ReLU())
        self.layer = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor):
        return self.layer(x)
