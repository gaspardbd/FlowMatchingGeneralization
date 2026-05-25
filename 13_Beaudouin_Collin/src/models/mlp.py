"""Simple MLP velocity network for 2D toy experiments."""

import torch
import torch.nn as nn


class VelocityMLP(nn.Module):
    """MLP that predicts u_theta(x_t, t) for low-dimensional data.

    Input: concatenation of x_t (dim d) and t (dim 1) -> output: velocity (dim d).
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(data_dim + 1, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, data_dim)
            t: (batch,) or (batch, 1)
        Returns:
            velocity: (batch, data_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1))
