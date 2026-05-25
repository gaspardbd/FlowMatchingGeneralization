"""Empirical Flow Matching — Algorithm 2 of the paper."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange


def compute_efm_target(
    x_t: torch.Tensor, x1: torch.Tensor, data: torch.Tensor, t: torch.Tensor, M: int
) -> torch.Tensor:
    """Compute the EFM target û★_M(x_t, t) using M samples (Equation 8).

    Key trick: b^(1) = x1 (the point that generated x_t), and b^(2)..b^(M) are
    sampled uniformly from the training set.

    Args:
        x_t: (batch, d) — interpolated points
        x1: (batch, d) — the data points used to construct x_t
        data: (n, d) — full training set
        t: (batch, 1) — time values
        M: number of samples for the Monte Carlo estimate

    Returns:
        target: (batch, d)
    """
    batch_size, d = x_t.shape
    device = x_t.device

    idx = torch.randint(0, len(data), (batch_size, M - 1), device=device)
    b_rest = data[idx]                                       # (batch, M-1, d)
    b = torch.cat([x1.unsqueeze(1), b_rest], dim=1)         # (batch, M, d)

    diff = x_t.unsqueeze(1) - t.unsqueeze(-1) * b           # (batch, M, d)
    sq_dist = (diff ** 2).sum(dim=-1)                        # (batch, M)
    logits = -sq_dist / (2.0 * (1.0 - t) ** 2 + 1e-12)     # (batch, M)
    weights = torch.softmax(logits, dim=-1)                  # (batch, M)

    directions = (b - x_t.unsqueeze(1)) / (1.0 - t.unsqueeze(-1) + 1e-12)  # (batch, M, d)
    target = (weights.unsqueeze(-1) * directions).sum(dim=1)                # (batch, d)
    return target


class EFMTrainer:
    """Trains a velocity network with the Empirical Flow Matching loss.

    Same as CFM but the target is û★_M instead of u^cond = x1 - x0.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        M: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.device = device
        self.train_data = train_data.to(device)
        self.M = M

    def train_step(self, x1: torch.Tensor) -> float:
        x1 = x1.to(self.device)
        batch_size = x1.shape[0]

        t = torch.rand(batch_size, 1, device=self.device)
        x0 = torch.randn_like(x1)
        x_t = (1.0 - t) * x0 + t * x1

        target = compute_efm_target(x_t, x1, self.train_data, t, self.M)
        pred = self.model(x_t, t.squeeze(-1))

        loss = ((pred - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset, n_steps: int = 10000, batch_size: int = 256, log_every: int = 1000):
        """Full training loop."""
        self.model.train()
        losses = []
        pbar = trange(n_steps, desc="EFM Training")
        for step in pbar:
            if hasattr(dataset, "sample"):
                x1 = dataset.sample(batch_size)
            else:
                idx = torch.randint(0, len(dataset), (batch_size,))
                x1 = dataset[idx]
            loss = self.train_step(x1)
            if step % log_every == 0:
                losses.append(loss)
                pbar.set_postfix(loss=f"{loss:.4f}")
        return losses
