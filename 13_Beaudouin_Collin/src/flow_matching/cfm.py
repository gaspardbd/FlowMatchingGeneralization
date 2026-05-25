"""Vanilla Conditional Flow Matching — Algorithm 1 of the paper."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange


class CFMTrainer:
    """Trains a velocity network u_theta with the standard CFM loss.

    L_CFM = E_{x0~N(0,I), x1~p_data, t~U(0,1)} || u_theta(x_t, t) - (x1 - x0) ||²
    where x_t = (1 - t)·x0 + t·x1.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.device = device

    def train_step(self, x1: torch.Tensor) -> float:
        """One training step.

        Args:
            x1: (batch, *shape) — batch of training data

        Returns:
            loss value (float)
        """
        x1 = x1.to(self.device)
        batch_size = x1.shape[0]

        t = torch.rand(batch_size, device=self.device)
        x0 = torch.randn_like(x1)

        t_expand = t.view(batch_size, *([1] * (x1.dim() - 1)))
        x_t = (1.0 - t_expand) * x0 + t_expand * x1

        target = x1 - x0
        pred = self.model(x_t, t)

        loss = ((pred - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset, n_steps: int = 10000, batch_size: int = 256, log_every: int = 1000):
        """Full training loop.

        Args:
            dataset: ToyDataset with a .sample(batch_size) method,
                     or a torch.Tensor of shape (n, d).
            n_steps: number of gradient steps.
            batch_size: batch size.
            log_every: logging frequency.

        Returns:
            list of loss values (one per logged step).
        """
        self.model.train()
        losses = []
        pbar = trange(n_steps, desc="CFM Training")
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
