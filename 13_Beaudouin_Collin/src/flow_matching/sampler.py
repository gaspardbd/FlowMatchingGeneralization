"""ODE sampler: generate new data by integrating the learned velocity field."""

import torch
import torch.nn as nn
from .closed_form import optimal_velocity


@torch.no_grad()
def ode_sample(
    model: nn.Module,
    n_samples: int,
    data_shape: tuple,
    n_steps: int = 100,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples by solving dx/dt = u_theta(x, t) with Euler from t=0 to t=1.

    Args:
        model: trained velocity network u_theta(x, t)
        n_samples: number of samples to generate
        data_shape: shape of one sample, e.g. (2,) or (1, 28, 28)
        n_steps: number of Euler steps
        device: device

    Returns:
        x: (n_samples, *data_shape)
    """
    model.eval()
    x = torch.randn(n_samples, *data_shape, device=device)
    dt = 1.0 / n_steps

    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((n_samples,), t_val, device=device)
        velocity = model(x, t)
        x = x + velocity * dt

    return x


@torch.no_grad()
def ode_sample_hybrid(
    model: nn.Module,
    data: torch.Tensor,
    x0: torch.Tensor,
    tau: float,
    n_steps: int = 200,
    device: str = "cpu",
) -> torch.Tensor:
    """Hybrid sampling (Figure 3): û★ on [0, τ], then u_theta on [τ, 1].

    Args:
        model: trained velocity network
        data: (n, d) — training set for closed-form û★
        x0: (batch, d) — initial noise
        tau: switching time in [0, 1]
        n_steps: total Euler steps

    Returns:
        x: (batch, d) — generated samples
    """
    model.eval()
    data = data.to(device)
    x = x0.clone().to(device)
    batch_size = x.shape[0]
    dt = 1.0 / n_steps

    for k in range(n_steps):
        t_val = k * dt
        if t_val < tau:
            t = torch.full((batch_size, 1), t_val, device=device)
            velocity = optimal_velocity(x, data, t)
        else:
            t = torch.full((batch_size,), t_val, device=device)
            velocity = model(x, t)
        x = x + velocity * dt

    return x
