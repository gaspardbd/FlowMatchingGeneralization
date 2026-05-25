"""Evaluation metrics for flow matching experiments."""

import torch
from src.flow_matching.closed_form import optimal_velocity


def cosine_similarity_batch(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between two batches of vectors.

    Args:
        u: (batch, d)
        v: (batch, d)

    Returns:
        cos_sim: (batch,)
    """
    return torch.nn.functional.cosine_similarity(u, v, dim=-1)


def nearest_neighbor_distance(generated: torch.Tensor, train_data: torch.Tensor) -> torch.Tensor:
    """For each generated sample, compute the L2 distance to its nearest training point.

    Args:
        generated: (m, d)
        train_data: (n, d)

    Returns:
        distances: (m,) — min L2 distance for each generated sample
    """
    dists = torch.cdist(generated, train_data, p=2)
    return dists.min(dim=1).values


@torch.no_grad()
def velocity_approximation_error(
    model: torch.nn.Module,
    data: torch.Tensor,
    t_values: torch.Tensor,
    n_eval: int = 256,
    device: str = "cpu",
) -> dict:
    """Compute E||u_theta(x_t,t) - û★(x_t,t)||² for multiple time values.

    Used in Figure 2 (left panel).

    Args:
        model: trained velocity network
        data: (n, d) — training set
        t_values: (T,) — time grid
        n_eval: number of (x0, x1) pairs for evaluation
        device: device

    Returns:
        dict mapping t (float) -> mean squared error (float)
    """
    model.eval()
    data = data.to(device)
    d = data.shape[-1]

    idx = torch.randint(0, len(data), (n_eval,))
    x1 = data[idx].to(device)
    x0 = torch.randn(n_eval, d, device=device)

    results = {}
    for t_val in t_values:
        t_scalar = t_val.item() if isinstance(t_val, torch.Tensor) else t_val
        t_col = torch.full((n_eval, 1), t_scalar, device=device)
        t_flat = torch.full((n_eval,), t_scalar, device=device)
        x_t = (1.0 - t_col) * x0 + t_col * x1

        u_star = optimal_velocity(x_t, data, t_col)
        u_theta = model(x_t, t_flat)

        mse = ((u_theta - u_star) ** 2).sum(dim=-1).mean().item()
        results[t_scalar] = mse

    return results
