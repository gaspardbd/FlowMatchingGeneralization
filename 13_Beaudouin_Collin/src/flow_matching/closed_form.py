"""Closed-form velocity field û★ from Proposition 1 of the paper."""

import torch


def softmax_weights(x: torch.Tensor, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute softmax weights λ_i(x, t) from Equation (6).

    λ(x, t) = softmax( -||x - t·x^(j)||² / (2·(1-t)²) )

    Args:
        x: (batch, d)
        data: (n, d) — training points x^(1), ..., x^(n)
        t: (batch, 1)

    Returns:
        weights: (batch, n)
    """
    t_expand = t.unsqueeze(1)                                # (batch, 1, 1)
    diff = x.unsqueeze(1) - t_expand * data.unsqueeze(0)    # (batch, n, d)
    sq_dist = (diff ** 2).sum(dim=-1)                        # (batch, n)
    logits = -sq_dist / (2.0 * (1.0 - t) ** 2 + 1e-12)     # (batch, n)
    return torch.softmax(logits, dim=-1)


def optimal_velocity(x: torch.Tensor, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Closed-form optimal velocity field û★(x, t) from Equation (6).

    û★(x, t) = Σ_i λ_i(x,t) · (x^(i) - x) / (1 - t)

    Args:
        x: (batch, d)
        data: (n, d)
        t: (batch, 1)

    Returns:
        velocity: (batch, d)
    """
    weights = softmax_weights(x, data, t)                    # (batch, n)
    directions = (data.unsqueeze(0) - x.unsqueeze(1))        # (batch, n, d)
    directions = directions / (1.0 - t.unsqueeze(-1) + 1e-12)
    return (weights.unsqueeze(-1) * directions).sum(dim=1)


def cosine_sim_u_star_vs_ucond(
    x0: torch.Tensor, x1: torch.Tensor, data: torch.Tensor, t_values: torch.Tensor
) -> dict:
    """Compute cosine similarity between û★ and u_cond for each time value.

    For Figure 1: at each t, compute
        cos_sim( û★((1-t)x0 + t·x1, t),  x1 - x0 )

    Args:
        x0: (batch, d) — noise samples
        x1: (batch, d) — data samples
        data: (n, d) — full training set
        t_values: (T,) — time grid

    Returns:
        dict mapping each t value (float) to a tensor of cosine similarities (batch,)
    """
    results = {}
    for t_val in t_values:
        t_scalar = t_val.item() if isinstance(t_val, torch.Tensor) else t_val
        t = torch.full((x0.shape[0], 1), t_scalar, device=x0.device)
        x_t = (1.0 - t) * x0 + t * x1
        u_star = optimal_velocity(x_t, data, t)
        u_cond = x1 - x0
        cos = torch.nn.functional.cosine_similarity(u_star, u_cond, dim=-1)
        results[t_scalar] = cos
    return results
