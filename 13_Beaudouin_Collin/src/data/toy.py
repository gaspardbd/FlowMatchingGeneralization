"""Toy 2D and higher-dimensional datasets for flow matching experiments."""

import torch
import numpy as np
from sklearn.datasets import make_moons


class ToyDataset:
    """Generates samples from synthetic distributions.

    Supported: "moons", "rings", "gaussian_mixture", "gaussian_mixture_nd".
    For higher-dimensional experiments, use gaussian_mixture_nd with a custom dim.
    """

    def __init__(self, name: str, n_samples: int = 2000, dim: int = 2, seed: int = 42):
        self.name = name
        self.n_samples = n_samples
        self.dim = dim
        self.seed = seed
        self.data = self._generate()

    def _generate(self) -> torch.Tensor:
        rng = np.random.RandomState(self.seed)
        if self.name == "moons":
            return self._make_moons(rng)
        elif self.name == "rings":
            return self._make_rings(rng)
        elif self.name == "gaussian_mixture":
            return self._make_gaussian_mixture(rng)
        elif self.name == "gaussian_mixture_nd":
            return self._make_gaussian_mixture_nd(rng)
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

    def _make_moons(self, rng) -> torch.Tensor:
        X, _ = make_moons(n_samples=self.n_samples, noise=0.06, random_state=rng)
        X = (X - X.mean(axis=0)) / X.std(axis=0) * 0.5
        return torch.from_numpy(X).float()

    def _make_rings(self, rng) -> torch.Tensor:
        n = self.n_samples
        radii = [0.5, 1.0, 1.5]
        samples_per_ring = n // len(radii)
        points = []
        for r in radii:
            theta = rng.uniform(0, 2 * np.pi, samples_per_ring)
            noise = rng.randn(samples_per_ring) * 0.05
            x = (r + noise) * np.cos(theta)
            y = (r + noise) * np.sin(theta)
            points.append(np.stack([x, y], axis=1))
        remaining = n - samples_per_ring * len(radii)
        if remaining > 0:
            theta = rng.uniform(0, 2 * np.pi, remaining)
            noise = rng.randn(remaining) * 0.05
            x = (radii[-1] + noise) * np.cos(theta)
            y = (radii[-1] + noise) * np.sin(theta)
            points.append(np.stack([x, y], axis=1))
        return torch.from_numpy(np.concatenate(points, axis=0)).float()

    def _make_gaussian_mixture(self, rng, n_components: int = 8) -> torch.Tensor:
        n = self.n_samples
        angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
        centers = 2.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        assignments = rng.randint(0, n_components, n)
        sigma = 0.15
        samples = centers[assignments] + rng.randn(n, 2) * sigma
        return torch.from_numpy(samples).float()

    def _make_gaussian_mixture_nd(self, rng, n_components: int = 8) -> torch.Tensor:
        """Gaussian mixture in d dimensions — centers spread across all dimensions.

        Centers lie on a hypersphere of radius sqrt(2d) so that
        ||center_i - center_j||^2 ~ 4d, analogous to interpoint distances in image data.
        Using sigma=0.5 gives intra-cluster collapse at t ~ 0.15-0.2 for d ~ 100-200,
        matching the stochastic-to-deterministic transition observed on CIFAR-10.
        """
        n, d = self.n_samples, self.dim
        center_rng = np.random.RandomState(1234)
        raw = center_rng.randn(n_components, d)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        centers = raw / norms * np.sqrt(2.0 * d)
        assignments = rng.randint(0, n_components, n)
        sigma = 0.5
        samples = centers[assignments] + rng.randn(n, d) * sigma
        return torch.from_numpy(samples).float()

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, len(self.data), (batch_size,))
        return self.data[idx]
