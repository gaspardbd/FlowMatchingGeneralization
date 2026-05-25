"""Image dataset loaders for small-scale experiments (Colab-friendly)."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_image_dataloader(
    name: str = "mnist",
    batch_size: int = 128,
    image_size: int = 28,
    train: bool = True,
    n_samples: int | None = None,
    root: str = "./data",
) -> DataLoader:
    """Return a DataLoader for MNIST or Fashion-MNIST.

    Images are resized to image_size x image_size and normalized to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if name == "mnist":
        ds = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif name == "fashion_mnist":
        ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if n_samples is not None and n_samples < len(ds):
        ds = Subset(ds, list(range(n_samples)))

    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=True)


def extract_all_images(loader: DataLoader) -> torch.Tensor:
    """Iterate through a DataLoader and return all images as a single tensor."""
    images = []
    for x, _ in loader:
        images.append(x)
    return torch.cat(images, dim=0)


def flat_image_dim(image_size: int = 28, channels: int = 1) -> int:
    return channels * image_size * image_size
