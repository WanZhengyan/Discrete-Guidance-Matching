import time
import torch

from torch import nn, Tensor

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

# data
from sklearn.datasets import make_moons
from torch.utils.data import Dataset, DataLoader

# visualization
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def MoonDistribution(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> Tensor:
    assert n_grid_points % 4 == 0, "number of grid points has to be divisible by 4"
    x_1 = Tensor(make_moons(batch_size, noise=0.05)[0])
    x_1 = torch.round(torch.clip(x_1 * 35 + 50, min=0.0, max=n_grid_points - 1))
    return x_1.long()

def TransMoonDistribution(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> Tensor:
    assert n_grid_points % 4 == 0, "number of grid points has to be divisible by 4"
    total_samples = batch_size * 5
    X, y = make_moons(total_samples, noise=0.05)
    X = torch.tensor(X)
    y = torch.tensor(y)
    n1 = int(batch_size * 9 / 10)
    n0 = batch_size - n1
    idx1 = (y == 1).nonzero(as_tuple=True)[0]
    idx0 = (y == 0).nonzero(as_tuple=True)[0]
    idx1 = idx1[torch.randperm(len(idx1))[:n1]]
    idx0 = idx0[torch.randperm(len(idx0))[:n0]]
    idx = torch.cat([idx1, idx0])
    X = X[idx]
    X = torch.round(torch.clip(X * 35 + 50, min=0.0, max=n_grid_points - 1))
    X = X[torch.randperm(len(X))]
    return X.long()

def MixGaussianDistribution(n_grid_points: int = 128, batch_size: int = 200, n_components: int = 8, std: float = 8.0, device: str = "cpu") -> Tensor:
    center_xy = n_grid_points / 2 - 0.5 
    radius = n_grid_points * 0.35       
    angles = torch.linspace(0, 2 * torch.pi, steps=n_components + 1)[:-1]
    centers = torch.stack([
        center_xy + radius * torch.cos(angles),
        center_xy + radius * torch.sin(angles)
    ], dim=1).to(device)
    centers = centers.float()

    comp_ids = torch.randint(0, n_components, size=(batch_size,), device=device)
    samples = torch.normal(
        mean=centers[comp_ids],
        std=std
    )
    samples = torch.round(samples).clamp(0, n_grid_points - 1).long()
    return samples

def CheckerboardDistribution(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> Tensor:
    assert n_grid_points % 4 == 0, "number of grid points has to be divisible by 4"
    n_grid_points = n_grid_points // 4
    x1 = torch.randint(low=0, high=n_grid_points * 4, size=(batch_size,), device=device)
    samples_x2 = torch.randint(low=0, high=n_grid_points, size=(batch_size,), device=device)
    x2 = (
        samples_x2
        + 2 * n_grid_points
        - torch.randint(low=0, high=2, size=(batch_size,), device=device) * 2 * n_grid_points
        + (torch.floor(x1 / n_grid_points) % 2) * n_grid_points
    )
    x_end = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1)
    return x_end.long()

def get_distribution_tensor(name, n_grid_points=128, batch_size=200, device="cpu"):
    if name.lower() == "moon":
        return MoonDistribution(n_grid_points, batch_size, device=device)
    elif name.lower() == "transmoon":
        return TransMoonDistribution(n_grid_points, batch_size, device=device)
    elif name.lower() == "mixgaussian":
        return MixGaussianDistribution(n_grid_points, batch_size, device=device)
    elif name.lower() == "checkerboard":
        return CheckerboardDistribution(n_grid_points, batch_size, device=device)
    else:
        raise ValueError(f"Unknown distribution name: {name}")

def get_loaders(names, n_samples=1000000, batch_size=200, n_grid_points=128, device="cpu"):
    if isinstance(names, str):
        tensor = get_distribution_tensor(names, n_grid_points, n_samples, device)
        loader = DataLoader(torch.utils.data.TensorDataset(tensor), batch_size=batch_size, shuffle=True)
        return loader
    else:
        tensors = [get_distribution_tensor(name, n_grid_points, n_samples, device) for name in names]
        loaders = [DataLoader(torch.utils.data.TensorDataset(tensor), batch_size=batch_size, shuffle=True) for tensor in tensors]
        return loaders

# Example usage:
# loader1, loader2 = get_loaders(["checkerboard", "mixgaussian"])
