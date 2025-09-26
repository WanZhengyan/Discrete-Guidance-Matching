import argparse
import numpy as np
import torch
from torch import nn, Tensor
from flow_matching.path.scheduler.scheduler import SchedulerOutput, ConvexScheduler
from flow_matching.utils import ModelWrapper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all") # OpenAI gym environment name
    parser.add_argument("--expid", default="toy")  # Experiment ID
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--device", default="cpu", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=3.0)        # gamma parameter in the paper
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--method', type=str, default="rate_based")
    parser.add_argument('--schedule', type=str, default="KineticEnergy")  
    parser.add_argument('--vocab_size', type=int, default=33)  # vocabulary size for the toy dataset
    parser.add_argument('--source_distribution', type=str, default="uniform")  # source distribution for the toy dataset
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args

def get_big_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="guidance_model") # OpenAI gym environment name
    parser.add_argument("--expid", default="big_toy")  # Experiment ID
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--device", default="cpu", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=3.0)        # gamma parameter in the paper
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--method', type=str, default="posterior_based")
    parser.add_argument('--schedule', type=str, default="KineticEnergy")  
    parser.add_argument('--vocab_size', type=int, default=128)  # vocabulary size for the toy dataset
    parser.add_argument('--source_distribution', type=str, default="uniform")  # source distribution for the toy dataset
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args

class KOConvexScheduler(ConvexScheduler):
    """KO Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=torch.cos(0.5 * torch.pi * (1 - t)) ** 2,
            sigma_t=torch.sin(0.5 * torch.pi * (1 - t)) ** 2,
            d_alpha_t=0.5 * torch.pi * torch.sin(torch.pi * (1 - t)),
            d_sigma_t=0 - 0.5 * torch.pi * torch.sin(torch.pi * (1 - t)),
        )


    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return 1 - torch.arccos(torch.sqrt(kappa)) * 0.5 / torch.pi
    

class WrappedModel(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)

class DensityRatioWrapper(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, **extras):
        ratio = (1 - self.model(x)) / (self.model(x) + 1e-16)
        return ratio
    
class GuidanceModelWrapper(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.exp(self.model(x, t))
    
class LogitsModelWrapper(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.log(self.model(x, t) + 1e-16)


def weighted_conditional_prob(x, t, wrapped_probability_denoiser, density_ratio_means_model):
    """
    x: (batch, length)
    t: (batch,)
    return: (batch, length, vocab_size)
    """
    probs = wrapped_probability_denoiser(x, t)  # (batch, length, vocab_size)
    out = torch.zeros_like(probs)
    dr = density_ratio_means_model(x, t)  # (batch, length, vocab_size)

    out = dr * probs + 1e-16  # (batch, length, vocab_size)
    out = out / (out.sum(dim=-1, keepdim=True))
    return out

class WrappedConditionalProb:
    def __init__(self, wrapped_probability_denoiser, guidance_model):
        self.wrapped_probability_denoiser = wrapped_probability_denoiser
        self.guidance_model = guidance_model

    @torch.no_grad()
    def __call__(self, x, t):
        return weighted_conditional_prob(
            x, t, self.wrapped_probability_denoiser, self.guidance_model
        )

def get_nearest_times(time_grid: Tensor, t_discretization: Tensor) -> Tensor:
    distances = torch.cdist(
        time_grid.unsqueeze(1),
        t_discretization.unsqueeze(1),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    nearest_indices = distances.argmin(dim=1)

    return t_discretization[nearest_indices]
