import os
import tqdm
import functools
import ipdb
import scipy
import torch
from torch import nn, Tensor
import wandb

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_toy import ToyMLP, RateGuidance 
from utils import get_args
from utils import KOConvexScheduler
from dataset.dataset import Toy_dataset, inf_train_gen

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler.scheduler import SchedulerOutput, ConvexScheduler

def train(args, guidance_model, data_loader, info, start_epoch=0):
    n_epochs = 50
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    optimizer = Adam(guidance_model.parameters(), lr=1e-4)
    scheduler = KOConvexScheduler()  # KO scheduler
    path = MixtureDiscreteProbPath(scheduler=scheduler) # mixture discrete path

    # info
    env = info["env"]
    vocab_size = info["vocab_size"]
    mask_token = info["mask_token"]
    s = info["s"]
    added_token = info["added_token"]
    
    epsilon = 1e-3 # early stopping threshold

    all_exp_e = torch.exp(s * torch.cat([batch["e"] for batch in data_loader]))
    mean_exp_e = all_exp_e.mean() # Calculate mean exp_energy for normalization
    print(env, s, mean_exp_e)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        # training behavior
        for data in data_loader:
            data = {k: d.to(args.device) for k, d in data.items()}
            x_source = data["a"]
            e_source = data["e"]
            if args.source_distribution == "uniform":
                x_0_p = torch.randint_like(x_source, high=vocab_size)
            elif args.source_distribution == "mask":
                x_0_p = torch.zeros_like(x_source) + mask_token
            else:
                raise NotImplementedError
            
            # sample time (user's responsibility)
            t_p = torch.rand(x_source.shape[0]).to(args.device) * (1 - epsilon)

            # sample probability path
            path_sample_p = path.sample(t=t_p, x_0=x_0_p, x_1=x_source)
            
            h_p = guidance_model(path_sample_p.x_t, t_p) # shape: (batch)
            loss_p = ((torch.exp(h_p) - (torch.exp(s * e_source) / mean_exp_e) * h_p)).mean()
            loss = loss_p 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * x_source.shape[0]
            num_items += x_source.shape[0]
        wandb.log({"loss": avg_loss / num_items, "epoch": epoch})
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        if epoch % 25 == 24 and args.save_model:
            save_dir = os.path.join("./models", "toy_{}".format(args.source_distribution), str(env), str(s))
            os.makedirs(save_dir, exist_ok=True) 
            torch.save(guidance_model.state_dict(), os.path.join(save_dir, "ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

def main(args):
    for dir in ["./models", "./toylogs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", "toy_{}".format(args.source_distribution))):
        os.makedirs(os.path.join("./models", "toy_{}".format(args.source_distribution)))
    writer = SummaryWriter("./toylogs/" + "toy_{}".format(args.source_distribution))
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.writer = writer

    if args.source_distribution == "uniform":
        added_token = 0
    elif args.source_distribution == "mask":
        mask_token = args.vocab_size  # tokens starting from zero
        added_token = 1
    else:
        raise NotImplementedError
    vocab_size = args.vocab_size + added_token
    info = {
        "vocab_size": vocab_size,
        "mask_token": mask_token if args.source_distribution == "mask" else None,
        "added_token": added_token,
    }
    if args.env == "all":
        for env in ["moons","swissroll", "8gaussians","rings","checkerboard","2spirals"]:
            for s in [1, 3, 10, 20]:
                print("Training on dataset: ", env, s)
                info["env"] = env
                info["s"] = s
                if wandb.run is not None:
                    wandb.finish()
                wandb.init(project="Rate_Guidance_Toy", name=f"{args.source_distribution}_{env}_{s}_guidance")
                dataset = Toy_dataset(name=env, device=args.device)
                data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
                guidance_model = RateGuidance(vocab_size=vocab_size, hidden_dim=256).to(args.device)
                train(args, guidance_model, data_loader, info, start_epoch=0)
    else:
        for s in [1, 3, 10, 20]:
            print("Training on dataset: ", env, s)
            info["env"] = env
            info["s"] = s
            if wandb.run is not None:
                wandb.finish()
            wandb.init(project="Rate_Guidance_Toy", name=f"{args.source_distribution}_{env}_{s}_guidance")
            dataset = Toy_dataset(name=env, device=args.device)
            data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
            guidance_model = RateGuidance(vocab_size=vocab_size, hidden_dim=256).to(args.device)
            train(args, guidance_model, data_loader, info, start_epoch=0)

if __name__ == "__main__":
    args = get_args()
    main(args)