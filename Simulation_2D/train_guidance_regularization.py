import os
import tqdm
import time
import torch
import wandb

from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_big_toy import *
from model_toy import *
from utils import *

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler.scheduler import SchedulerOutput, ConvexScheduler

# visualization
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def train_source_model(args, pretrained_model, data_loader, info, start_epoch=0):
    """
    Train the source model.
    """
    n_epochs = 50
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    optimizer = Adam(pretrained_model.parameters(), lr=1e-4)
    scheduler = KOConvexScheduler()  # KO scheduler
    path = MixtureDiscreteProbPath(scheduler=scheduler) # mixture discrete path
    loss_fn = MixturePathGeneralizedKL(path=path) # loss function
    
    # info
    vocab_size = info["vocab_size"]
    mask_token = info["mask_token"]
    added_token = info["added_token"]

    epsilon = 1e-3 # early stopping threshold

    for epoch in tqdm_epoch:
        avg_loss = 0
        num_items = 0
        # training behavior
        for data in data_loader:
            x = data[0]
            if args.source_distribution == "uniform":
                x_0 = torch.randint_like(x, high=vocab_size)
            elif args.source_distribution == "mask":
                x_0 = torch.zeros_like(x) + mask_token
            else:
                raise NotImplementedError
            t = torch.rand(x.shape[0]).to(args.device) * (1 - epsilon)
            # sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x)
            logits = pretrained_model(x=path_sample.x_t, t=path_sample.t)
            loss = loss_fn(logits=logits, x_1=x, x_t=path_sample.x_t, t=path_sample.t)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        if epoch % 50 == 49 and args.save_model:
            torch.save(pretrained_model.state_dict(), os.path.join("./models", str(args.expid), "source_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("source/loss", avg_loss / num_items, global_step=epoch)




def train_density_ratio(args, density_ratio_model, data_loader_p, data_loader_q, info, start_epoch=0):
    """
    Train the density ratio model.
    """
    n_epochs = 50
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    optimizer = Adam(density_ratio_model.parameters(), lr=5e-4)

    # info
    vocab_size = info["vocab_size"]
    mask_token = info["mask_token"]
    added_token = info["added_token"]
    
    eta = 1e-2 # noise level for source distribution; learning on low density region more accurately
    for epoch in tqdm_epoch:
        avg_loss = 0
        num_items = 0
        # training behavior
        for data_q, data_p in zip(data_loader_q, data_loader_p):
            x_p = data_p[0]
            x_q = data_q[0]
            rand_tensor = torch.randint(low=0, high=vocab_size - added_token, size=x_p.shape, device=args.device)
            row_mask = (torch.rand(x_p.shape[0], device=args.device) < eta) 
            row_mask = row_mask.unsqueeze(1) 
            x_p = torch.where(row_mask, rand_tensor, x_p)
            out_source = density_ratio_model(x_p)
            out_target = density_ratio_model(x_q)

            loss = - torch.log(out_source + 1e-8).mean() - torch.log(1 - out_target + 1e-8).mean()
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x_p.shape[0]
            num_items += x_p.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        if epoch % 50 == 49 and args.save_model:
            torch.save(density_ratio_model.state_dict(), os.path.join("./models", str(args.expid), "density_ratio_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

def train_guidance_model(args, guidance_model, data_loader_p, data_loader_q,
                         wrapped_probability_denoiser, wrapped_density_ratio_model, info, start_epoch=0):
    n_epochs = 101
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    optimizer = Adam(guidance_model.parameters(), lr=1e-4)
    scheduler = KOConvexScheduler()  # KO scheduler
    path = MixtureDiscreteProbPath(scheduler=scheduler) # mixture discrete path

    # info
    vocab_size = info["vocab_size"]
    mask_token = info["mask_token"]
    added_token = info["added_token"]
    loss_tuning = info["loss_tuning"]
    
    epsilon = 1e-3 # early stopping threshold
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        # training behavior
        for data_q, data_p in zip(data_loader_q, data_loader_p):
            x_p = data_p[0]
            x_q = data_q[0]

            if args.source_distribution == "uniform":
                x_0_p = torch.randint_like(x_p, high=vocab_size)
                x_0_q = torch.randint_like(x_q, high=vocab_size)
            elif args.source_distribution == "mask":
                x_0_p = torch.zeros_like(x_p) + mask_token
                x_0_q = torch.zeros_like(x_q) + mask_token
            else:
                raise NotImplementedError
            
            # sample time (user's responsibility)
            t_p = torch.rand(x_p.shape[0]).to(args.device) * (1 - epsilon)
            t_q = torch.rand(x_q.shape[0]).to(args.device) * (1 - epsilon)

            # sample probability path
            path_sample_p = path.sample(t=t_p, x_0=x_0_p, x_1=x_p)
            path_sample_q = path.sample(t=t_q, x_0=x_0_q, x_1=x_q)

            
            h_p = guidance_model(path_sample_p.x_t, t_p) # shape: (batch, length, vocab_size)
            h_q = guidance_model(path_sample_q.x_t, t_q) # shape: (batch, length, vocab_size)
            h_p = h_p.gather(dim=2, index=x_p.unsqueeze(-1)).squeeze(-1)  # shape: (batch, length)
            h_q_agg = wrapped_probability_denoiser(path_sample_q.x_t, t_q) * torch.exp(h_q)
            h_q = h_q.gather(dim=2, index=x_q.unsqueeze(-1)).squeeze(-1)  # shape: (batch, length)
            h_q_agg = h_q_agg.sum(dim=-1)  # shape: (batch, length)
            loss_p = ((torch.exp(h_p) - wrapped_density_ratio_model(x_p) * h_p)).mean()
            loss_q = ((torch.log(h_q_agg + 1e-8) - h_q)).mean()
            loss = loss_p + loss_tuning * loss_q
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * x_p.shape[0]
            num_items += x_p.shape[0]
        wandb.log({"loss": avg_loss / num_items, "epoch": epoch})
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        if epoch % 50 == 49 and args.save_model:
            save_dir = os.path.join("./models", str(args.expid), str(args.env), f"{loss_tuning:.1f}".replace('.', '_'))
            os.makedirs(save_dir, exist_ok=True) 
            torch.save(guidance_model.state_dict(), os.path.join(save_dir, "guidance_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)


def main(args):
    for dir in ["./models", "./toylogs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", str(args.expid))):
        os.makedirs(os.path.join("./models", str(args.expid)))
    writer = SummaryWriter("./toylogs/" + str(args.expid))
    
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
    if args.env == "source_model":
        print("Training source model.")
        data_loader = get_loaders(names="mixgaussian", batch_size=2048, n_grid_points=vocab_size - added_token, device=args.device)
        pretrained_model = ToyMLP(vocab_size=vocab_size, hidden_dim=256).to(args.device)
        train_source_model(args, pretrained_model, data_loader, info, start_epoch=0)
    elif args.env == "density_ratio":
        print("Training density ratio model.")
        data_loader_p = get_loaders(names="mixgaussian", batch_size=2048, n_grid_points=vocab_size - added_token, device=args.device)
        data_loader_q = get_loaders(names="moon", batch_size=2048, n_grid_points=vocab_size - added_token, device=args.device)
        density_ratio_model = DensityRatio(vocab_size=vocab_size, hidden_dim=256).to(args.device)
        train_density_ratio(args, density_ratio_model, data_loader_p, data_loader_q, info, start_epoch=0)
    elif args.env == "guidance_model":
        for loss_tuning in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            if wandb.run is not None:
                wandb.finish()
            wandb.init(project="Transfer_Guidance_Toy", name=f"{loss_tuning}_guidance")
            print("Training guidance model.")
            info["loss_tuning"] = loss_tuning
            pretrained_model = ToyMLP(vocab_size=vocab_size, hidden_dim=256).to(args.device)
            density_ratio_model = DensityRatio(vocab_size=vocab_size, hidden_dim=256).to(args.device)
            pretrained_model.load_state_dict(torch.load(os.path.join("./models", str(args.expid), "source_ckpt50.pth")))
            density_ratio_model.load_state_dict(torch.load(os.path.join("./models", str(args.expid), "density_ratio_ckpt50.pth")))
            wrapped_density_ratio_model = DensityRatioWrapper(density_ratio_model)
            wrapped_probability_denoiser = WrappedModel(pretrained_model)
            data_loader_p = get_loaders(names="mixgaussian", batch_size=2048, n_grid_points=vocab_size - added_token, device=args.device)
            data_loader_q = get_loaders(names="moon", batch_size=2048, n_grid_points=vocab_size - added_token, device=args.device)
            guidance_model = PosteriorGuidance(vocab_size=vocab_size, hidden_dim=256).to(args.device)
            # guidance_model.load_state_dict(torch.load(os.path.join(os.path.join("./models", str(args.expid), str(args.env), f"{loss_tuning:.1f}".replace('.', '_')), "guidance_ckpt50.pth")))
            train_guidance_model(args, guidance_model, data_loader_p, data_loader_q, 
                                wrapped_probability_denoiser, wrapped_density_ratio_model, info, start_epoch=0)

if __name__ == "__main__":
    args = get_big_args()
    main(args)
