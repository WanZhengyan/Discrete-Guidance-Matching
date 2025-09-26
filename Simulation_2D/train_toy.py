import os
import tqdm
import functools
import ipdb
import torch
from torch import nn, Tensor

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_toy import ToyMLP
from utils import get_args
from utils import KOConvexScheduler
from dataset.dataset import Toy_dataset

# flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path.scheduler.scheduler import SchedulerOutput, ConvexScheduler

def train(args, pretrained_model, data_loader, info, start_epoch=0):
    n_epochs = 100
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    optimizer = Adam(pretrained_model.parameters(), lr=1e-4)
    scheduler = KOConvexScheduler()  # KO scheduler
    path = MixtureDiscreteProbPath(scheduler=scheduler) # mixture discrete path
    loss_fn = MixturePathGeneralizedKL(path=path) # loss function

    # info
    vocab_size = info["vocab_size"]
    mask_token = info["mask_token"]
    added_token = info["added_token"]
    env = info["env"]

    pretrained_model.load_state_dict(torch.load(os.path.join("./models", "toy_{}".format(args.source_distribution),str(env), "ckpt50.pth"), map_location=args.device))
    pretrained_model.eval()
    epsilon = 1e-3 # early stopping threshold

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        # training behavior
        for data in data_loader:
            data = {k: d.to(args.device) for k, d in data.items()}
            x = data["a"]
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
            save_dir = os.path.join("./models", "toy_{}".format(args.source_distribution), str(env))
            os.makedirs(save_dir, exist_ok=True) 
            torch.save(pretrained_model.state_dict(), os.path.join(save_dir, "ckpt{}.pth".format(epoch+1)))
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
        for env in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]:
            info["env"] = env
            print("Training on dataset: ", env)
            dataset = Toy_dataset(name=env, device=args.device)
            data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
            pretrained_model = ToyMLP(vocab_size=vocab_size, hidden_dim=256).to(args.device)
            train(args, pretrained_model, data_loader, info, start_epoch=50)
    else:
        dataset = Toy_dataset(name=args.env) # datanum = 1000000 samples
        data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)
        pretrained_model= ToyMLP(vocab_size=vocab_size, hidden_dim=256).to(args.device)
        print("training")
        train(args, pretrained_model, data_loader, info, start_epoch=0)
        print("finished")

if __name__ == "__main__":
    args = get_args()
    main(args)