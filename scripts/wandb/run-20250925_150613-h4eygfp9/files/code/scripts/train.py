import argparse

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import wandb
import os
import random
from ev.models.network import ModelTransformerSE3
from ev.dataset import Dataset
from pathlib import Path

from ev.dataset import Dataset

def to_device(samples, device="cuda:0"):
    if type(samples) is torch.Tensor:
        return samples.to(device)
    return {k: to_device(v) for k, v in samples.items()}

def so3_log_torch(delta_R):
    trace = torch.einsum("...ii->...", delta_R)
    cos_theta = torch.clamp(0.5 * (trace - 1), -1, 1)
    theta = torch.arccos(cos_theta)

    sinc = torch.sin(theta) / theta
    sinc[theta.abs()<1e-4] = 1

    w = 0.5 / sinc[:,None,None] * (delta_R - delta_R.mT)
    w = w[...,[2,0,1],[1,2,0]]

    return w

def so3_J_l_inv_torch(r):
    r_hat = so3_hat_torch(r)
    norm = torch.norm(r, dim=-1)
    k = (1 / norm**2 - (1 + torch.cos(norm)) / (2 * norm * torch.sin(norm)))
    I = torch.eye(3, device=r.device)
    J_inv = I[None] - 0.5 * r_hat + k[...,None,None] * r_hat @ r_hat
    J_inv[norm < 1e-4] = I[None] - 0.5 * r_hat[norm < 1e-4]
    return J_inv

def so3_hat_torch(r):
    out = torch.zeros(size=(r.shape[:-1] + (3,3)), device=r.device)
    out[...,2,1] = r[...,0]
    out[...,1,0] = r[...,2]
    out[...,0,2] = r[...,1]

    out[...,1,2] = -r[...,0]
    out[...,0,1] = -r[...,2]
    out[...,2,0] = -r[...,1]

    return out

def se3_log_torch(delta_T):
    delta_R = delta_T[...,:3,:3]

    delta_r = so3_log_torch(delta_R)
    delta_t = so3_J_l_inv_torch(delta_r) @ delta_T[...,:3, 3:4]

    return torch.cat([delta_r, delta_t[...,0]], dim=-1)

def train(loader, model, optimizer=None, log_every=-1):
    loss_aggr = 0

    for i, samples in enumerate(tqdm.tqdm(loader)):
        if optimizer is not None:
            optimizer.zero_grad()

        samples = to_device(samples)
        Delta_T_pred = model(timestamps=samples["timestamps"],
                             samples=samples["samples"])

        Delta_T_gt = samples["target"]
        error = se3_log_torch(torch.linalg.inv(Delta_T_gt) @ Delta_T_pred)
        loss = error.pow(2).sum(-1).mean()

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)

            optimizer.step()
        else:
            loss_aggr = loss_aggr + loss

        if log_every > 0 and i % log_every == 0:
            wandb.log({"tot/train": loss})

    loss_aggr /= len(loader)

    return loss_aggr

def val(loader, model):
    model.eval()
    with torch.no_grad():
        ret = train(loader, model, optimizer=None)
    model.train()
    return ret

def plot_events(ax, timestamps, values):
    for t0, t1, v0, v1 in zip(timestamps, timestamps[1:], values, values[1:]):
        ax.plot([t0, t1], [v0, v0], color="r")
        ax.plot([t1, t1], [v0, v1], color="b")

def worker_init_fn(worker_id):
    np.random.seed(np.random.randint(2**32))


def set_seed(seed):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds for CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # If you are using CuDNN, you can set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def FLAGS():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=5)

    parser.add_argument("--training-root", type=Path, default="/home/dgehrig/Documents/projects/ev/ev/data/train_spline")
    parser.add_argument("--validation-root", type=Path, default="/home/dgehrig/Documents/projects/ev/ev/data/valid_spline")

    parser.add_argument("--sampling", type=str, default="regular")
    args = parser.parse_args()

    return args




if __name__ == '__main__':
    set_seed(42)

    args = FLAGS()
    wandb.init(project="ev", config=vars(args))

    model = ModelTransformerSE3(f=1)
    model = model.cuda()

    wandb.watch(model, log="all", log_freq=100)

    training_dataset = Dataset(root=args.training_root)
    validation_dataset = Dataset(root=args.validation_root)

    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    min_error = torch.inf

    for e in range(args.num_epochs):
        error = val(test_loader, model)

        if error < min_error:
            min_error = error
            print("Min Error ", min_error)

        wandb.log({"tot/val": error})
        error = train(train_loader, model, optimizer, log_every=10)

