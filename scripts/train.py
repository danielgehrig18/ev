import argparse

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import wandb
import os
import random
import pypose as pp
from ev.models.network import ModelTransformerSE3, ModelSE3GPT
from ev.dataset import Dataset
from pathlib import Path
from mingpt.trainer import Trainer

from ev.dataset import Dataset
from ev.se3 import se3_log_map

def to_device(samples, device="cuda:0"):
    if type(samples) is torch.Tensor:
        return samples.to(device)
    return {k: to_device(v) for k, v in samples.items()}

def se3_log_torch(delta_T):
    #return se3_log_map(delta_T)
    return pp.mat2SE3(delta_T, check=False).Log().matrix()

def train(loader, model, optimizer=None, log_every=-1):
    loss_aggr = 0

    for i, samples in enumerate(tqdm.tqdm(loader)):
        if optimizer is not None:
            optimizer.zero_grad()

        samples = to_device(samples)
        Delta_T_pred = model(samples["samples"])

        Delta_T_gt_inv = samples["target_inv"]
        error = se3_log_torch(Delta_T_gt_inv @ Delta_T_pred)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-capacity", type=float, default=1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=200)

    parser.add_argument("--training-root", type=Path, default="/home/dgehrig/Documents/projects/ev/ev/data/train_spline")
    parser.add_argument("--validation-root", type=Path, default="/home/dgehrig/Documents/projects/ev/ev/data/valid_spline")

    parser.add_argument("--sampling-type", type=str, default="regular")
    parser.add_argument("--use-frame", action="store_true")
    args = parser.parse_args()

    return args




if __name__ == '__main__':
    set_seed(42)

    args = FLAGS()
    wandb.init(project="ev_se3", config=vars(args))

    #train_config = Trainer.get_default_config()
    #train_config.learning_rate = args.learning_rate  # many possible options, see the file

    # setup the optimizer
    #model_config = ModelSE3GPT.get_default_config()
    #model_config.model_type = 'gpt2'
    #model_config.vocab_size = 9 + (12 if args.use_frame else 0)  # openai's model vocabulary
    #model_config.block_size = 1024  # openai's model block_size (i.e. input context length)

    #model = ModelSE3GPT(model_config, use_frame=args.use_frame)
    model = ModelTransformerSE3(
        use_frame=args.use_frame,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        model_capacity=args.model_capacity
    )

    model = model.cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    #optimizer = model.configure_optimizers(train_config)

    wandb.watch(model, log="all", log_freq=100)

    training_dataset = Dataset(root=args.training_root, split="train", sampling_type=args.sampling_type,
                               use_frame=args.use_frame)
    validation_dataset = Dataset(root=args.validation_root, split="valid", sampling_type=args.sampling_type,
                                 use_frame=args.use_frame)

    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, worker_init_fn=worker_init_fn)

    min_error = torch.inf

    for e in range(args.num_epochs):
        error = val(test_loader, model)

        if error < min_error:
            min_error = error
            print("Min Error ", min_error)

        wandb.log({"tot/val": error})
        #with torch.autograd.detect_anomaly(True):
        error = train(train_loader, model, optimizer, log_every=10)

