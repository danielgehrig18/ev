import argparse

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import random
from ema_pytorch import EMA
from models import ModelTransformer
from data import EventSinDataset, SinDataset
import torch_scatter

def train(loader, model, optimizer=None, ema=None, log_every=-1):
    loss_aggr = 0

    for i, (batch_size, max_num_events, counter, batch, timestamps, x, displacement) in enumerate(tqdm.tqdm(loader)):

        if optimizer is not None:
            optimizer.zero_grad()

        x = x.cuda()
        batch = batch.cuda()
        timestamps = timestamps.cuda()
        displacement = displacement.cuda()
        counter = counter.cuda()

        #x = torch.stack([timestamps, x], dim=1)
        displacement_pred = model(batch_size, max_num_events, counter, batch, timestamps, x)

        error = (displacement - displacement_pred).abs()
        loss = error.mean()

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)

            optimizer.step()

            if ema is not None:
                ema.update()

        else:
            loss_aggr = loss_aggr + loss

        if log_every > 0 and i % log_every == 0:
            wandb.log({"tot/train": loss})

    loss_aggr /= len(loader)

    return loss_aggr

def val(loader, model):
    model.eval()
    with torch.no_grad():
        ret = train(loader, model, ema=None, optimizer=None)
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--num_samples", type=int, default=101)
    parser.add_argument("--time_window_s", type=float, default=1)

    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--use-events", action="store_true")
    parser.add_argument("--set-middle-to-zero", action="store_true")
    parser.add_argument("--fixed-events", action="store_true")
    parser.add_argument("--aligned-gt", action="store_true")

    parser.add_argument("--num-epochs", type=int, default=5)

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    set_seed(42)

    args = FLAGS()

    B = args.batch_size
    S = args.num_samples
    T = args.time_window_s
    R = S // T


    if args.use_events:
        train_dataset = EventSinDataset(num_samples=1000000, time_window_s=T, threshold=args.threshold,
                                        split="train", sample_rate_hz=R, num_events=S,
                                        set_middle_to_zero=args.set_middle_to_zero,
                                        fixed_events=args.fixed_events,
                                        aligned_gt=args.aligned_gt)

        test_dataset = EventSinDataset(num_samples=10000, time_window_s=T, threshold=args.threshold,
                                       split="test", sample_rate_hz=R, num_events=S,
                                       set_middle_to_zero=args.set_middle_to_zero,
                                       fixed_events=args.fixed_events,
                                       aligned_gt=args.aligned_gt)
    else:
        train_dataset = SinDataset(num_samples=1000000, time_window_s=T, sample_rate_hz=R,
                                   split="train", overfit=False, num_events=S, threshold=args.threshold,
                                   set_middle_to_zero=args.set_middle_to_zero,
                                   fixed_events=args.fixed_events,
                                   aligned_gt=args.aligned_gt)
        test_dataset = SinDataset(num_samples=10000, time_window_s=T, sample_rate_hz=R,
                                  split="test", overfit=False, num_events=S, threshold=args.threshold,
                                  set_middle_to_zero=args.set_middle_to_zero,
                                  fixed_events=args.fixed_events,
                                  aligned_gt=args.aligned_gt)


    for i in range(len(test_dataset)):
        output = test_dataset[i]
        a=  2