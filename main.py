import argparse

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import random
from ema_pytorch import EMA
from models import ModelTransformer, ModelTransformerJagged, ReconstructionModel
from data import SinDataset
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

    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--time_window_s", type=float, default=1)

    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--set-middle-to-zero", action="store_true")
    parser.add_argument("--fixed-events", action="store_true")
    parser.add_argument("--aligned-gt", action="store_true")
    parser.add_argument("--perturb-window-by-rate", action="store_true")

    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--sampling", type=str, default="regular")
    parser.add_argument("--dimensions", type=int, default=1)
    parser.add_argument("--precision", type=int, default=10000)

    args = parser.parse_args()

    return args




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    set_seed(42)

    args = FLAGS()
    wandb.init(config=vars(args))

    B = args.batch_size
    S = args.num_samples
    T = args.time_window_s
    R = S // T

    #model = ModelCNN(f=1, batch_size=B, sequence_length=S, batch_for_loop=False)
    model = ModelTransformer(f=1, batch_size=B, sequence_length=S,
                             dimensions=args.dimensions)
    model = model.cuda()
    #model = torch.compile(model)

    #model = torch.compile(model)

    ema = EMA(
        model,
        beta=0.9999,  # exponential moving average factor
        update_after_step=1,  # only after this number of .update() calls will it start updating
        update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    wandb.watch(model, log="all", log_freq=100)

    train_dataset = SinDataset(num_samples=1000000, time_window_s=T, threshold=args.threshold,
                               split="train", sample_rate_hz=R,
                               set_middle_to_zero=args.set_middle_to_zero,
                               fixed_events=args.fixed_events,
                               aligned_gt=args.aligned_gt,
                               sampling=args.sampling,
                               dimensions=args.dimensions,
                               precision=args.precision)

    test_dataset = SinDataset(num_samples=10000, time_window_s=T, threshold=args.threshold,
                              split="test", sample_rate_hz=R,
                              set_middle_to_zero=args.set_middle_to_zero,
                              fixed_events=args.fixed_events,
                              aligned_gt=args.aligned_gt,
                              sampling=args.sampling,
                              dimensions=args.dimensions,
                              precision=args.precision)

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=SinDataset.collate, num_workers=8, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, collate_fn=SinDataset.collate, num_workers=8, worker_init_fn=worker_init_fn)


    output = test_dataset[1]

    #for i in tqdm.tqdm(range(len(test_dataset))):
    #    test_dataset[i]
    #test_dataset.debug_2d(1)
    #train_dataset.debug_2d(1)

    num_events = []
    for batch_size, max_num_events, counter, batch, timestamps, x, displacement in test_loader:
        num_events.extend([(batch == b).sum() for b in range(batch_size)])

    print("Mean num events test set ", np.mean(num_events))
    print("Median num events test set ", np.median(num_events))

    wandb.log({"mean_num_events": np.mean(num_events), "median_num_events": np.median(num_events)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    min_error = torch.inf

    for e in range(args.num_epochs):
        error = val(test_loader, ema)

        if error < min_error:
            min_error = error
            print("Min Error ", min_error)

        wandb.log({"tot/val": error})
        error = train(train_loader, model, optimizer, ema=ema, log_every=10)

