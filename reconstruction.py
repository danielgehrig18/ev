import argparse

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import random
from models import ReconstructionModel
from data import SinDataset


def plot_signal(signal, signal_t, x, t, b):
    a = 2

    fig, ax = plt.subplots(nrows=x.shape[1])
    for i in range(x.shape[1]):
        ax[i].plot(signal_t.cpu().numpy(), signal[0,:,i].cpu().numpy())
        ax[i].scatter(t[b==0].cpu().numpy(), x[b==0,i].cpu().numpy())

    if x.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.plot(signal[0,:,1].cpu().numpy(), signal[0,:,1].cpu().numpy())
        ax.scatter(x[b==0,0].cpu().numpy(), x[b==0,1].cpu().numpy())

    return fig

def val(loader, model, debug=False):
    for i, (batch_size, max_num_events, counter, batch, timestamps, x, displacement, params) in enumerate(tqdm.tqdm(loader)):

        x = x.cuda()
        batch = batch.cuda()
        timestamps = timestamps.cuda()
        displacement = displacement.cuda()
        counter = counter.cuda()

        gt_x, gt_timestamps = loader.dataset.generate_gt_signals(params, resolution=10000)

        gt_x = torch.from_numpy(gt_x).cuda()
        gt_timestamps = torch.from_numpy(gt_timestamps).cuda()

        #x = torch.stack([timestamps, x], dim=1)
        signal, timestamps_signal = model(batch_size, max_num_events, counter, batch, timestamps, x,
                                          gt_timestamps=gt_timestamps, gt_x=gt_x)

        if debug:
            fig = plot_signal(signal, timestamps_signal, x, timestamps, batch)
            plt.show()

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
    parser.add_argument("--batch-size", type=int, default=256)
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

    B = args.batch_size
    S = args.num_samples
    T = args.time_window_s
    R = S // T

    #model = ModelCNN(f=1, batch_size=B, sequence_length=S, batch_for_loop=False)
    model = ReconstructionModel(batch_size=B, sequence_length=S, dimensions=args.dimensions, bandwidth=10,
                                resolution=10000, time_window_s=T, threshold=args.threshold)
    model = model.cuda()
    model.eval()

    test_dataset = SinDataset(num_samples=10000, time_window_s=T, threshold=args.threshold,
                              split="test", sample_rate_hz=R,
                              set_middle_to_zero=args.set_middle_to_zero,
                              fixed_events=args.fixed_events,
                              aligned_gt=args.aligned_gt,
                              sampling=args.sampling,
                              dimensions=args.dimensions,
                              precision=args.precision,
                              return_parameters=True)

    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, collate_fn=SinDataset.collate, num_workers=8, worker_init_fn=worker_init_fn)


    num_events = []
    #for batch_size, max_num_events, counter, batch, timestamps, x, displacement, params in tqdm.tqdm(test_loader):
    #    num_events.extend([(batch == b).sum() for b in range(batch_size)])

    print("Mean num events test set ", np.mean(num_events))
    print("Median num events test set ", np.median(num_events))

    with torch.no_grad():
        error = val(test_loader, model, debug=True)
        print(error)
