import argparse
from pathlib import Path

from ev.dataset import DataGen
import torch
from ev.taskmanager import TaskManager
import yaml
import copy
import numpy as np
import random
import pypose as pp
import tqdm
import os


def FLAGS():
    parser = argparse.ArgumentParser("""Generate path spline data.""")
    #parser.add_argument("--unique-trajectory-fraction", default=0.2, type=float)
    parser.add_argument("--output-directory", default="/home/dgehrig/Documents/projects/ev/ev/data/valid_spline", type=Path)
    parser.add_argument("--num-processes", type=int, default=8)

    args = parser.parse_args()
    return args

def get_path(path):
    data = torch.load(path, weights_only=False)
    spline = data['spline']

    # target is path length
    t_fine_samples = np.linspace(spline.t0, spline.t1, 10000, endpoint=True)
    d1f = spline.sample(t_fine_samples, n=1)['d1f']
    dt = t_fine_samples[1] - t_fine_samples[0]
    data['target'] = np.array([np.sum(np.linalg.norm(d1f, axis=1) * dt)])
    torch.save(data, path)




if __name__ == '__main__':
    args = FLAGS()
    paths = list((args.output_directory / "data").glob("*.pth"))

    with TaskManager(total=len(paths), processes=args.num_processes, queue_size=8) as tm:
        for path in paths:
            tm.new_task(get_path, path)
            #get_path(path)
            #generate_sample(datagen, args.output_directory / ("data/%06d.pth" % i))
