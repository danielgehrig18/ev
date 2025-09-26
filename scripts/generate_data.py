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
    parser = argparse.ArgumentParser("""Generate spline data with different motions and speed profiles.""")
    parser.add_argument("--num-samples", default=10000, type=int)
    #parser.add_argument("--unique-trajectory-fraction", default=0.2, type=float)
    parser.add_argument("--output-directory", default="/home/dgehrig/Documents/projects/ev/ev/data/valid_spline", type=Path)
    parser.add_argument("--pose-range", default=2, type=float)
    parser.add_argument("--num-poses", default=5, type=int)
    parser.add_argument("--num-tokens", default=100, type=int)
    parser.add_argument("--velocity-range", default=2, type=float)
    parser.add_argument("--acceleration-range", default=2, type=float)
    parser.add_argument("--threshold", default=0.2, type=float)
    parser.add_argument("--sampling-type", default="path")
    parser.add_argument("--time-range", nargs="+", default=[0, 10], type=int)
    parser.add_argument("--random-seed", default=1000000, type=int)
    parser.add_argument("--num-processes", default=8, type=int)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # For hash-based operations
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_sample(dataset, i, output_path):
    set_seed(i)
    torch.save(dataset.generate_sample(), output_path)


if __name__ == '__main__':
    args = FLAGS()
    set_seed(args.random_seed)

    datagen = DataGen(time_range=args.time_range,
                      n_poses=args.num_poses,
                      num_tokens=args.num_tokens,
                      pose_range=args.pose_range ,
                      velocity_range=args.velocity_range,
                      acceleration_range=args.acceleration_range,
                      threshold=args.threshold,
                      sampling_type=args.sampling_type)

    args.output_directory.mkdir(exist_ok=True, parents=True)
    with (args.output_directory / "config.yaml").open("w") as fh:
        config = copy.copy(vars(args))
        config["output_directory"] = str(config["output_directory"])
        yaml.dump(config, stream=fh)

    (args.output_directory / "data").mkdir(exist_ok=True, parents=True)
    with TaskManager(total=args.num_samples, processes=args.num_processes, queue_size=8) as tm:
        for i in range(args.num_samples):
            tm.new_task(generate_sample, datagen, args.random_seed + i+1, args.output_directory / ("data/%06d.pth" % i))
            #generate_sample(datagen, args.output_directory / ("data/%06d.pth" % i))
