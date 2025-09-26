import tqdm
import yaml
import torch
import os
import random

import numpy as np
from pathlib import Path

from ev.sample import Sampler, random_spline
from ev.process import compute_frame
from ev.spline import generate_random_phi


class DataGen:
    def __init__(self, time_range, n_poses, num_tokens,
                 pose_range, velocity_range, acceleration_range, threshold=0.8,
                 sampling_type="regular"):
        self.n_poses = n_poses
        self.time_range = time_range
        self.pose_range = pose_range
        self.velocity_range = velocity_range
        self.acceleration_range = acceleration_range
        self.threshold = threshold
        self.num_tokens = num_tokens
        self.sampling_type = sampling_type

        self.sampler = Sampler(threshold=self.threshold, num_tokens=self.num_tokens, sampling_type=self.sampling_type,
                               time_range=self.time_range)

    def generate_sample(self):
        params = dict(n_poses=self.n_poses, pose_range=self.pose_range, velocity_range=self.velocity_range,
                      acceleration_range=self.acceleration_range, time_range=self.time_range)
        spline = random_spline(**params)
        t_sample, spline = self.sampler.sampling_timestamps(spline, **params)
        samples = spline.sample(t_sample)

        output = {
            "samples": samples,
            "timestamps": t_sample,
            "spline": spline
        }

        assert len(t_sample) == self.num_tokens, (len(t_sample), self.num_tokens)

        return output


def augment_samples_with_frames(samples):
    rotation_frames_orth, translation_frames_orth = compute_frame(samples)
    samples["rotation_frames_orth"] = rotation_frames_orth
    samples["translation_frames_orth"] = translation_frames_orth
    return samples



class Dataset:
    def __init__(self, root: Path, split="train", sampling_type="event_based"):
        self.root = root
        self.split = split

        self.files = sorted(root.glob("data/*.pth"))
        self.config_path = root / "config.yaml"
        self.sampling_type = sampling_type

        with self.config_path.open("r") as fh:
            self.config = yaml.load(fh, Loader=yaml.SafeLoader)

    def __getitem__(self, item):
        data = torch.load(self.files[item], weights_only=False)

        # compute the invariant frames
        spline = data.pop("spline")
        samples = spline.sample(data["timestamps"], n=3)
        data["samples"] = augment_samples_with_frames(samples)
        num_tokens = len(data["timestamps"])
        t0, t1 = data["timestamps"][[0, -1]]

        if self.split == "train":
            # change phi
            phi_t = spline.phi(data["timestamps"])
            spline.phi = generate_random_phi(time_range=[t0, t1], phi_max=phi_t[-1])
            data["timestamps"] = solve_spline(spline.phi, t0, t1, phi_t)

        # target is relative pose
        T0, T1 = data["samples"]['f'][[0, -1]]
        Delta_T = np.linalg.inv(T1) @ T0
        data["target"] = Delta_T

        if self.sampling_type == "regular":
            timestamps_regular = np.linspace(t0, t1, num=num_tokens, endpoint=True)
            data["timestamps_regular"] = timestamps_regular
            data["samples_regular"] = spline.sample(timestamps_regular)

            data["samples"] = data["samples_regular"]

        data["samples"]["position"] = np.linspace(0, 1, num=num_tokens, endpoint=True)
        data = cast_dtype(data, dtype="float32")

        return data

    def __len__(self):
        return len(self.files)

def solve_spline(spline, t0, t1, phi_t, num=10000):
    t_sample = np.linspace(t0, t1, num=num, endpoint=True)
    phi_sample = spline(t_sample)

    idx = np.clip(np.searchsorted(phi_sample, phi_t) - 1, 0, len(phi_sample)-2)
    s0 = phi_sample[idx]
    s1 = phi_sample[idx+1]

    t0 = t_sample[idx]
    dt = t_sample[1] - t_sample[0]
    t = dt * (phi_t - s0) / (s1 - s0 + 1e-9) + t0

    return t


def cast_dtype(data, dtype):
    if type(data) is np.ndarray:
        return data.astype(dtype)
    return {k: cast_dtype(v, dtype) for k, v in data.items()}


if __name__ == '__main__':
    dataset = Dataset(root=Path("/tmp/train_spline/"), split="train")
    for i in tqdm.tqdm(range(len(dataset))):
        data = dataset[i]