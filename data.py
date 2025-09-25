import numpy as np
from torch.utils.data import Dataset
import  torch
import matplotlib.pyplot as plt
from sampling import event_sample, regular_sample, level_sample, path_sample
import tqdm

def generate_aligned_signal_and_params(threshold, time_window_s):
    w, A, phi, gamma = _generate_random_sin_params(1)[0]
    t, v, d = event_sample(w, A, phi, gamma, threshold, time_window_s)
    w *= ((t[-1]+1e-9)/time_window_s)**gamma
    t, v, d = event_sample(w, A, phi, gamma, threshold, time_window_s)
    return t, w, A, phi, gamma


def generate_until_aligned(threshold, time_window_s, num_events, fixed_events):
    t, w, A, phi, gamma = generate_aligned_signal_and_params(threshold, time_window_s)
    bad = abs(time_window_s - t[-1]) > 1e-6

    while bad:
        t, w, A, phi, gamma = generate_aligned_signal_and_params(threshold, time_window_s)
        bad = abs(time_window_s - t[-1]) > 1e-6

    return w, A, phi, gamma


def generate_sample(threshold, time_window_s, num_events, fixed_events, aligned_gt, dimensions=1):
    params = _generate_random_sin_params(dimensions)
    if not fixed_events:
        if aligned_gt:
            params = generate_until_aligned(threshold, time_window_s, num_events, fixed_events)
        return params

    t, v, d = event_sample(params, threshold, time_window_s)
    n = len(t)
    while n < num_events:
        params = _generate_random_sin_params(dimensions)
        t, v, d = event_sample(params, threshold, time_window_s)
        n = len(t)


    t = t[:num_events]
    w *= ((t[-1]+1e-9)/time_window_s)**gamma

    assert len(event_sample(params, threshold, time_window_s)[0]) == num_events, gamma

    return np.array([*params, num_events])

def _generate_random_sin_params(num_samples):
    w, A, phi, gamma = np.random.rand(4, num_samples).astype("float64")
    phi *= 0.1#2 * np.pi
    A[:] *= 10
    w[:] *= 10
    gamma[:] = 10**((gamma - .5) / .5)
    #num_periods = np.random.randint(low=1, high=4, size=(num_samples,))

    return np.stack([w, A, phi, gamma], axis=-1)


class SinDataset(Dataset):
    def __init__(self, num_samples, time_window_s, sample_rate_hz, threshold, split="train", overfit=False,
                 set_middle_to_zero=False,
                 fixed_events=True,
                 aligned_gt=True,
                 dimensions=1,
                 sampling="regular",
                 precision=10000,
                 return_parameters=False):

        Dataset.__init__(self)
        self.fixed_events = fixed_events
        self.split = split
        self.num_samples = num_samples
        self.time_window_s = time_window_s
        self.sample_rate_hz = sample_rate_hz
        self.overfit = overfit
        self.threshold = threshold
        self.set_middle_to_zero = set_middle_to_zero
        self.aligned_gt = aligned_gt
        self.dimensions = dimensions
        self.sampling = sampling
        self.precision = precision

        self.sin_params_wAphi = None
        self.sampling_randomness = None
        self.return_parameters = return_parameters

        if split == "test":
            np.random.seed(10)
            self.sin_params_wAphi = np.stack([self.generate_sample() for _ in range(num_samples)])
            self.sampling_randomness = np.random.rand(num_samples, dimensions+1)

    def __len__(self):
        return self.num_samples

    def generate_gt_signals(self, params, resolution=10000):
        timestamps = np.linspace(0, self.time_window_s, resolution, endpoint=True)
        signals = np.stack([self.signal(p, timestamps) for p in params.numpy()])
        return signals, timestamps

    def generate_sample(self):
        return generate_sample(threshold=self.threshold,
                               time_window_s=self.time_window_s,
                               num_events=None,
                               fixed_events=self.fixed_events,
                               aligned_gt=self.aligned_gt,
                               dimensions=self.dimensions)

    def signal(self, params, t):
        w, A, phi, gamma = params.T
        values_dense = A[None] * np.sin(w[None] * (t[:,None] / self.time_window_s) ** gamma[None] * self.time_window_s + phi[None])
        return values_dense

    def debug_2d(self, index):
        params = self.generate_sample() if self.split != "test" else self.sin_params_wAphi[index]
        sampling_randomness = np.random.rand(self.dimensions+1) if self.split != "test" else self.sampling_randomness[index]

        timestamps, values, displacement, *axis = self.sample(params, return_axis=True, sampling_randomness=sampling_randomness)

        timestamps_dense = np.linspace(0, self.time_window_s, 10000)
        values_dense = self.signal(params, timestamps_dense)

        fig, ax = plt.subplots()
        ax.plot(values_dense[:,0], values_dense[:,1], color="g")
        ax.scatter(values[:,0], values[:,1], color="r")
        ax.scatter(values[0,0], values[0,1], color="k")

        vx_min, vy_min = np.min(values, axis=0)
        vx_max, vy_max = np.max(values, axis=0)

        if len(axis) > 0: # plot debug stuff for level-based
            for i, (vx, vy) in enumerate(values):
                a = axis[i]
                if a == 0:
                    ax.plot([vx, vx], [vy_min, vy_max], color="b")
                else:
                    ax.plot([vx_min, vx_max], [vy, vy], color="b")
        else:
            for i, (vx, vy) in enumerate(values): # draw circles around each one of them
                angles = np.linspace(0, 2*np.pi, 1000)
                ax.plot(vx + np.cos(angles) * self.threshold, vy + np.sin(angles) * self.threshold, color="b")


        ax.set_aspect('equal')
        plt.show()

    def debug(self, num_examples=1):
        for i in range(num_examples):
            params = self.generate_sample() if self.split != "test" else self.sin_params_wAphi[i]
            sampling_randomness = np.random.rand(self.dimensions) if self.split != "test" else self.sampling_randomness[i]
            timestamps, values, displacement = self.sample(params, sampling_randomness=sampling_randomness)

            fig, ax = plt.subplots()
            for t0, t1, v0, v1 in zip(timestamps, timestamps[1:], values, values[1:]):
                ax.plot([t0, t1], [v0, v0], color="r")
                ax.plot([t1, t1], [v0, v1], color="b")

            timestamps_dense = np.linspace(0, self.time_window_s, 10000)
            ax.plot(timestamps_dense, self.signal(params, timestamps_dense), color="g")

        plt.show()

    def __getitem__(self, index):
        if self.overfit:
            np.random.seed(10)

        params = self.generate_sample() if self.split != "test" else self.sin_params_wAphi[index]
        sampling_randomness = np.random.rand(self.dimensions+1) if self.split != "test" else self.sampling_randomness[index]
        timestamps, values, displacement = self.sample(params, sampling_randomness=sampling_randomness, precision=self.precision)

        if self.set_middle_to_zero:
            values[1:-1] = 0

        ret = [timestamps, values.astype("float32"), displacement]

        if self.return_parameters:
            ret.append(params)

        return ret

    def sample(self, params, sampling_randomness=None, return_axis=False, precision=10000):
        if self.sampling == "regular":
            return regular_sample(params, self.time_window_s, self.sample_rate_hz, sampling_randomness=sampling_randomness)
        elif self.sampling == "variation":
            return path_sample(params, self.threshold, self.time_window_s, sampling_randomness=sampling_randomness)
        elif self.sampling == "level":
            return level_sample(params, self.threshold, self.time_window_s, return_axis=return_axis, sampling_randomness=sampling_randomness)
        elif self.sampling == "event":
            return event_sample(params, self.threshold, self.time_window_s, sampling_randomness=sampling_randomness, precision=precision)
        elif self.sampling == "canonical":
            return canonical_sample(params, self.threshold, self.time_window_s, sampling_randomness=sampling_randomness, precision=precision)

    @staticmethod
    def collate(data_list):
        timestamps = torch.cat([torch.from_numpy(d[0]) for d in data_list])
        values = torch.cat([torch.from_numpy(d[1]) for d in data_list])
        displacement = torch.utils.data.default_collate([d[2] for d in data_list])
        batch = torch.cat([torch.full(size=(len(d[0]),), fill_value=i) for i, d in enumerate(data_list)])
        counter = torch.cat([torch.arange(len(d[0])) for d in data_list])
        max_num_events = max([len(d[0]) for d in data_list])

        ret = [len(data_list), max_num_events, counter, batch, timestamps, values, displacement]

        if len(data_list[0]) > 3:
            params = torch.utils.data.default_collate([d[3] for d in data_list])
            ret.append(params)

        return ret


