import numpy as np
import matplotlib.pyplot as plt
import ev_cpp
import tqdm

def level_sample_timestamps(params, threshold, time_window_s):
    w, A, phi, gamma = params

    # 1/w (asin(n th / A) - phi)
    init_reference = 0
    n_min = int(np.ceil((- A - init_reference) / threshold))
    n_max = int(np.floor((A - init_reference) / threshold))
    n = np.arange(n_min, n_max + 1)

    # find the values for wt + phi = arcsin ( sin(phi) + n theta / A )
    wt_phi = np.arcsin(np.clip((n * threshold + init_reference) / A, -1, 1))
    wt_phi = np.concatenate([wt_phi, np.pi - wt_phi])
    wt = wt_phi - phi
    wt = np.sort(wt)

    # extend wt by so many 2*pi until the interval is filled.
    if len(wt) == 0:
        return wt

    wt_min = wt[0]
    wt_max = wt[-1]

    wt_min_des = 0
    wt_max_des = w * time_window_s

    n_extend_right = int(np.ceil((wt_max_des - wt_max) / (2 * np.pi)))
    n_extend_left = int(np.ceil((wt_min - wt_min_des) / (2 * np.pi)))

    wt_ = wt.copy()
    if n_extend_right > 0:
        wt = np.concatenate([wt] + [wt_ + i * 2 * np.pi for i in range(1, n_extend_right + 1)])
    if n_extend_left > 0:
        wt = np.concatenate([wt_ - i * 2 * np.pi for i in range(1, n_extend_left + 1)] + [wt])

    wt = wt[(wt >= wt_min_des) & (wt <= wt_max_des)]
    wt = np.sort(wt)

    timestamps = (wt / w) ** (1 / gamma)

    return timestamps

def level_sample(params, threshold, time_window_s, return_gt=False):
    timestamps = np.concatenate([level_sample_timestamps(p, threshold, time_window_s) for p in params])
    timestamps = np.unique(timestamps[timestamps.argsort()])

    def sin(t):
        t = t[:,None]
        w, A, phi, gamma = params.T
        return A[None] * np.sin(w[None] * (t/time_window_s)**gamma[None] * time_window_s + phi[None])

    values = sin(timestamps)
    displacement = np.diff(sin(np.array([time_window_s, 0])), axis=0)[0]

    if return_gt:
        t = np.linspace(0, time_window_s, 1000)
        v = sin(t)
        gt = (t, v)

        return timestamps, values, displacement, gt
    else:
        return timestamps, values, displacement


class ReconstructionModel:
    def __init__(self, bandlimit_hz, resolution, time_window_s, threshold):
        self.bandlimit_hz = bandlimit_hz
        self.resolution = resolution
        self.time_window_s = time_window_s
        self.threshold = threshold
        self.timestamps = np.linspace(0, self.time_window_s, self.resolution, endpoint=True)

        self.S_inv = None
        self.S = None

    @staticmethod
    def plot(gt_t, gt_v, recons, t_sample, v_sample, title="bla"):
        n_signals = v_sample.shape[1]
        fig, ax = plt.subplots(nrows=n_signals)
        for i in range(n_signals):
            ax[i].plot(gt_t, gt_v[:,i], label="GT")
            ax[i].plot(gt_t, recons[:,i], label="Recons")
            ax[i].scatter(t_sample, v_sample[:,i], label="Samples")
            ax[i].legend()
        fig.suptitle(title)
        plt.show()

    @staticmethod
    def whittaker_shannon(t, v, gt_t, gt_v, bandlimit_hz, plot=False):
        # According to Whittaker shannon
        delta_t = 1/(2*bandlimit_hz)
        S = np.sinc((gt_t[:, None] - t[None, :]) / delta_t)
        v_recons = S @ v

        if plot:
            ReconstructionModel.plot(gt_t, gt_v=gt_v, recons=v_recons, t_sample=t, v_sample=v)

        return v_recons

    @staticmethod
    def dirichlet(t, v, gt_t, gt_v, bandlimit_hz, plot=False):
        threshold = 1/(2*bandlimit_hz)

        x = gt_t / threshold # query timestamps
        n = t / threshold    # sample index
        xn = np.pi * (x[:, None] - n[None, :])

        N = len(v)
        S = np.sin(xn) / (N * np.sin(xn / N))
        S[np.abs(xn) < 1e-5] = 1

        if N % 2 == 0:
            S *= np.cos(xn / N)

        v_recons = S @ v

        if plot:
            ReconstructionModel.plot(gt_t, gt_v=gt_v, recons=v_recons, t_sample=t, v_sample=v)

        return v_recons







    def __call__(self, t, v, gt_t, gt_v):

        rmse = lambda gt, est: np.linalg.norm(gt - est, axis=-1).mean()

        v0 = self.init_pocs(t, v, order=0)
        output = self.init_pocs(t, v, order=1)


        self.plot(self.timestamps, output, t, v, title="Init")

        rmses = []

        e = rmse(gt_v, output)
        rmses.append(e)
        print("RMSE: ", e)

        for i in range(100):
            output = self.project_bandlimited_through_points(self.timestamps, output, t, v)
            e = rmse(gt_v, output)
            rmses.append(e)
            print("RMSE: ", e)

            self.plot(self.timestamps, output, t, v, title="After band")

            output = self.project_range(v0, output, self.threshold)
            e = rmse(gt_v, output)
            rmses.append(e)
            print("RMSE: ", e)

            self.plot(self.timestamps, output, t, v, title="After range proj")

        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(rmses)
        ax[0].set_yscale("log")
        ax[1].plot(np.linalg.norm(output - gt_v, axis=-1))
        ax[1].set_yscale("log")
        plt.show()


        return output

    def fit_residuals(self, t, g_w, t_n, x_n, delta_t):
        # compute g_w_n at time stamps t_n via interpolation
        idx = np.searchsorted(t, t_n)
        r = (t_n - t[idx-1]) / (t[idx] - t[idx-1])
        r[idx<0] = 1
        g_w_n = g_w[idx] * r[:,None] + g_w[idx-1] * (1 - r[:,None])

        # find the residuals y_n to measurements x_n
        y_n = x_n - g_w_n

        # compute the coefficients c_n
        if self.S_inv is None:
            damping = 1e-3
            S = np.sinc((t_n[:,None] - t_n[None,:])/delta_t)
            I = np.eye(len(S))
            self.S_inv = np.linalg.pinv(S + I * damping)

        c_n = self.S_inv @ y_n

        # compute the signal at original timestamps t
        if self.S is None:
            self.S = np.sinc((t[:,None] - t_n[None,:])/delta_t)

        y_hat = self.S @ c_n

        return y_hat

    def project_bandlimited_through_points(self, t, g, t_n, v_n):
        fft = np.fft.rfft(g, axis=0, norm="ortho")
        fft[self.bandlimit_hz:] = 0
        g_w = np.fft.irfft(fft, axis=0, norm="ortho")

        delta_t = 1/(2 * self.bandlimit_hz)
        y_hat = self.fit_residuals(t=t, g_w=g_w, t_n=t_n, x_n=v_n, delta_t=delta_t)
        output = g_w + y_hat

        return output

    def project_range(self, v_0, output, threshold, sampling_type="event"):
        n = output - v_0
        if len(n.shape) == 2:
            if sampling_type == "level":
                norm = np.abs(n)
            else:
                norm = np.linalg.norm(n, axis=-1, keepdims=True)
        else:
            norm = np.abs(n)
        mask = norm[:,0] > threshold
        output[mask] = v_0[mask] + n[mask] / norm[mask] * threshold
        return output

    def init_pocs(self, t, v, order=0):
        index = np.searchsorted(t, self.timestamps)
        index = np.clip(index-1, 0, np.inf).astype("int64")

        if order == 0:
            v_output = v[index]
        else:
            index1 = np.clip(index+1, 0, len(v)-1)
            v0 = v[index]
            v1 = v[index1]
            eps = 1e-6
            r = (self.timestamps - t[index]) / (t[index1] - t[index] + eps)
            r[index1==index] = 0
            v_output = v0 * (1-r[:,None]) + v1 * r[:,None]

        return v_output



def event_sample_nd(params, threshold, time_window_s, sampling_randomness, precision=10000, return_gt=False):
    def sin(t):
        t = t[:, None]
        w, A, phi, gamma = params.T
        return A[None] * np.sin(w[None] * (t / time_window_s) ** gamma[None] * time_window_s + phi[None])

    t_gt = np.linspace(0, time_window_s, precision, endpoint=True)
    v_gt = sin(t_gt)

    ref = sin(np.array([0.0]))[0]

    idx = ev_cpp.generate_events(v_gt, threshold, ref)

    idx = np.concatenate([np.array([0]), idx])
    timestamps = idx / (precision - 1) * time_window_s

    values = sin(timestamps)
    displacement = sin(np.array([time_window_s]))[0] - sin(np.array([0.0]))[0]

    if return_gt:
        t = np.linspace(0, time_window_s, 1000)
        v = sin(t)
        gt = (t, v)

        return timestamps, values, displacement, gt
    else:
        return timestamps, values, displacement


def path_sample(params, threshold, time_window_s, precision=10000, return_gt=False):
    return path_sample_nd(params, threshold, time_window_s, precision=precision, return_gt=return_gt)

def path_sample_nd(params, threshold, time_window_s, precision=10000, return_gt=False):
    def sin(t):
        t = t[:, None]
        w, A, phi, gamma = params.T
        return A[None] * np.sin(w[None] * (t / time_window_s) ** gamma[None] * time_window_s + phi[None])

    timestamps = np.linspace(0, time_window_s, precision, endpoint=True)
    values = sin(timestamps)

    norms = np.linalg.norm(np.diff(values, axis=0), axis=-1)
    norms = np.concatenate((np.zeros(1), norms))
    s = np.cumsum(norms)[:,None]

    idx = ev_cpp.generate_events(s, threshold, np.array([0]))
    idx = np.concatenate([np.array([0]), idx])

    timestamps = idx / (precision - 1) * time_window_s

    values = sin(timestamps)
    displacement = sin(np.array([time_window_s]))[0] - sin(np.array([0.0]))[0]

    if return_gt:
        t = np.linspace(0, time_window_s, 1000)
        v = sin(t)
        gt = (t, v)

        return timestamps, values, displacement, gt
    else:
        return timestamps, values, displacement


def path_sample_1d(params, threshold, time_window_s):
    # sample from \int_0^x |d/dt (A sin (wt + phi)| dt
    # this integral becomes A \int_phi^{wx+phi} |cos(u)|du
    # we have the integral equal to A(G((wt+phi)/pi - 1/2) - G(phi/pi - 1/2))
    w, A, phi, gamma = params
    init_reference = np.random.rand() * threshold

    # signal to sample is r_0 + A G((wt+phi)/pi - 1/2), where G = 2+2floor(u)- (-1)^\floor(u) cos(pi u)
    # solve r_0 + A G(u) = n th
    u1 = (w * time_window_s + phi)/np.pi - 1/2
    u0 = phi/np.pi - 1/2

    n_max = int((init_reference + A * (G(u1) - G(u0)))/threshold)
    n = np.arange(1,n_max+1)

    # invert equation above G(u) = (n th - r_0)/A = g
    # we see that floor((G(u)-1)/2) = floor(u)
    g = (n * threshold - init_reference) / A + G(u0)
    floor_u = np.floor((g - 1)/2)

    # resub u = floor(u) + r, then cos(pi r ) = 2+2floor(u)-g
    cos_pi_u = 2 + 2 * floor_u - g
    u = np.arccos(np.clip(cos_pi_u, -1, 1)) / np.pi
    u += floor_u

    # u = (wt + phi)/pi - 1/2 -> t = ((u+1/2) pi - phi)/w
    t = ((u + .5)*np.pi - phi)/w
    t = t[t.argsort()]

    def sin(t):
        return A * np.sin(w * (t / time_window_s) ** gamma * time_window_s + phi)

    values  = sin(t)
    displacement = sin(time_window_s) - sin(0)

    return t, values, displacement

def canonical_2d(t_gt, v_gt):
    pass


def compute_inv(t, f_t, s):
    # find t_q such that f(t_q) = s
    idx = np.searchsorted(f_t, s)
    idx = np.clip(idx - 1, a_min=0, a_max=None)

    r = (s - f_t[idx]) / (f_t[idx+1] - f_t[idx])
    t_q = t[idx] + r * (t[idx+1] - t[idx])

    return t_q

def canonical(t, params, time_window_s):
    def sin(t):
        t = t[:, None]
        w, A, phi, gamma = params.T
        return A[None] * np.sin(w[None] * (t / time_window_s) ** gamma[None] * time_window_s + phi[None])

    values = sin(t)
    dvalues_norm = np.linalg.norm(np.diff(values, axis=0), axis=1)
    f_t = np.concatenate([np.array([0]), dvalues_norm.cumsum()])
    s = np.linspace(0, f_t[-1], len(t), endpoint=True)
    t_q = compute_inv(t, f_t, s)

    values_canonical = sin(t_q)

    if True:
        fig, ax = plt.subplots(nrows=3)
        ax[0].plot(t[1:], dvalues_norm, label="Original Signal")
        ax[0].plot(t[1:], np.linalg.norm(np.diff(values_canonical, axis=0), axis=1), label="Canonicalized Signal")
        ax[0].set_ylabel("Velocity")
        ax[0].legend()

        ax[1].set_xlabel("Time [s]")
        ax[1].plot(t, values[:,0], label="x coordinate")
        ax[1].plot(t, values[:,1], label="y coordinate")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Coordinates")
        ax[1].legend()

        ax[2].plot(s, values_canonical[:,0], label="x coordinate")
        ax[2].plot(s, values_canonical[:,1], label="y coordinate")
        ax[2].set_xlabel("Time [s]")
        ax[2].set_ylabel("Coordinates")
        ax[2].legend()

        fig.savefig("canonicalization.png", bbox_inches='tight')
        plt.show()

    return values_canonical, s

def event_sample(params, threshold, time_window_s, sampling_randomness=None, precision=10000, return_gt=False):
    # find all integers such that A sin(w t + phi) = A sin(phi) + n theta, for integers n
    # considering all values of sin(wt + phi) we have that n must be all integers such that
    # (- A - A sin(phi)) / theta < n < (A - A sin(phi))  / theta
    return event_sample_nd(params, threshold, time_window_s, sampling_randomness, precision=precision, return_gt=return_gt)


if __name__ == '__main__':
    threshold = 0.01
    time_window_s = 1
    params = np.array([[2*np.pi, 1, 0, 1], [4*np.pi, 3, np.pi/3, 1]]) # w, A, phi, gamma
    timestamps, values, displacement, (timestamps_gt, values_gt) = level_sample(params,
                                                                                threshold=threshold,
                                                                                time_window_s=time_window_s, return_gt=True)

    timestamps, values, displacement, (timestamps_gt, values_gt) = event_sample(params,
                                                                                threshold=threshold,
                                                                                time_window_s=time_window_s, return_gt=True)

    timestamps, values, displacement, (timestamps_gt, values_gt) = path_sample(params[:1],
                                                                               threshold=threshold,
                                                                               time_window_s=time_window_s, return_gt=True)


    # in the arclength case, the signal is sampled at rate 1/th, meaning that the bandlimit it 1/(2th)

    #model = ReconstructionModel(bandlimit_hz=bandlimit_hz, resolution=1000, time_window_s=1, threshold=threshold)

    timestamps_gt_can = np.linspace(0, time_window_s, 1000)
    values_gt_can, timestamps_gt_can = canonical(timestamps_gt_can, params, time_window_s)

    thresholds = np.linspace(0.01, 0.5, 1000)
    rmses_diff = []
    rmses_path = []

    num_samples_diff = []
    num_samples_path = []


    for j, threshold in enumerate(tqdm.tqdm(thresholds)):
        bandlimit_hz = 1 / (2 * threshold)

        # recons with path canon
        _, values_path, displacement, (timestamps_gt, values_gt) = path_sample(params,
                                                                          threshold=threshold,
                                                                          time_window_s=1, return_gt=True)
        timestamps_path = np.arange(len(values_path)) * threshold

        num_samples_path.append(len(timestamps_path))

        plot = False #j == 500#threshold>0.28
        recons_path = ReconstructionModel.dirichlet(timestamps_path, values_path, timestamps_gt_can, values_gt_can, bandlimit_hz,
                                                       plot=plot)

        # recons with geod
        # recons with path canon
        _, values_events, displacement, (timestamps_gt, values_gt) = event_sample(params,
                                                                          threshold=threshold,
                                                                          time_window_s=time_window_s, return_gt=True)
        timestamps_events = np.arange(len(values_events)) * threshold
        num_samples_diff.append(len(timestamps_events))

        plot = False#j==500#threshold>0.28
        recons_diff = ReconstructionModel.dirichlet(timestamps_events, values_events, timestamps_gt_can, values_gt_can, bandlimit_hz, plot=plot)

        if j == 500:
            fig, ax = plt.subplots(nrows=2)
            ax[0].scatter(timestamps_path, values_path[:,0], label="Samples Path")
            ax[0].scatter(timestamps_events, values_events[:,0], label="Samples Events")
            ax[0].plot(timestamps_gt_can, recons_path[:,0], label="Path events")
            ax[0].plot(timestamps_gt_can, recons_diff[:,0], label="Regular events")
            ax[0].plot(timestamps_gt_can, values_gt_can[:,0], label="Ground truth")
            ax[0].set_xlabel("Time [s]")
            ax[0].set_ylabel("X coordinate")

            ax[1].plot(timestamps_gt_can, recons_path[:,1], label="Path events")
            ax[1].plot(timestamps_gt_can, recons_diff[:,1], label="Regular events")
            ax[1].plot(timestamps_gt_can, values_gt_can[:,1], label="Ground truth")
            ax[1].scatter(timestamps_path, values_path[:,1], label="Samples Path")
            ax[1].scatter(timestamps_events, values_events[:,1], label="Samples Events")
            ax[1].set_xlabel("Time [s]")
            ax[1].set_ylabel("Y coordinate")

            ax[0].legend()
            ax[1].legend()
            fig.savefig("reconstructions.png", bbox_inches="tight")


        rmse_path = np.abs((recons_path - values_gt_can)**2).mean()
        rmse_diff = np.abs((recons_diff - values_gt_can)**2).mean()
        rmses_path.append(rmse_path)
        rmses_diff.append(rmse_diff)

    fig, ax = plt.subplots()
    ax.plot(thresholds, rmses_path, label="path events")
    ax.plot(thresholds, rmses_diff, label="regular events")
    ax.set_xlabel("threshold")
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    ax.legend()

    fig.savefig("reconstruction_vs_threshold.png", bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.scatter(num_samples_path, rmses_path, label="path events")
    ax.scatter(num_samples_diff, rmses_diff, label="regular events")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Num Samples")
    ax.legend()
    fig.savefig("reconstruction_vs_samples.png", bbox_inches='tight')

    plt.show()

