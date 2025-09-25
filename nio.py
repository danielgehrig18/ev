import numpy as np

import matplotlib.pyplot as plt
import pypose as pp
from src.ev.lie_group_utils import Ad_vee_se3, ad_vee_se3, exp_se3, exp_so3, log_se3, Gamma1_Gamma2_gal, log_so3
from scipy.interpolate import PchipInterpolator

def matrix_product(matrices, x):
    for m in matrices:
        x = m @ x
    return x


def derivative_matrix(A_i, d1_a_p_i, d2_a_p_i, d3_a_p_i):
    Ad_A_i = Ad_vee_se3(A_i)
    ad_d1_a_p1 = ad_vee_se3(d1_a_p_i)
    ad_d2_a_p1 = ad_vee_se3(d2_a_p_i)

    matrices = np.zeros(shape=A_i.shape[:2] + (19, 19))
    matrices[...,-1,-1] = 1

    matrices[...,0:6,  0:6] = Ad_A_i
    matrices[...,0:6, 6:12] = 2 * ad_d1_a_p1 @  Ad_A_i
    matrices[...,0:6,12:18] = (ad_d2_a_p1 +  ad_d1_a_p1 @ ad_d1_a_p1) @ Ad_A_i
    matrices[...,0:6,   18] = d3_a_p_i

    matrices[...,6:12, 6:12] = Ad_A_i
    matrices[...,6:12,12:18] = ad_d1_a_p1 @  Ad_A_i
    matrices[...,6:12,   18] = d2_a_p_i

    matrices[...,12:18,12:18] = Ad_A_i
    matrices[...,12:18,   18] = d1_a_p_i

    dS_0 = np.zeros(shape=(A_i.shape[1], 19)) # n_samples x 13
    dS_0[...,-1] = 1

    dS = matrix_product(matrices, dS_0[...,None])[...,0]

    dS_3 = dS[...,0:6]
    dS_2 = dS[...,6:12]
    dS_1 = dS[...,12:18]

    return dS_1, dS_2, dS_3


class SE3Spline:

    def __init__(self, poses, velocities, accelerations, phi):
        # implements Fritsch Carlson monotone spline https://epubs.siam.org/doi/epdf/10.1137/0717021
        self.phi = phi
        self.t0 = phi.x[ 0]
        self.t1 = phi.x[-1]

        # implements this: https://ethaneade.com/lie_spline.pdf
        self.num_poses = len(poses)
        B0 = poses[:-1]
        B1 = poses[1:]

        d0 = velocities[:-1]
        d1 = velocities[1:]

        h0 = accelerations[:-1]
        h1 = accelerations[1:]

        self.timestamps = np.arange(len(poses))

        self.B0 = B0
        self.p_i = np.stack([log_se3(B1 @ np.linalg.inv(B0)), d0, d1, h0, h1]) # 5 x 6

        self.C_0 = np.array([[0, 0,   0,   10, -15,    6],
                             [0, 1,   0,   -6,   8,   -3],
                             [0, 0,   0,   -4,   7,   -3],
                             [0, 0, 0.5, -1.5, 1.5, -0.5],
                             [0, 0,   0,  0.5,  -1,  0.5]])

        D = np.diag(np.arange(5)+1,k=-1)
        self.C_1 = self.C_0 @ D
        self.C_2 = self.C_1 @ D
        self.C_3 = self.C_2 @ D
        self.n = np.arange(6)

    def sample(self, t, eps=1e-9):
        # map the timestamp through the spline
        phi = self.phi(t) * (self.num_poses - 1)
        phi = np.clip(phi, 0, self.num_poses - 1 - eps)

        # figure out which spline is meant where, and what is the offset
        idx = phi.astype("int32")
        t_norm = phi - idx
        S, dS_1, dS_2, dS_3 = self._scatter_sample(idx, t_norm)

        # change derivatives
        d1phi = self.phi(t, nu=1) * (self.num_poses - 1)
        d2phi = self.phi(t, nu=2) * (self.num_poses - 1)
        d3phi = self.phi(t, nu=3) * (self.num_poses - 1)

        dS_3 = d1phi[:,None]**3 * dS_3 + 3 * d2phi[:,None] * d1phi[:,None] * dS_2 + d3phi[:,None] * dS_1
        dS_2 = d1phi[:,None]**2 * dS_2 + d2phi[:,None] * dS_1
        dS_1 = d1phi[:,None] * dS_1

        return {"f": S, "d1f": dS_1, "d2f": dS_2, "d3f": dS_3}

    def _scatter_sample(self, idx, t_norm):

        t_n = t_norm[None,:] ** self.n[:,None]
        a_i = self.C_0 @ t_n # n_samples x 6
        d1_a_i = self.C_1 @ t_n
        d2_a_i = self.C_2 @ t_n
        d3_a_i = self.C_3 @ t_n

        # compute products, taking scattering into accound
        p_i = self.p_i[:, idx] # 5 x num_samples x 6
        B0 = self.B0[idx]

        a_p_i = p_i * a_i[...,None]
        d1_a_p_i = p_i * d1_a_i[...,None]
        d2_a_p_i = p_i * d2_a_i[...,None]
        d3_a_p_i = p_i * d3_a_i[...,None]

        # compute all things
        A_i = exp_se3(a_p_i) # n_samples x 6 x 6
        dS_1, dS_2, dS_3 = derivative_matrix(A_i, d1_a_p_i, d2_a_p_i, d3_a_p_i)
        S = matrix_product(A_i, B0)

        return S, dS_1, dS_2, dS_3

def plot_frames(ax, positions, frames, color):
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Trajectory',
            linewidth=2, color=color)

    scale = 1
    for i in range(len(positions)):
        # Plot orientation frames at select intervals
        origin = positions[i]
        frame = frames[i]

        # X axis (Red)
        ax.quiver(*origin, *(frame[:, 0] * scale), color=color, linewidth=1)
        # Y axis (Green)
        ax.quiver(*origin, *(frame[:, 1] * scale), color=color, linewidth=1)
        # Z axis (Blue)
        ax.quiver(*origin, *(frame[:, 2] * scale), color=color, linewidth=1)

    # Labels and plot settings
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('6 DoF Trajectory with Orientation Frames')
    ax.legend()
    ax.grid(True)
    v_intervals = np.vstack((ax.xaxis.get_view_interval(),
                             ax.yaxis.get_view_interval(),
                             ax.zaxis.get_view_interval()))
    deltas = np.ptp(v_intervals, axis=1)
    ax.set_box_aspect(deltas)

    return fig, ax

def plot(positions, orientations):
    """
    Plot the 3D trajectory with orientation frames.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Interpolated Trajectory',
            linewidth=2)

    # Plot orientation frames at select intervals
    frame_interval = len(positions) // 20  # Plot 20 frames evenly spaced
    scale = 0.5  # Scale for orientation axes
    for i in range(0, len(positions), frame_interval):
        origin = positions[i]
        R = orientations[i]

        # X axis (Red)
        ax.quiver(*origin, *(R[:, 0] * scale), color='r', linewidth=1)
        # Y axis (Green)
        ax.quiver(*origin, *(R[:, 1] * scale), color='g', linewidth=1)
        # Z axis (Blue)
        ax.quiver(*origin, *(R[:, 2] * scale), color='b', linewidth=1)

    # Labels and plot settings
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('6 DoF Trajectory with Orientation Frames')
    ax.legend()
    ax.grid(True)
    v_intervals = np.vstack((ax.xaxis.get_view_interval(),
                             ax.yaxis.get_view_interval(),
                             ax.zaxis.get_view_interval()))
    deltas = np.ptp(v_intervals, axis=1)
    ax.set_box_aspect(deltas)

    return fig, ax

def plot_velocity(t, vel, title=""):
    fig, ax = plt.subplots(nrows=2)
    for c, l, v in zip("rgb", "xyz", vel[:,:3].T):
        ax[0].plot(t, v, label=f"angular {title} {l} coord", color=c)
    ax[0].legend()
    for c, l, v in zip("rgb", "xyz", vel[:,3:].T):
        ax[1].plot(t, v, label=f"linear {title} {l} coord", color=c)
    ax[1].legend()
    return fig, ax


def generate_random_phi(time_range, n=5):
    t0, t1 = time_range
    #phi_values = np.linspace(0, 1, num=n, endpoint=True)
    values = np.concatenate([np.array([0]), np.sort(np.random.rand(n-2)), np.array([1])])
    times = np.linspace(t0, t1, num=n, endpoint=True)
    phi = PchipInterpolator(times, values)

    phi.times = times
    phi.values = values

    return phi

def event_based_sample(trajectory, threshold=0.1, num_timestamps=10000, sample_type="send_on_delta", debug=False):
    t_sample = np.linspace(trajectory.t0, trajectory.t1, num=num_timestamps, endpoint=True)
    samples = trajectory.sample(t_sample)

    if sample_type == "path":
        return path_sample(t_sample, samples, threshold, debug)
    elif sample_type == "send_on_delta":
        return send_on_delta_sample(t_sample, samples, threshold, debug)

def send_on_delta_sample(trajectory, threshold, debug=False, num_timestamps=10000, **params):
    t = np.linspace(trajectory.t0, trajectory.t1, num=num_timestamps, endpoint=True)
    samples = trajectory.sample(t)

    f = samples["f"]
    ref = f[0]

    solver = Solver(t[0], ref)

    for t0, t1, f0, f1 in zip(t, t[1:], f, f[1:]):
        solver.solve(threshold, t0, t1, f0, f1)

    t_events = np.array(solver.t_events)
    refs = np.stack(solver.refs)

    if debug:
        fig, ax = plt.subplots()
        for i in range(1, len(refs)):
            mask = (t >= t_events[i - 1]) & (t <= t_events[i])
            distance = solver.distance(refs[i-1], f[mask])
            ax.plot(t[mask], distance)
        plt.show()

    return t_events

class Solver:
    def __init__(self, t0, ref, type="left_se3_trans", alpha=0):
        self.t0 = t0
        self.ref = ref

        self.t_events = [t0]
        self.refs = [ref]
        self.type = type
        self.alpha = alpha

    def distance(self, ref, f):
        if self.type.startswith("left"):
            # Error: || log_se3(x_ref^-1 x(t)) ||
            df = np.einsum("ij,njk->nik", np.linalg.inv(ref), f)
        else:
            # Error: || log_se3(x(t) x_ref^-1) ||
            df = np.einsum("nij,jk->nik", f, np.linalg.inv(ref))

        if self.type.endswith("log"):
            r = log_se3(df)
            distance = self.alpha * np.linalg.norm(r[:,:3], axis=-1)**2 + np.linalg.norm(r[:,3:], axis=-1)**2
        else:
            distance = np.linalg.norm(df[:,:3,3], axis=-1)**2

        return distance

    def solve(self, threshold, t0, t1, f0, f1):
        precomp = self.precomp(self.ref, t0, t1, f0, f1, type=self.type)
        t = self.solve_t(self.type, threshold=threshold, **precomp)
        if t is not None:
            self.ref = self.interpolate(t, type=self.type, f0=f0, f1=f1)
            self.refs.append(self.ref)
            self.t_events.append(t0 + (t1 - t0) * t)

    def interpolate(self, t, type, f0, f1):
        if type.startswith("left"):
            # Interp: x(t) = x0 exp(t log(x0^-1 x1))
            return f0 @ exp_se3(t * log_se3(np.linalg.inv(f0) @ f1))
        elif type.startswith("right"):
            # Interp: x(t) = exp(t log(x1 x0^-1)) x0
            return exp_se3(t * log_se3(f1 @ np.linalg.inv(f0))) @ f0
        else:
            raise ValueError

    def solve_t(self, type, threshold, **precomp):
        R, p, r, v = precomp["R"], precomp["p"], precomp['r'], precomp['v']
        t = 0
        for _ in range(3):
            t = self.solve_t_single(t, type, threshold, R, p, r, v)
            if t > 1 or t < 0:
                return None
        else:
            return t

    def solve_t_single(self, t, type, threshold, R, p, r, v):
        E = exp_so3(r * t)
        J = Gamma1_Gamma2_gal(r * t)[0]

        if type == "left_se3_log":
            # a * |log(R exp(rt)|^2 + |J^-1(log(R exp(rt))) (R J(rt) vt + p)|^2 == theta^2
            # --> a * A^T A + |B t + C|^2 == theta^2
            # (a * A^T A - theta^2 + C^T C) + 2 B^T C t + B^T B t^2 = 0
            A = log_so3(R @ E)
            J_inv = np.linalg.inv(Gamma1_Gamma2_gal(A)[0])
            B = J_inv @ R @ J @ v
            C = J_inv @ p

        elif type == "right_se3_log":
            # a * |log(exp(rt) R)|^2 + |J^-1(log(exp(rt) R)) (exp(rt)p + J(rt) vt))|^2 == theta^2
            # a * A^T A + |C + B t|^2 = theta^2
            A = log_so3(E @ R)
            J_inv = np.linalg.inv(Gamma1_Gamma2_gal(A)[0])
            B = J_inv @ J @ v
            C = J_inv @ exp_so3(r * t) @ p

        elif type == "left_se3_trans":
            # |R J(rt) vt + p|^2 == theta^2
            # |Bt + C|^2 = theta^2
            B = R @ J @ v
            C = p
            A = np.zeros_like(B)

        elif type == "right_se3_trans":
            # ||exp(rt)p + J(rt) vt|| == theta^2
            # |Bt + C|^2 = theta^2
            B = J @ v
            C = E @ p
            A = np.zeros_like(B)
        else:
            raise ValueError

        # quadratic terms
        c = self.alpha * A.T @ A + C.T @ C - threshold**2
        b = B.T @ C
        a = B.T @ B
        b_a = b / a
        c_a = c / a

        t = - b_a + np.sqrt(b_a**2 - c_a)

        return t

    def precomp(self, ref, t0, t1, f0, f1, type):
        if type == "left_se3_log":
            # Error: || log_se3(x_ref^-1 x(t)) ||
            # Interp: x(t) = x0 exp(t log(x0^-1 x1))
            # Deriv: || log_se3(x_ref^-1 x(t)) ||
            #        =  ||log_se3((R exp(rt), R J(rt) vt + p))||
            #        =  ||(log(R exp(rt), J^-1(log(R exp(rt))) (R J(rt) vt + p))||
            #        = a * |log(R exp(rt)|^2 + |J^-1(log(R exp(rt))) (R J(rt) vt + p)|^2
            dT = np.linalg.inv(ref) @ f0
            R, p  = dT[:3,:3], dT[:3, 3]
            dt = log_se3(np.linalg.inv(f0) @ f1)
            r, v = dt[:3], dt[3:]
            return dict(r=r,v=v,R=R,p=p)
        elif type == "right_se3_log":
            # Error: || log_se3(x(t) x_ref^-1) ||
            # Interp: x(t) = exp(t log(x1 x0^-1)) x0
            # Deriv: || log_se3(x(t) x_ref^-1) ||
            #        =  ||log_se3((exp(rt) R, exp(rt)p + J(rt) vt))||
            #        =  ||(log(exp(rt) R) , J^-1(log(exp(rt) R)) (exp(rt)p + J(rt) vt))||
            #        = a * |log(exp(rt) R)|^2 + |J^-1(log(exp(rt) R)) (exp(rt)p + J(rt) vt))|^2
            dT = f0 @ np.linalg.inv(ref)
            R, p = dT[:3, :3], dT[:3, 3]
            dt = log_se3(f1 @ np.linalg.inv(f0))
            r, v = dt[:3], dt[3:]
            return dict(r=r, v=v, R=R, p=p)
        elif type == "left_se3_trans":
            # Error: || pos(x_ref^-1 x(t)) ||
            # Interp: x(t) = x0 exp(t log(x0^-1 x1))
            # Deriv: || pos(x_ref^-1 x(t)) ||
            #        =  ||pos((R exp(rt), R J(rt) vt + p))||
            #        =  ||R J(rt) vt + p||
            #        =  |R J(rt) vt + p|^2
            dT = np.linalg.inv(ref) @ f0
            R, p = dT[:3, :3], dT[:3, 3]
            dt = log_se3(np.linalg.inv(f0) @ f1)
            r, v = dt[:3], dt[3:]
            return dict(r=r, v=v, R=R, p=p)
        elif type == "right_se3_trans":
            # Error: || pos(x(t) x_ref^-1) ||
            # Interp: x(t) = exp(t log(x1 x0^-1)) x0
            # Deriv: || pos(x(t) x_ref^-1) ||
            #        =  ||pos((exp(rt) R, exp(rt)p + J(rt) vt))||
            #        =  ||exp(rt)p + J(rt) vt||
            dT = f0 @ np.linalg.inv(ref)
            R, p = dT[:3, :3], dT[:3, 3]
            dt = log_se3(f1 @ np.linalg.inv(f0))
            r, v = dt[:3], dt[3:]
            return dict(r=r, v=v, R=R, p=p)


def path_sample(spline, threshold, debug=False, num_timestamps=10000, **params):
    t = np.linspace(spline.t0, spline.t1, num=num_timestamps, endpoint=True)
    samples = spline.sample(t)

    df = samples["d1f"]
    dt = t[1] - t[0]
    norm = np.linalg.norm(df, axis=-1) * dt
    s = np.cumsum(np.concatenate([np.array([0]), norm]))[...,:-1]

    # find triggers
    M_max = int(np.ceil(s[-1]/threshold))
    thresholds = np.arange(M_max) * threshold

    idx = np.searchsorted(s, thresholds) - 1
    idx[idx<0] = 0

    s0 = s[idx]
    s1 = s[idx+1]

    t0 = t[idx]
    t_events = dt * (thresholds - s0) / (s1 - s0 + 1e-9) + t0

    if debug:
        fig, ax = plt.subplots()

        for th in thresholds:
            ax.plot([t[0], t[-1]], [th, th], color="g")

        ax.plot(t, s, marker="o")
        ax.scatter(t_events, thresholds, marker="*", s=1000, c="r")

    return t_events


class Sampler:
    def __init__(self, threshold, num_tokens, sampling_type, time_range):
        self.threshold = threshold
        self.num_tokens = num_tokens
        self.sampling_type = sampling_type
        self.time_range = time_range

    def sampling_timestamps(self, spline, **params):
        if self.sampling_type == "regular":
            t0, t1 = self.time_range
            t_sample = np.linspace(t0, t1, self.num_tokens, endpoint=True)
            return t_sample, spline
        elif self.sampling_type == "send_on_delta":
            return sample_enforce_num_tokens(func=send_on_delta_sample,
                                             spline=spline, num_tokens=self.num_tokens,
                                             threshold=self.threshold, **params)
        elif self.sampling_type == "path":
            return sample_enforce_num_tokens(func=path_sample,
                                             spline=spline, num_tokens=self.num_tokens,
                                             threshold=self.threshold, **params)

def random_spline(n_poses, pose_range, velocity_range, acceleration_range, time_range):
    # generate random poses, and first and second order
    poses = pp.randn_se3(n_poses, sigma=pose_range).matrix().numpy()
    velocities = np.random.randn(n_poses, 6) * velocity_range
    accelerations = np.random.randn(n_poses, 6) * acceleration_range

    # generate spline
    phi = generate_random_phi(time_range=time_range)
    spline = SE3Spline(poses, velocities, accelerations, phi=phi)

    return spline

def sample_enforce_num_tokens(func, spline, num_tokens, **params):
    timestamps = func(spline, **params)

    # enough randomization until more token than necessary
    while len(timestamps) < num_tokens:
        spline = random_spline(params["n_poses"], params["pose_range"], params["velocity_range"],
                               params["acceleration_range"], params["time_range"])
        timestamps = func(spline, **params)

    # rescale phi, such that num events
    rescale = params["time_range"][1] / timestamps[num_tokens-1]
    spline.phi = PchipInterpolator(spline.phi.times * rescale , spline.phi.values)
    timestamps = func(spline, **params)

    return timestamps, spline

def modify_phi(phi, time_range, new_time_range):
    return

def path_sample(spline, enforce_num_tokens=False):
    pass


class Dataset:
    def __init__(self, time_range, n_poses, num_tokens, num_samples,
                 pose_range, velocity_range, acceleration_range, threshold=0.8, split="train",
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

        self.data = None
        self.data = [self.__getitem__(i) for i in range(num_samples)]
        self.split = split

    def __getitem__(self, item):
        return self.generate_sample() if self.data is None else self.data[item]

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

        return output

def compute_frame(output):
    df1_r, df1_t = output["d1f"][:,:3], output["d1f"][:,3:]
    df2_r, df2_t = output["d2f"][:,:3], output["d2f"][:,3:]
    df3_r, df3_t = output["d3f"][:,:3], output["d3f"][:,3:]

    df1_r_o, df2_r_o, df3_r_o = orthogonalize(df1_r, df2_r, df3_r)
    df1_t_o, df2_t_o, df3_t_o = orthogonalize(df1_t, df2_t, df3_t)

    df_r_o = np.stack([df1_r_o, df2_r_o, df3_r_o], axis=1)
    df_t_o = np.stack([df1_t_o, df2_t_o, df3_t_o], axis=1)

    return df_r_o, df_t_o

def orthogonalize(df1, df2, df3, eps=1e-9):
    df1_o = df1 / (eps + np.linalg.norm(df1, axis=-1, keepdims=True))

    df2_o = df2 - (df2 * df1_o).sum(-1)[:,None] * df1_o
    df2_o = df2_o / (eps + np.linalg.norm(df2_o, axis=-1, keepdims=True))

    df3_o = df3 - (df3 * df1_o).sum(-1)[:,None] * df1_o - (df3 * df2_o).sum(-1)[:,None] * df2_o
    df3_o = df3_o / (eps + np.linalg.norm(df3_o, axis=-1, keepdims=True))

    return df1_o, df2_o, df3_o


if __name__ == '__main__':
    np.random.seed(4)

    n_poses = 5
    poses = pp.randn_se3(n_poses, sigma=2).matrix().numpy()
    velocities = np.random.randn(n_poses, 6) * 2
    accelerations = np.random.randn(n_poses, 6) * 0.2

    timestamps = np.array([0, 1, 2])
    #timestamps = np.array([0, 1])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for _ in range(5):
        phi = generate_random_phi(time_range=[0, 10])
        spline = SE3Spline(poses, velocities, accelerations, phi=phi)
        t = send_on_delta_sample(spline, threshold=0.8, debug=False)
        t_sample = np.linspace(0.01, 9.99, 10000)
        output = spline.sample(t)

        position = output['f'][...,:3,3]
        frame_r, frame_t = compute_frame(output)
        plot_frames(ax, position, frame_r, color=np.random.rand(3))

    plt.show()

    t_sample = np.linspace(0.01, 9.99, 10000)
    output = spline.sample(t_sample)

    pose_sample = output['f']
    vel_sample = output['d1f']
    acc_sample = output['d2f']
    jerk_sample = output['d3f']

    fig, ax = plot(pose_sample[:,:3,3], pose_sample[:,:3,:3])
    fig_vel, ax_vel = plot_velocity(t_sample, vel_sample, "velocity")
    fig_acc, ax_acc = plot_velocity(t_sample, acc_sample, "acceleration")
    fig_jer, ax_jer = plot_velocity(t_sample, jerk_sample, "jerk")

    plt.show()

    time_range = [0, 10]
    num_tokens = 100
    num_samples = 1000
    pose_range = 2
    velocity_range = 2
    acceleration_range = 2

    dataset = Dataset(time_range, n_poses, num_tokens, num_samples,
                 pose_range, velocity_range, acceleration_range, threshold=0.2, split="train",
                 sampling_type="send_on_delta")
