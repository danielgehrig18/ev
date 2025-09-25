import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from ev.lie_group_utils import exp_se3, exp_so3, log_se3, Gamma1_Gamma2_gal, log_so3
from ev.spline import random_spline


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
    samples = spline.sample(t, n=1)

    df = samples["d1f"]
    dt = t[1] - t[0]
    norm = np.linalg.norm(df, axis=-1) * dt
    s = np.cumsum(np.concatenate([np.array([0]), norm]))[...,:-1]

    # find triggers
    M_max = params["num_tokens"] if "num_tokens" in params else int(np.ceil(s[-1]/threshold))
    thresholds = np.arange(M_max) * threshold

    idx = np.searchsorted(s, thresholds) - 1
    idx = np.clip(idx, 0, len(s)-2)

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


def sample_enforce_num_tokens(func, spline, num_tokens, **params):
    timestamps = func(spline, **params)

    # enough randomization until more token than necessary
    while len(timestamps) < num_tokens:
        spline = random_spline(params["n_poses"], params["pose_range"], params["velocity_range"],
                               params["acceleration_range"], params["time_range"])
        timestamps = func(spline, **params)

    # rescale phi, such that num events
    rescale = (params["time_range"][1] / timestamps[num_tokens-1])
    spline.phi = PchipInterpolator(spline.phi.times * rescale , spline.phi.values)
    params["num_tokens"] = num_tokens
    timestamps = func(spline, **params)

    return timestamps, spline

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
