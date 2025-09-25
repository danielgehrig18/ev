import matplotlib.pyplot as plt
import numpy as np
import ev_cpp
import scipy.special

def g(t, A, w, phi):
    wtphi = w * t  + phi
    return A * (G(wtphi / np.pi - 1/2) - G(phi/np.pi - 1/2))

def g_inv(s, A, w, phi):
    return 1/w * (np.pi * G_inv(s/A + G(phi/np.pi - 1/2)) + np.pi/2 - phi)


def G(u):
    return 2 + 2 * np.floor(u) - (-1)**np.floor(u) * np.cos(np.pi * u)

def G_inv(g):
    floor_u = np.floor((g-1)/2)
    return floor_u + np.acos(2 + 2 * floor_u - g) / np.pi


def path_sample(params, threshold, time_window_s):
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


def regular_sample(params, time_window_s, sample_rate_hz, sampling_randomness=None):
    w, A, phi, gamma = params.T

    def sin(t):
        return A[None] * np.sin(w[None] * (t[:,None] / time_window_s) ** gamma[None] * time_window_s + phi[None])

    delta_t_s = 1 / sample_rate_hz

    sampling_randomness = np.random.rand() if sampling_randomness is None else sampling_randomness[0]
    start_time = sampling_randomness * delta_t_s
    n_samples = int(np.floor((time_window_s - start_time) / delta_t_s))
    timestamps = np.arange(n_samples + 1) * delta_t_s + start_time

    values = sin(timestamps)
    displacement = np.diff(sin(np.array([time_window_s, 0])), axis=0)[0]

    return timestamps, values, displacement

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

def level_sample(params, threshold, time_window_s, return_axis=False, sampling_randomness=None):
    timestamps = [level_sample_timestamps(p, threshold, time_window_s) for p in params]
    axis = [np.full_like(t, fill_value=i) for i, t in enumerate(timestamps)]

    timestamps = np.concatenate(timestamps)

    if return_axis:
        axis = np.concatenate(axis)
        axis = axis[timestamps.argsort()]

    timestamps = timestamps[timestamps.argsort()]

    def sin(t):
        t = t[:,None]
        w, A, phi, gamma = params.T
        return A[None] * np.sin(w[None] * (t/time_window_s)**gamma[None] * time_window_s + phi[None])

    values = sin(timestamps)
    displacement = np.diff(sin(np.array([time_window_s, 0])), axis=0)[0]

    if return_axis:
        return timestamps, values, displacement, axis
    else:
        return timestamps, values, displacement

def event_sample_1d(params, threshold, time_window_s, sampling_randomness=None):
    w, A, phi, gamma = params.T

    n_min = int(np.ceil((- A - A * np.sin(phi))/threshold))
    n_max = int(np.floor(( A - A * np.sin(phi))/threshold))
    n = np.arange(n_min, n_max + 1)

    # find the values for wt + phi = arcsin ( sin(phi) + n theta / A )
    wt_phi = np.arcsin(np.clip(n * threshold / A + np.sin(phi), -1, 1))
    wt_phi = np.concatenate([wt_phi, np.pi - wt_phi])
    wt = wt_phi - phi
    wt = np.sort(wt)

    # extend wt by so many 2*pi until the interval is filled.
    wt_min = wt[0]
    wt_max = wt[-1]

    wt_min_des = 0
    wt_max_des = w * time_window_s

    n_extend_right = int(np.ceil((wt_max_des - wt_max) / (2 * np.pi)))
    n_extend_left = int(np.ceil((wt_min - wt_min_des) / (2 * np.pi)))

    wt_ = wt.copy()
    if n_extend_right > 0:
        wt = np.concatenate([wt] + [wt_ + i * 2 * np.pi for i in range(1,n_extend_right+1)])
    if n_extend_left > 0:
        wt = np.concatenate([wt_ - i * 2 * np.pi for i in range(1,n_extend_left+1)] + [wt])

    wt[np.abs(wt) < 1e-6] = 0
    wt = wt[(wt >= wt_min_des) & (wt <= wt_max_des)]
    wt = np.sort(wt)

    timestamps = (wt / w) ** (1/gamma)

    def sin(t):
        return A * np.sin(w * (t/time_window_s)**gamma * time_window_s + phi)

    #timestamps = np.linspace(0, time_window_s, len(timestamps), endpoint=True)
    values = sin(timestamps)
    displacement = sin(time_window_s) - sin(0)

    return timestamps, values, displacement


def random_sample_ball(rand):
    X, Y = rand[:-1], rand[-1]
    erf_X = scipy.special.erfinv(2 * X - 1)
    exp_Y = - np.log(1 - Y)
    r = erf_X / np.sqrt(exp_Y + np.linalg.norm(erf_X)**2)
    return r


def event_sample_nd(params, threshold, time_window_s, sampling_randomness, precision=10000):
    def sin(t):
        t = t[:, None]
        w, A, phi, gamma = params.T
        return A[None] * np.sin(w[None] * (t / time_window_s) ** gamma[None] * time_window_s + phi[None])

    t_gt = np.linspace(0, time_window_s, precision, endpoint=True)
    v_gt = sin(t_gt)

    ref = sin(np.array([0.0]))[0]
    if sampling_randomness is not None:
        ref += random_sample_ball(sampling_randomness) * threshold

    idx = ev_cpp.generate_events(v_gt, threshold, ref)

    timestamps = idx / (precision - 1) * time_window_s

    values = sin(timestamps)
    displacement = sin(np.array([time_window_s]))[0] - sin(np.array([0.0]))[0]

    return timestamps, values, displacement


def event_sample(params, threshold, time_window_s, sampling_randomness=None, precision=10000):
    # find all integers such that A sin(w t + phi) = A sin(phi) + n theta, for integers n
    # considering all values of sin(wt + phi) we have that n must be all integers such that
    # (- A - A sin(phi)) / theta < n < (A - A sin(phi))  / theta
    if len(params) == 1:
        return event_sample_1d(params, threshold, time_window_s, sampling_randomness)
    else:
        return event_sample_nd(params, threshold, time_window_s, sampling_randomness, precision=precision)


def plot_events(ax, timestamps, values):
    for t0, t1, v0, v1 in zip(timestamps, timestamps[1:], values, values[1:]):
        ax.plot([t0, t1], [v0, v0], color="r")
        ax.plot([t1, t1], [v0, v1], color="b")
        ax.scatter([t0], [v0])
        ax.scatter([t1], [v1])

if __name__ == '__main__':
    import tqdm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    xs = []
    for i in tqdm.tqdm(range(10000000)):
        r = np.random.rand(3)
        x = random_sample_ball(r)
        xs.append(x)

    xs = np.stack(xs)

    bins, xedges, yedges = np.histogram2d(xs[:, 0], xs[:, 1], bins=100, range=[[-1.1,1.1],[-1.1,1.1]])
    XX, YY = np.meshgrid(xedges[1:], yedges[1:])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(XX, YY, bins)
    plt.show()
