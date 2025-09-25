import numpy as np
import ev_cpp
import matplotlib.pyplot as plt


def sin(t, params, time_window_s):
    t = t[:, None]
    w, A, phi, gamma = params.T
    return A[None] * np.sin(w[None] * (t / time_window_s) ** gamma[None] * time_window_s + phi[None])

def event_sample_nd(params, threshold, time_window_s, precision=10000, ref_offset=0):

    t_gt = np.linspace(0, time_window_s, precision, endpoint=True)
    v_gt = sin(t_gt, params, time_window_s)

    ref = sin(np.array([0.0]), params, time_window_s)[0] + ref_offset

    idx = ev_cpp.generate_events(v_gt, threshold, ref)

    idx = np.concatenate([np.array([0]), idx])
    timestamps = idx / (precision - 1) * time_window_s

    values = sin(timestamps, params, time_window_s)

    return timestamps

if __name__ == '__main__':
    t_max = 2*np.pi
    threshold = 0.5

    params = np.array([[2*np.pi, 2, 0, 1], [8*np.pi, 3, np.pi/3, 1]]) # w, A, phi, gamma

    t = np.linspace(0, t_max, 1000)
    fx = sin(t, params, t_max)
    theta = np.linalg.norm(fx[None] - fx[:,None], axis=-1)

    X, Y = np.meshgrid(t, t)

    fig, ax = plt.subplots()
    ax.contour(X, Y, theta, [threshold])  # Plot the contour where F(x, y) = 0

    t_ev = event_sample_nd(params, threshold, t_max, precision=10000, ref_offset=0)

    for t0, t1 in zip(t_ev, t_ev[1:]):
        ax.plot([t0, t0], [t0, t1], color="b")
        ax.plot([t0, t1], [t1, t1], color="b")

    ax.plot(t, t, color="r")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.show()


