import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def generate_events(y, t, threshold=0.05):
    ref = y[0]
    events = []
    for y_0, y_1, t_0, t_1 in zip(y, y[1:], t, t[1:]):
        while np.abs(ref - y_1) > threshold:
            p = np.sign(y_1 - ref)
            ref += p * threshold
            t_int = t_0 + (t_1 - t_0) * (ref - y_0) / (y_1 - y_0)
            events.append([t_int, p, ref])
    return np.array(events)

def plot_events(ax, timestamps, values):
    for t0, t1, v0, v1 in zip(timestamps, timestamps[1:], values, values[1:]):
        ax.plot([t0, t1], [v0, v0], color="r")
        ax.plot([t1, t1], [v0, v1], color="b")

def remap_events(events, t, phi):
    e_t, e_p, e_ref = events.T
    level_crossings = generate_level_crossings(t, phi, e_t)
    c_t, index, c_p = level_crossings.T
    index = index.astype("int32")
    e_p_new = c_p * e_p[index]
    e_ref_new = e_ref[index]
    e_t_new = c_t

    events_new = np.stack([e_t_new, e_p_new, e_ref_new], axis=1)

    return events_new

def generate_level_crossings(t, phi, levels):
    # find the level below current value

    levels = np.concatenate([[phi[0]], levels])
    level_idx = 0
    level_0 = levels[level_idx]
    level_1 = levels[level_idx+1]

    crossings = []
    for t_0, t_1, phi_0, phi_1 in zip(t, t[1:], phi, phi[1:]):
        # check if level is between phi0, phi1, then just add that
        level_0_between_phi_1_phi_0 = (phi_1 - level_0) * (level_0 - phi_0) > 0
        level_1_between_phi_1_phi_0 = (phi_1 - level_1) * (level_1 - phi_0) > 0

        if level_0_between_phi_1_phi_0 and not level_1_between_phi_1_phi_0:

            t_corr = t_0 + (t_1 - t_0) * (level_0 - phi_0) / (phi_1 - phi_0)
            idx = level_idx

            crossings.append([t_corr, idx, -1])

            level_idx -= 1

            level_0 = levels[level_idx]
            level_1 = levels[level_idx + 1]

        elif not level_0_between_phi_1_phi_0 and level_1_between_phi_1_phi_0:

            t_corr = t_0 + (t_1 - t_0) * (level_1 - phi_0) / (phi_1 - phi_0)
            idx = level_idx+1

            crossings.append([t_corr, idx, 1])

            level_idx += 1

            level_0 = levels[level_idx]
            level_1 = levels[level_idx + 1]

    crossings =  np.array(crossings)
    crossings = crossings[1:]
    crossings[:,1] -= 1

    return crossings

def signal_function(y):
    return y**2

def remapping_function2(t):
    return t**0.2

def remapping_function(t):
    return 0.9*(1-4*(t-.5)**2)

def remap_signal(signal, t, phi):
    linear_interp = interp1d(t, signal, kind='linear')
    return linear_interp(phi)


if __name__ == '__main__':
    N = 1000000
    timestamps = np.linspace(0, 1, N)
    signal = signal_function(timestamps)
    signal_remapped = remap_signal(signal, timestamps, remapping_function(timestamps))
    events = generate_events(signal, timestamps)
    events_remapped = generate_events(signal_remapped, timestamps)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

    # signal
    ax[0,0].plot(timestamps, signal, color="g")
    e_t, e_p, e_ref = events.T
    plot_events(ax[0,0], e_t, e_ref)
    ax[0,0].set_ylabel("Signal")
    ax[0,0].set_xlabel("Time")
    ax[0,0].set_title("Events from Signal")

    # signal 2
    ax[1,0].plot(timestamps, signal_remapped, color="g")
    e_t, e_p, e_ref = events_remapped.T
    plot_events(ax[1,0], e_t, e_ref)
    ax[1,0].set_ylabel("Signal")
    ax[1,0].set_xlabel("Time")
    ax[1,0].set_title("Events from Remapped Signal")

    # plot remapping function
    ax[0,1].plot(timestamps, remapping_function(timestamps), color="g")
    ax[0,1].set_ylabel("Remapped Time")
    ax[0,1].set_xlabel("Time")
    ax[0,1].set_title("Time Remapping")

    # signal 2
    ax[1,1].plot(timestamps, signal_remapped, color="g")
    events_remapped = remap_events(events, timestamps, remapping_function(timestamps))
    e_t, e_p, e_ref = events_remapped.T
    plot_events(ax[1,1], e_t, e_ref)
    ax[1,1].set_ylabel("Signal")
    ax[1,1].set_xlabel("Time")
    ax[1,1].set_title("Remapped Events")

    plt.show()