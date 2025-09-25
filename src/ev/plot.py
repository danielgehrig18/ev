import matplotlib.pyplot as plt
import numpy as np

def plot_positions_and_frames(positions, frames, color, fig=None, ax=None, scale=1):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Trajectory',
            linewidth=2, color=color)

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

def plot_velocities(t, vel, title="", ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(ncols=2)

    for c, l, v in zip("rgb", "xyz", vel[:,:3].T):
        ax[0].plot(t, v, label=f"angular {title} {l} coord", color=c)
    ax[0].legend()
    for c, l, v in zip("rgb", "xyz", vel[:,3:].T):
        ax[1].plot(t, v, label=f"linear {title} {l} coord", color=c)
    ax[1].legend()
    return fig, ax
