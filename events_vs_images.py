import matplotlib.pyplot as plt

import esim_torch
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
import skimage.transform
import torch
import evlicious

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm


def create_figure():
    """
    Create a figure with the following layout:

        +------+------+------+
        | img  | flt  | env  |
        +------+------+------+
        |       wave         |
        +--------------------+
        |    bottom plot     |
        +--------------------+

    Returns (fig, axes_dict) where axes_dict is a dictionary
    of named subplot axes.
    """
    fig = plt.figure(figsize=(10, 8))

    # GridSpec with 3 rows × 3 columns:
    # row 0: 3 subplots side-by-side
    # row 1: 1 wide subplot (spans all 3 columns)
    # row 2: 1 wide subplot (spans all 3 columns)
    gs = gridspec.GridSpec(
        nrows=3, ncols=3,
        height_ratios=[1, 1, 1],  # You can adjust to taste
        hspace=0.3, wspace=0.3
    )

    # Top row subplots
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.set_title("Image")
    ax_img.axis("off")

    ax_flt = fig.add_subplot(gs[0, 1])
    ax_flt.set_title("Image Filtered")
    ax_flt.axis("off")

    ax_env = fig.add_subplot(gs[0, 2])
    ax_env.set_title("Events")
    ax_env.axis("off")

    # Middle row: a wide subplot spanning all 3 columns
    ax_wave = fig.add_subplot(gs[1, 0:3])
    ax_wave.set_title("Rotation Angle")

    # Bottom row: another wide subplot spanning all 3 columns
    ax_bottom = fig.add_subplot(gs[2, 0:3])
    ax_bottom.set_title("Events 2D")

    # Collect axes in a dictionary for easy access
    axes = {
        'img': ax_img,
        'flt': ax_flt,
        'env': ax_env,
        'wave': ax_wave,
        'bottom': ax_bottom
    }
    return fig, axes


def init_plots(axes):
    """
    Initialize some example content in each subplot.
    Returns a dictionary of artists that can be updated.
    """
    # For the 'img' subplot, let's just show a random image
    img_data = np.random.rand(10, 10)
    im_handle = axes['img'].imshow(img_data, cmap='viridis')

    # For 'flt' subplot, also show a random image or any other placeholder
    flt_data = np.random.rand(10, 10)
    flt_handle = axes['flt'].imshow(flt_data, cmap='plasma')

    # For 'env', we could do a simple line
    x = np.linspace(0, 2 * np.pi, 200)
    line_env, = axes['env'].plot(x, 0.5 * np.sin(x), 'r-')

    # For 'wave', let's plot an initial sine wave
    line_wave, = axes['wave'].plot(x, np.sin(x), 'b-')
    axes['wave'].set_xlim(0, 2 * np.pi)
    axes['wave'].set_ylim(-1.1, 1.1)

    # Bottom plot might be another line or bar, etc.
    # Here we do a random line
    x_bottom = np.linspace(0, 1, 50)
    line_bottom, = axes['bottom'].plot(
        x_bottom, np.random.rand(50), 'g-'
    )
    axes['bottom'].set_ylim(0, 1)

    # Return handles so you can update them later
    return {
        'img_handle': im_handle,
        'flt_handle': flt_handle,
        'line_env': line_env,
        'line_wave': line_wave,
        'line_bottom': line_bottom
    }


def update_plots(handles, frame):
    """
    Update the plots for a new 'frame' (e.g. in an animation loop).
    """
    # Update the 'env' line
    x = np.linspace(0, 2 * np.pi, 200)
    y_env = 0.5 * np.sin(x + 0.2 * frame)
    handles['line_env'].set_ydata(y_env)

    # Update the 'wave'
    y_wave = np.sin(x + 0.5 * frame)
    handles['line_wave'].set_ydata(y_wave)

    # Update the bottom plot line with random data
    x_bottom = np.linspace(0, 1, 50)
    y_bottom = np.random.rand(50)
    handles['line_bottom'].set_ydata(y_bottom)

    # Optionally update the images (random or processed data)
    new_img = np.random.rand(10, 10)
    handles['img_handle'].set_data(new_img)
    new_flt = np.random.rand(10, 10)
    handles['flt_handle'].set_data(new_flt)


def main_demo():
    """
    Demonstrate a simple update loop. In practice,
    you might use FuncAnimation or a GUI event timer.
    """
    fig, axes = create_figure()
    handles = init_plots(axes)

    # Interactive mode so we can see updates
    plt.ion()
    plt.show()

    for frame in range(100):
        update_plots(handles, frame)
        plt.pause(0.05)

    plt.ioff()
    plt.show()


class NonIdealEsim:
    def __init__(self, f_3db_max=1, linlog_threshold=5, contrast_threshold=0.2, init_random=False):
        self.esim = esim_torch.ESIM(contrast_threshold, contrast_threshold, 0, init_random=init_random)
        self.filter = Filter(f_3db_max, linlog_threshold)

    def __call__(self, img, t):
        img = self.filter(img, t)
        img_torch = torch.from_numpy(img[None]).cuda().float()
        t_ns = torch.LongTensor([t * 1e9]).cuda()
        events = self.esim.forward(img_torch, t_ns)

        if events is None or len(events['p']) == 0:
            events = None
        else:
            events = evlicious.Events(x=events['x'].cpu().numpy().astype("uint16"),
                                      y=events['y'].cpu().numpy().astype("uint16"),
                                      t=(events['t'].cpu().numpy()/1e3).astype("int64"),
                                      p=events['p'].cpu().numpy().astype("int8"),
                                      height=img.shape[0], width=img.shape[1])

        return events, img



class Filter:
    def __init__(self, f_3db_max=1, linlog_threshold=5):
        self.f_3db_max = f_3db_max
        self.linlog_threshold = linlog_threshold

        self.l1 = None
        self.l1p = None

    def linlog(self, luma, threshold=5):
        output = luma.copy()
        mask = luma<=threshold
        output[mask] = luma[mask] * np.log(threshold)/ threshold
        output[~mask] = np.log(luma[~mask])
        return output

    def f_3db(self, luma, f_3db_max=1):
        return (luma + 20) / 275 * f_3db_max

    def __call__(self, luma, t):
        l1_in = self.linlog(luma, threshold=self.linlog_threshold)

        if self.l1p is None:
            self.l1 = l1_in.copy()
            self.l1p = l1_in.copy()
            self.t = t
        else:
            dt  = t - self.t
            self.t = t
            tau  = 1/ (2*np.pi * self.f_3db(luma, f_3db_max=self.f_3db_max))
            epsilon = np.clip(dt / tau, 0, 1)
            self.l1 = (1-epsilon) * self.l1 + epsilon * l1_in
            self.l1p = (1-epsilon) * self.l1p + epsilon * self.l1

        return self.l1p.copy()




def rotate_image_bilinear_scipy(img, alpha_rad):
    alpha_degrees = alpha_rad * 180 / np.pi
    rotated = skimage.transform.rotate(img, alpha_degrees, resize=False, center=None, order=1,
                                       mode='constant', cval=0, clip=True, preserve_range=True)

    return rotated


def shift_image_bilinear_scipy(image, shift):
    inverse_map = np.eye(3)
    inverse_map[0,-1] = shift
    rotated = skimage.transform.warp(image, inverse_map, order=1, mode='edge', preserve_range=True)

    return rotated


def angle(t, A=np.pi):
    return A * np.sin(t)

def canoncial_sin(s):
    out = np.zeros_like(s)
    periods = int(np.ceil(s[-1] / 4))
    for p in range(periods):
        s_ = s - 4 * p
        mask1 = (0 <= s_) & (s_ <= 1)
        out[mask1] = s_[mask1]
        mask2 = (1 <= s_) & (s_ <= 3)
        out[mask2] = 2 - s_[mask2]
        mask3 = (3 <= s_) & (s_ <= 4)
        out[mask3] = s_[mask3]  - 4
    return out


def g(t, A, w, phi):
    wtphi = w * t  + phi
    return A * (G(wtphi / np.pi - 1/2) - G(phi/np.pi - 1/2))

def g_inv(s, A, w, phi):
    return 1/w * (np.pi * G_inv(s/A + G(phi/np.pi - 1/2)) + np.pi/2 - phi)


def G(u):
    return 2 + 2 * np.floor(u) - (-1)**np.floor(u) * np.cos(np.pi * u)

def G_inv(g):
    floor_u = np.floor((g-1)/2)
    return floor_u + np.arccos(2 + 2 * floor_u - g) / np.pi

def canonical_angle(s, A=np.pi):
    # A sin(f^-1(s/A))
    # 0 < t < A -> 180 / (pi/2) * t
    # pi / 2 < t < 3 pi / 2
    return A * canoncial_sin(s / A)

def remap_timestamps_to_can(events, A=np.pi):
    s = g(A=A, w=1, phi=0, t=events.t / 1e6)
    s_us = (1e6 * s).astype("int64")
    return evlicious.Events(t=s_us,
                            x=events.x,
                            y=events.y,
                            p=events.p,
                            height=events.height,
                            width=events.width)

def plot_slice(events, ax, colors="rb", y=50, label=None, s=1):
    mask = events.y == y
    x = events.x[mask]
    t = events.t[mask]
    p = events.p[mask]

    ax.scatter(t[p==1], x[p==1], color=colors[0], label=label, s=s)
    ax.scatter(t[p==-1], x[p==-1], color=colors[1], s=s)


class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()


def generate_events_square(filter, image, angle_generator, A=np.pi, t_max=2*np.pi):
    events_list = []
    t_range = np.linspace(0, t_max, 10000)
    all_angles = angle_generator(t_range, A=A)

    for t, a in tqdm.tqdm(zip(t_range, all_angles), total=len(all_angles)):
        # Rotate by 30 degrees around the image center
        #rotated = rotate_image_bilinear_scipy(image, a)
        shifted = shift_image_bilinear_scipy(image, a)
        events, image_filtered = filter(shifted, t)

        if events is not None:
            events_list.append(events)

    events = evlicious.tools.stack(events_list)
    return events


if __name__ == '__main__':


    # constructor
    esim_fast = NonIdealEsim(f_3db_max=1000, linlog_threshold=5, contrast_threshold=0.2)
    esim_medium = NonIdealEsim(f_3db_max=200, linlog_threshold=5, contrast_threshold=0.2)
    esim_slow = NonIdealEsim(f_3db_max=20, linlog_threshold=5, contrast_threshold=0.2)

    filters = dict(
        fast=NonIdealEsim(f_3db_max=1000, linlog_threshold=5, contrast_threshold=0.2),
        medium=NonIdealEsim(f_3db_max=200, linlog_threshold=5, contrast_threshold=0.2),
        slow=NonIdealEsim(f_3db_max=20, linlog_threshold=5, contrast_threshold=0.2),
        can_fast = NonIdealEsim(f_3db_max=1000, linlog_threshold=5, contrast_threshold=0.2),
        can_medium = NonIdealEsim(f_3db_max=200, linlog_threshold=5, contrast_threshold=0.2),
        can_slow = NonIdealEsim(f_3db_max=20, linlog_threshold=5, contrast_threshold=0.2)
    )

    events_dict = {k: [] for k in filters}

    A = 30

    my_image = np.zeros((101, 101), dtype="uint8")
    my_image[:,:50] = 255

    t_max = 2 * np.pi
    s_max = t_max * 4 * A / t_max

    events_can = generate_events_square(filter=NonIdealEsim(f_3db_max=1000000, linlog_threshold=5, contrast_threshold=0.2, init_random=False),
                                         image=my_image, angle_generator=canonical_angle, A=A, t_max=s_max)
    events_fast = generate_events_square(filter=NonIdealEsim(f_3db_max=1000, linlog_threshold=5, contrast_threshold=0.2, init_random=False),
                                         image=my_image, angle_generator=angle, A=A, t_max=t_max)
    events_medium = generate_events_square(filter=NonIdealEsim(f_3db_max=200, linlog_threshold=5, contrast_threshold=0.2, init_random=False),
                                         image=my_image, angle_generator=angle, A=A, t_max=t_max)
    events_slow = generate_events_square(filter=NonIdealEsim(f_3db_max=30, linlog_threshold=5, contrast_threshold=0.2, init_random=False),
                                           image=my_image, angle_generator=angle, A=A, t_max=t_max)

    #events_fast_can = generate_events_square(filter=NonIdealEsim(f_3db_max=1000, linlog_threshold=5, contrast_threshold=0.2, init_random=False),
    #                                     image=my_image, angle_generator=canonical_angle, A=A, t_max=s_max)

        #panel = np.clip(np.concatenate(panel, axis=1) * 50, 0, 255).astype("uint8")

    #events_fast_can_test = remap_timestamps_to_can(events_fast, A=A)
    #evlicious.art.visualize_3d(events_fast, factor=0.00001, time_window_us=10000000)
    #evlicious.art.visualize_3d(events_fast_can, factor=0.00001, time_window_us=10000000)

    fig, ax = plt.subplots(nrows=3)
    plot_slice(events_fast, ax[0], colors="rb", y=50)
    plot_slice(events_medium, ax[1], colors="rb", y=50)
    plot_slice(events_slow, ax[2], colors="rb", y=50)

    for a in ax:
        a.set_ylabel("X [px]")
        a.set_xlabel("Time [us]")
        a.grid(axis="y")
        a.legend()

    fig.savefig("sinusoid_different_speeds.png", bbox_inches="tight")


    fig, ax = plt.subplots(nrows=3)

    events_fast_can_test = remap_timestamps_to_can(events_fast, A=A)
    events_medium_can_test = remap_timestamps_to_can(events_medium, A=A)
    events_slow_can_test = remap_timestamps_to_can(events_slow, A=A)

    plot_slice(events_can, ax[0], colors="rb", y=50, label="Canonical", s=5)
    plot_slice(events_fast_can_test, ax[0], colors="gk", y=50, label="Fast Canonicalized", s=5)

    plot_slice(events_can, ax[1], colors="rb", y=50, label="Canonical", s=5)
    plot_slice(events_medium_can_test, ax[1], colors="gk", y=50, label="Fast Canonicalized", s=5)

    plot_slice(events_can, ax[2], colors="rb", y=50, label="Canonical", s=5)
    plot_slice(events_slow_can_test, ax[2], colors="gk", y=50, label="Slow Canonicalized", s=5)

    for a in ax:
        a.set_ylabel("X [px]")
        a.set_xlabel("Time [us]")
        a.grid(axis="y")
        a.legend()

    fig.savefig("sinusoid_different_speeds_canonical.png", bbox_inches="tight")

    plt.show()
