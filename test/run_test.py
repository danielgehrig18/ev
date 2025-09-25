import pypose as pp
import numpy as np
import matplotlib.pyplot as plt

from ev.spline import SE3Spline, generate_random_phi
from ev.sample import send_on_delta_sample
from ev.plot import plot_positions_and_frames
from ev.process import compute_frame

np.random.seed(4)

n_poses = 5
poses = pp.randn_se3(n_poses, sigma=2).matrix().numpy()
velocities = np.random.randn(n_poses, 6) * 2
accelerations = np.random.randn(n_poses, 6) * 0.2

timestamps = np.array([0, 1, 2])
# timestamps = np.array([0, 1])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    phi = generate_random_phi(time_range=[0, 10])
    spline = SE3Spline(poses, velocities, accelerations, phi=phi)
    t = np.linspace(spline.t0, spline.t1, num=10, endpoint=True)
    #t = send_on_delta_sample(spline, threshold=0.8, debug=False)
    output = spline.sample(t, n=3)

    position = output['f'][..., :3, 3]
    frame_r, frame_t = compute_frame(output)
    plot_positions_and_frames(position, frame_t, color=np.random.rand(3), ax=ax, scale=(i+1) * 0.2)

plt.show()