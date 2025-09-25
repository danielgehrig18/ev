import matplotlib.pyplot as plt
import numba
import numpy as np
import pypose as pp

from scipy.interpolate import PchipInterpolator

from ev.lie_group_utils import Ad_vee_se3, ad_vee_se3, exp_se3, log_se3
from ev.plot import plot_positions_and_frames, plot_velocities


def generate_random_phi(time_range, n=5):
    t0, t1 = time_range
    #phi_values = np.linspace(0, 1, num=n, endpoint=True)
    values = np.concatenate([np.array([0]), np.sort(np.random.rand(n-2)), np.array([1])])
    times = np.linspace(t0, t1, num=n, endpoint=True)
    phi = PchipInterpolator(times, values)

    phi.times = times
    phi.values = values

    return phi

def random_spline(n_poses, pose_range, velocity_range, acceleration_range, time_range):
    # generate random poses, and first and second order
    poses = pp.randn_se3(n_poses, sigma=pose_range).matrix().numpy()
    velocities = np.random.randn(n_poses, 6) * velocity_range
    accelerations = np.random.randn(n_poses, 6) * acceleration_range

    # generate spline
    phi = generate_random_phi(time_range=time_range)
    spline = SE3Spline(poses, velocities, accelerations, phi=phi)

    return spline

def derivative_matrix(A_i, *dj_a_p_i):
    n = len(dj_a_p_i)

    matrices = np.zeros(shape=A_i.shape[:2] + (n*6 + 1, n*6 + 1))
    matrices[...,-1,-1] = 1

    if n > 0:
        Ad_A_i = Ad_vee_se3(A_i)
        matrices[...,0:6,  0:6] = Ad_A_i
        matrices[...,0:6,   -1] = dj_a_p_i[0]
    if n > 1:
        ad_1 = ad_vee_se3(dj_a_p_i[0])
        matrices[...,6:12, 0:6] = ad_1 @  Ad_A_i
        matrices[...,6:12, 6:12] = Ad_A_i
        matrices[...,6:12,   -1] = dj_a_p_i[1]
    if n > 2:
        ad_2 = ad_vee_se3(dj_a_p_i[1])
        matrices[...,12:18,0:6] = (ad_2 + ad_1 @ ad_1) @ Ad_A_i
        matrices[...,12:18,6:12] = 2 * ad_1 @ Ad_A_i
        matrices[...,12:18,12:18] = Ad_A_i
        matrices[...,12:18, -1] = dj_a_p_i[2]

    dS_0 = np.zeros(shape=(A_i.shape[1], n*6 +1)) # n_samples x 13
    dS_0[...,-1] = 1

    path, _ = np.einsum_path("nab,nbc,ncd,nde,nef,nf->na", *matrices, dS_0)
    dS = np.einsum("nab,nbc,ncd,nde,nef,nf->na", *matrices, dS_0, optimize=path)

    return dS[...,:-1]

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

    def sample(self, t, n=0, eps=1e-9):
        # map the timestamp through the spline
        phi = self.phi(t) * (self.num_poses - 1)
        phi = np.clip(phi, 0, self.num_poses - 1 - eps)

        # figure out which spline is meant where, and what is the offset
        idx = phi.astype("int32")
        t_norm = phi - idx
        S, dS = self._scatter_sample(idx, t_norm, n=n)
        dS = dS.reshape(len(dS), n, -1) if n > 0 else dS.reshape(len(dS), 0, 0)

        matrix = np.zeros(shape=S.shape[:-2] + (n,n), dtype="float32")
        if n > 0:
            d1phi = self.phi(t, nu=1) * (self.num_poses - 1)
            matrix[...,0,0] = d1phi
        if n > 1:
            d2phi = self.phi(t, nu=2) * (self.num_poses - 1)
            matrix[...,1,0] = d2phi
            matrix[...,1,1] = d1phi**2
        if n > 2:
            d3phi = self.phi(t, nu=3) * (self.num_poses - 1)
            matrix[...,2,0] = d3phi
            matrix[...,2,1] = 3*d1phi*d2phi
            matrix[...,2,2] = d1phi**3

        dS = np.einsum("nij,njk->...nik", matrix, dS )

        output = {"f": S}
        for i in range(dS.shape[-2]):
            output[f"d{i+1}f"] = dS[...,i,:]

        return output

    def _scatter_sample(self, idx, t_norm, n=3):

        t_n = t_norm[None,:] ** self.n[:,None]
        a_i = self.C_0 @ t_n # n_samples x 6

        # compute products, taking scattering into accound
        p_i = self.p_i[:, idx] # 5 x num_samples x 6
        B0 = self.B0[idx]
        a_p_i = p_i * a_i[...,None]
        # compute all things
        A_i = exp_se3(a_p_i) # n_samples x 6 x 6
        path, _ = np.einsum_path("nab,nbc,ncd,nde,nef,nfg->nag", *A_i, B0)
        S = np.einsum("nab,nbc,ncd,nde,nef,nfg->nag", *A_i, B0, optimize=path)

        dj_a_p_i = []
        if n > 0:
            dj_a_p_i.append(p_i * (self.C_1 @ t_n)[...,None])
        if n > 1:
            dj_a_p_i.append(p_i * (self.C_2 @ t_n)[...,None])
        if n > 2:
            dj_a_p_i.append(p_i * (self.C_3 @ t_n)[...,None])

        dS = derivative_matrix(A_i, *dj_a_p_i)

        return S, dS

    def plot(self):
        t_sample = np.linspace(self.t0, self.t1, num=10000, endpoint=True)
        data = self.sample(t_sample)

        plot_positions_and_frames(data['f'][:,:3,3], data['f'][:,:3,:3])

        fig, ax = plt.subplots(nrows=3, ncols=2)
        plot_velocities(t_sample, data['d1f'], "velocity", ax=ax[0], fig=fig)
        plot_velocities(t_sample, data['d2f'], "acceleration", ax=ax[1], fig=fig)
        plot_velocities(t_sample, data['d3f'], "jerk", ax=ax[2], fig=fig)

