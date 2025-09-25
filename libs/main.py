import ev_cpp

import numpy as np
import time
import tqdm
from matplotlib import pyplot as plt

def sin(t, A, w, phi):
    if type(t) is float:
        return A * np.sin(t * w + phi)
    return A[None] * np.sin(t[:, None] * w[None] + phi[None])


def evaluate_one(disc, N, th=0.4):

    disc_gt = 100000
    t_gt = np.linspace(0, 1, disc_gt, endpoint=True)

    dts = []
    loss = []
    t = np.linspace(0, 1, disc, endpoint=True)

    for i in tqdm.tqdm(range(N)):
        A = np.random.rand(2) * 5
        w = np.random.rand(2) * 10
        phi = np.random.rand(2) * 2 * np.pi

        v_gt = sin(t_gt, A, w, phi)
        idx = ev_cpp.generate_events(v_gt, th, sin(0.0, A, w, phi))
        t_ev_gt = idx / (disc_gt - 1)

        v = sin(t, A, w, phi)

        t0 = time.perf_counter()
        idx = ev_cpp.generate_events(v, th, sin(0.0, A, w, phi))
        t1 = time.perf_counter()
        dts.append(t1 - t0)

        t_ev = idx / (disc - 1)
        if len(t_ev) > 0:
            loss.append(np.abs(t_ev - t_ev_gt).mean())

    print("Disc: ", disc)
    print("\t time: ", np.median(dts))
    print("\t loss: ", np.median(loss))

    return np.median(dts), np.median(loss)



def generate_errors_for_set_of_params():
    fig, ax = plt.subplots()
    for disc in [100000, 90000, 80000, 70000, 60000, 50000, 40000, 30000, 20000, 10000, 9000, 8000, 7000,6000, 5000, 4000, 3000]:
        dt, loss = evaluate_one(disc=disc, N=10000)
        ax.scatter([dt], [loss], label=f"Disc: {disc}")
    ax.legend()
    ax.show()


if __name__ == '__main__':
    np.random.seed(2)

    if False:
        A = np.random.rand(2) * 5
        w = np.random.rand(2) * 10
        phi = np.random.rand(2) * 2 * np.pi
        print(A.mean())

        A = np.random.rand(2) * 5
        w = np.random.rand(2) * 10
        phi = np.random.rand(2) * 2 * np.pi
        print(A.mean())

        disc_gt = 100000
        t_gt = np.linspace(0, 1, disc_gt, endpoint=True)

        v_gt = sin(t_gt, A, w, phi)
        idx = ev_cpp.generate_events(v_gt, 0.1, sin(0.0))
        t_gt = idx / (disc_gt - 1)

        exit()


    generate_errors_for_set_of_params()
