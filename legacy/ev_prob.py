import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

def compute_signal(t):
    return np.sin(10*t)

def p_e1(L, L0, C):
    dL = (L-L0)/C
    p = np.array([-1, 1])
    return 0.25 * ( erf(dL[None] + 1 - p[:,None]) - erf(dL[None] - 1 - p[:,None]))

def marginalize(L, p_e1, C, sigma):
    dL = (L[:,None] - L[None,:]) / C
    p = np.array([-1, 1])
    p_e_e = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(p[:,None,None]*dL[None]-1)**2 / (2 * sigma**2))

    mask = np.arange(len(L))[:,None] < np.arange(len(L))[None,:]
    p_e_e[:,mask] = 0

    p_e2 = np.einsum("mij,pj->mi", p_e_e, p_e1)
    return p_e2


if __name__ == '__main__':
    timestamps = np.linspace(0, 1, 1000)
    signal = compute_signal(timestamps)

    C = 0.1
    p1 = p_e1(signal, signal[0], C)
    p2 = marginalize(signal, p1, C, sigma=1)

    fig, ax = plt.subplots()
    ax.plot(timestamps, signal, "g")
    ax.plot(timestamps, p1[0], 'r')
    ax.plot(timestamps, p1[1], 'b')
    ax.plot(timestamps, p2[0], 'r')
    ax.plot(timestamps, p2[1], 'b')
    plt.show()