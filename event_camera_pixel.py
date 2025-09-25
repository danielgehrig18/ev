import matplotlib.pyplot as plt
import numpy as np

def canonical(t):
    v = np.zeros_like(t)
    v[t<1] = t[t<1]
    v[(1<=t)&(t<=3)] = 2 - t[(1<=t)&(t<=3)]
    v[(3<=t)&(t<=4)] = t[(3<=t)&(t<=4)] - 4
    return v[:,None]

def sin(t):
    return np.sin(t)

def ramp(t):
    t[t<0] = 0
    return t

def exp_ramp(t, tau):
    y = t * np.exp(-t / tau) + 2 * tau * np.exp(-t / tau) + t - 2 * tau
    y[t<0] = 0
    return y

def function(t, I0, I1, t0, t1):
    alpha = (I1 - I0) / (t1 - t0)
    return I0 + alpha * (ramp(t - t0) - ramp(t - t1))

def response(t, I0, I1, t0, t1, tau):
    alpha = (I1 - I0) / (t1 - t0)
    return I0 + alpha * (exp_ramp(t - t0, tau) - exp_ramp(t - t1, tau))



if __name__ == '__main__':
    t = np.linspace(0, 10, 10000)
    I0 = 1
    t0 = 2
    I1 = 2
    t1 = 8
    tau = .2

    fig, ax = plt.subplots()
    ax.plot(t, function(t, I0, I1, t0, t1), label="Input")
    ax.plot(t, response(t, I0, I1, t0, t1, tau), label="Output")
    ax.legend()
    plt.show()
