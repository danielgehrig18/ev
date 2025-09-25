import numpy as np 
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')
plt.rcParams['lines.linewidth'] = 2.0


def plot_canonical(sampling="regular"):
    # plot canonical with levels
    fig, ax = plt.subplots(figsize=(4, 2))

    if sampling == "event":
        theta = 0.3333 
        for n in range(-3, 3+1):
            ax.plot([0, 4], [n*theta, n*theta], color="r", alpha=0.5, linestyle="-.", linewidth=2)
    elif sampling == "regular":
        dt = 0.25 
        for n in range(0, 16+1):
            ax.plot([dt*n, dt*n], [-1, 1], color="r", alpha=0.5, linestyle="-.", linewidth=2)

    s = np.linspace(0, 4, 1000, endpoint=True)
    ax.plot(s, canonical(s))
    ax.set_xlabel(r"Arc Length $s$")
    ax.set_ylabel(r"Canonical signal $\mathbf{x}^*(t)$")

    return fig, ax 


def plot_non_canonical(sampling="regular"):
    fig, ax = plt.subplots(figsize=(2*np.pi, 2))

    if sampling == "event":
        theta = 0.3333 
        for n in range(-3, 3+1):
            ax.plot([0, 2*np.pi], [n*theta, n*theta], color="r", alpha=0.5, linestyle="-.", linewidth=2)
    elif sampling == "regular":
        dt = np.pi * 2 / 16 
        for n in range(0, 16+1):
            ax.plot([dt*n, dt*n], [-1, 1], color="r", alpha=0.5, linestyle="-.", linewidth=2)

    t = np.linspace(0, 2 * np.pi, 1000, endpoint=True)

    for j, gamma in enumerate([0.5, 1, 2]):
        ax.plot(t, f_gamma(t, gamma), label=r"$\mathbf{x}_{" + str(j) + "}(t)=\mathbf{x}^*(\phi_{"+str(j)+"}(t))$")

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Signal $\mathbf{x}_1(t)=\mathbf{x}^*(\phi_1(t))$")

    return fig, ax 


# sample from \int_0^x |d/dt (sin (t)| dt
# this integral becomes \int_0^{x} |cos(u)|du
# we have the integral equal s(t) = G(t/pi - 1/2) - G(- 1/2)
# and G(u) = 2 + 2 * np.floor(u) - (-1)^floor(u) * np.cos(pi * u)

def canonical(t):
    x = np.zeros_like(t)
    x[t < 1] = t[t < 1]
    x[(t >= 1) & (t <= 3)] = 2 - t[(t >= 1) & (t <= 3)]
    x[t >= 3] = t[t >= 3] - 4
    return x

def G(u):
    return 2 + 2 * np.floor(u) - (-1) ** np.floor(u) * np.cos(np.pi * u)


def phi(t):
    return G(t / np.pi - 1/2) - G(-1/2)

def f_gamma(t, gamma):
    t_star = (t / (2 * np.pi)) ** gamma * 2 * np.pi
    return canonical(phi(t_star))


fig_can, ax_can = plot_canonical(sampling=None)
fig_can_level, ax_can_level = plot_canonical(sampling="event")
fig_can_regular, ax_can_regular = plot_canonical(sampling="regular")


fig_non_can, ax_non_can = plot_non_canonical(sampling=None)
fig_non_can_level, ax_non_can_level = plot_non_canonical(sampling="event")
fig_non_can_regular, ax_non_can_regular = plot_non_canonical(sampling="regular")


fig_phi, ax_phi = plt.subplots(figsize=(2*np.pi, 4))

t = np.linspace(0, 2 * np.pi, 1000, endpoint=True)
for j, gamma in enumerate([0.5, 1, 2]):
    ax_phi.plot(t, phi((t / (2 * np.pi)) ** gamma * 2 * np.pi), label=rf"$s=\phi_{j}(t)$")

ax_phi.set_xlabel(r"Time $t$")
ax_phi.set_ylabel(r"Arc Length $s$")


fig_phi.savefig("phi.png", bbox_inches="tight")
fig_can.savefig("can.png", bbox_inches="tight")
fig_can_regular.savefig("can_regular.png", bbox_inches="tight")
fig_can_level.savefig("can_level.png", bbox_inches="tight")
fig_non_can.savefig("non_can.png", bbox_inches="tight")
fig_non_can_regular.savefig("non_can_regular.png", bbox_inches="tight")
fig_non_can_level.savefig("non_can_level.png", bbox_inches="tight")


plt.show()

