import scienceplots
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
plt.rcParams['lines.linewidth'] = 2.0

def plot_stuff(sampling="regular"):
    fig, ax = plt.subplots()

    t = np.linspace(0, 1, 100)
    v = 2 * np.sin(0.5 * 2*np.pi* t + 2.3)

    if sampling == "regular":
        for i in range(5+1):
            ax.plot([i/5, i/5], [-2, 2], color="r", alpha=0.5, linewidth=2, linestyle="--")
    elif sampling == "level":
        for i in range(5+1):
            p = 4 * i / 5 - 2
            ax.plot([0, 1], [p, p], color="r", alpha=0.5, linewidth=2, linestyle="--")

    ax.plot(t, v, color="b")
    ax.set_xlabel("Time [s]")
    ax.set_ylim([-2, 2])
    ax.set_xlim([0, 1])

    return fig

plot_stuff(sampling="level").savefig("random_sinusoid_level.png", bbox_inches='tight')
plot_stuff(sampling="regular").savefig("random_sinusoid_regular.png", bbox_inches='tight')
plot_stuff(sampling="None").savefig("random_sinusoid.png", bbox_inches='tight')
plt.show()