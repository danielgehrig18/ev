import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')
plt.rcParams['lines.linewidth'] = 2.0

data = pd.read_csv('./num_events_vs_val.csv')


mean_num_events = data['mean_num_events'].values
val_loss = data['tot/val'].values
use_events = data['use_events'].values
use_variation = data['use_variation'].values


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    mean_num_events[use_events & use_variation],
    val_loss[use_events & use_variation], label="1D variation events"
)
ax.plot(
    mean_num_events[use_events & ~use_variation],
    val_loss[use_events & ~use_variation], label="1D geodesic events"
)

ax.plot(
    mean_num_events[~use_events],
    val_loss[~use_events], label="1D regular samples"
)

ax.set_xlabel('Mean number of samples')
ax.set_ylabel('Mean validation loss')
ax.legend()
fig.savefig("num_samples_vs_validation_loss.png", bbox_inches='tight')
plt.show()