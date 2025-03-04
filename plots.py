import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

# for some reason, when trying to extract row like df['steps_total'], it returns dataframe with line indexes cycling from 0 to 29 and the repeating. It might be the issue, because steps_total are sorted in the log  file itself
# Reduce the number of ticks
def reduce_ticks(ax, x_step=5000, y_step=500):
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_step))
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))

def format_ticks(value, tick_type='k'):
    if tick_type == 'k':
        # Format in thousands (e.g., 1k, 5k, etc.)
        return f'{value * 1e-3:.0f}k'
    elif tick_type == 'M':
        # Format in millions (e.g., 1M, 2M, etc.)
        return f'{value * 1e-6:.0f}M'
    else:
        return str(value)

# Load the data
file_path = "logs/sem2-ml_course_work-example_QR_DQN.py-q-net-03-01-08-21-zvkcx.out"  # Change this to the actual file path
df = pd.read_csv(file_path, delim_whitespace=True)

# Replace 'nan' values in loss with NaN (if applicable)

# Create the figure and axes
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
# Plot 1: Episode length (steps_local) vs. steps_total
axes[0].plot(df['steps_total'], df['steps_local'], label="Episode Length", color="blue")
axes[0].set_xlabel("Total Steps")
axes[0].set_ylabel("Episode Length")
reduce_ticks(axes[0], x_step=50000, y_step=100)
axes[0].set_title("Episode Length vs. Total Steps")
axes[0].grid(False)
axes[0].legend()
axes[0].set_xlim(-10000, 1500000)
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_ticks(x, tick_type='k')))

# Plot 2: Reward vs. steps_total
axes[1].plot(df['steps_total'], df['running_reward'], label="Episode Reward", color="green")
axes[1].set_xlabel("Total Steps")
axes[1].set_ylabel("Episode Reward")
reduce_ticks(axes[1], x_step=50000, y_step=100)
axes[1].set_title("Reward vs. Total Steps")
axes[1].grid(False)
axes[1].legend()
axes[1].set_xlim(-10000, 1500000)
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_ticks(x, tick_type='k')))

# Plot 3: Loss vs. steps_total
axes[2].plot(df['steps_total'], df['loss'], label="Loss", color="red")
axes[2].set_xlabel("Total Steps")
axes[2].set_ylabel("Loss")
reduce_ticks(axes[2], x_step=50000, y_step=1)
axes[2].set_title("Loss vs. Total Steps")
axes[2].grid(False)
axes[2].legend()
axes[2].set_xlim(-10000, 1500000)
axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_ticks(x, tick_type='k')))


# Adjust layout and show plots
plt.tight_layout()
plt.show()