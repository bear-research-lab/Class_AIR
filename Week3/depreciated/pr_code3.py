import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------------------------------------
# True Gaussian parameters
# --------------------------------------------------
np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
max_samples = 200

# x-range for plotting the true Gaussian PDF
x = np.linspace(true_mu - 4 * true_sigma, true_mu + 4 * true_sigma, 500)
true_pdf = (1 / (true_sigma * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x - true_mu) / true_sigma) ** 2
)

# --------------------------------------------------
# Figure and axes
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
plt.subplots_adjust(bottom=0.22)

# Initial sample count
init_n = 50

# Initial samples
samples = np.random.normal(true_mu, true_sigma, init_n)

# Plot true Gaussian
(pdf_line,) = ax.plot(x, true_pdf, label="True Gaussian PDF", linewidth=2)

# Histogram for samples
hist = ax.hist(
    samples,
    bins=15,
    density=True,
    alpha=0.6,
    label="Sample histogram"
)

# Scatter sampled points on x-axis
(scatter_plot,) = ax.plot(
    samples,
    np.zeros_like(samples),
    "x",
    label="Sampled points"
)

# Styling
ax.set_title("Interactive Sampling from a True Gaussian")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

# Keep y-limits stable
ax.set_ylim(-0.02, max(true_pdf) * 1.4)

# --------------------------------------------------
# Slider
# --------------------------------------------------
slider_ax = plt.axes([0.15, 0.08, 0.7, 0.04])
n_slider = Slider(
    ax=slider_ax,
    label="Number of samples",
    valmin=0,
    valmax=max_samples,
    valinit=init_n,
    valstep=1
)

# --------------------------------------------------
# Update function
# --------------------------------------------------


def update(val):
    global hist

    n = int(n_slider.val)

    # Remove old histogram bars
    for patch in hist[2]:
        patch.remove()

    # Resample
    if n > 0:
        samples = np.random.normal(true_mu, true_sigma, n)

        # New histogram
        hist = ax.hist(
            samples,
            bins=15,
            density=True,
            alpha=0.6,
            label="Sample histogram"
        )

        # Update scatter points
        scatter_plot.set_data(samples, np.zeros_like(samples))
    else:
        # Empty histogram
        hist = ([], [], [])

        # Clear scatter
        scatter_plot.set_data([], [])

    # Keep the true PDF visible
    pdf_line.set_data(x, true_pdf)

    # Remove duplicate legends and redraw
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    fig.canvas.draw_idle()


n_slider.on_changed(update)

plt.show()
