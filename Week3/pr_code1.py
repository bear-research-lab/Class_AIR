import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm

# --- Configuration ---
TRUE_MU = 0.0
TRUE_STD = 1.0
INITIAL_N = 50
INITIAL_EST_MU = 2.0  # Initial guess for the mean

# Create figure and layout
fig, (ax_dist, ax_lik) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.25)

# --- 1. Left Plot: Data & Distributions ---
x_axis = np.linspace(-5, 5, 500)
data = np.random.normal(TRUE_MU, TRUE_STD, INITIAL_N)

# Plot True Distribution
ax_dist.plot(x_axis, norm.pdf(x_axis, TRUE_MU, TRUE_STD),
             'g--', label='True Distribution')
# Plot Histogram
_, bins, hist_patches = ax_dist.hist(
    data, bins=20, density=True, alpha=0.3, color='blue', label='Data Histogram')
# Plot Estimated Distribution
est_line, = ax_dist.plot(x_axis, norm.pdf(
    x_axis, INITIAL_EST_MU, TRUE_STD), 'r', lw=2, label='Estimated Dist')
# Rug plot (individual points)
rug = ax_dist.plot(data, np.zeros_like(data), 'b|', ms=10, label='Samples')[0]

ax_dist.set_title("Data Sampling & Distribution Match")
ax_dist.set_ylim(-0.05, 0.6)
ax_dist.legend()

# --- 2. Right Plot: Likelihood Curve ---
mu_range = np.linspace(-4, 4, 200)


def calc_log_likelihood(mu, current_data):
    # Log-likelihood is easier to visualize as it prevents underflow
    return np.sum(norm.logpdf(current_data, loc=mu, scale=TRUE_STD))


# Initial Likelihood Curve
lik_values = [calc_log_likelihood(m, data) for m in mu_range]
lik_line, = ax_lik.plot(mu_range, lik_values, color='purple', lw=2)
# Current point on likelihood curve
curr_lik_point, = ax_lik.plot(
    INITIAL_EST_MU, calc_log_likelihood(INITIAL_EST_MU, data), 'ro')

ax_lik.set_title("Log-Likelihood vs. $\mu$")
ax_lik.set_xlabel("$\mu$ (Estimated Mean)")
ax_lik.set_ylabel("Log-Likelihood")

# --- 3. Sliders ---
ax_n = plt.axes([0.15, 0.1, 0.3, 0.03])
ax_mu = plt.axes([0.6, 0.1, 0.3, 0.03])

s_n = Slider(ax_n, 'N points', 5, 500, valinit=INITIAL_N, valstep=1)
s_mu = Slider(ax_mu, 'Est. $\mu$', -4.0, 4.0, valinit=INITIAL_EST_MU)

# --- Update Function ---


def update(val):
    N = int(s_n.val)
    est_mu = s_mu.val

    # Resample data if N changes (or just slice/expand)
    # For a smooth experience, we generate a fixed large pool and slice it
    global data
    np.random.seed(42)  # Keeping seed fixed for smoother slider transitions
    data = np.random.normal(TRUE_MU, TRUE_STD, N)

    # Update Histogram and Rug
    ax_dist.cla()
    ax_dist.set_title("Data Sampling & Distribution Match")
    ax_dist.plot(x_axis, norm.pdf(x_axis, TRUE_MU, TRUE_STD),
                 'g--', label='True Distribution')
    ax_dist.hist(data, bins=20, density=True, alpha=0.3, color='blue')
    ax_dist.plot(data, np.zeros_like(data), 'b|', ms=10)

    # Update Estimated Line
    ax_dist.plot(x_axis, norm.pdf(x_axis, est_mu, TRUE_STD),
                 'r', lw=2, label='Estimated Dist')
    ax_dist.legend(loc='upper right')
    ax_dist.set_ylim(-0.05, 0.6)

    # Update Likelihood Curve
    new_lik_values = [calc_log_likelihood(m, data) for m in mu_range]
    lik_line.set_ydata(new_lik_values)
    ax_lik.set_ylim(min(new_lik_values), max(new_lik_values) + 10)

    # Update Red Dot
    curr_lik_point.set_data([est_mu], [calc_log_likelihood(est_mu, data)])

    fig.canvas.draw_idle()


s_n.on_changed(update)
s_mu.on_changed(update)

plt.show()
