import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import multivariate_normal

# --- 1. Data & Distribution Setup ---
np.random.seed(42)
# Component 1: Morning Rush (9am, 15km/h)
mu1 = [9, 15]
cov1 = [[1.5, -2], [-2, 100]]
# Component 2: Afternoon Flow (3pm, 50km/h)
mu2 = [15, 50]
cov2 = [[4, 0], [0, 400]]

# Define the grid for the Heatmap
time_grid = np.linspace(0, 24, 100)
speed_grid = np.linspace(0, 100, 100)
T, S = np.meshgrid(time_grid, speed_grid)
grid_coords = np.vstack([T.ravel(), S.ravel()]).T


def calculate_mixture_z(w):
    """Calculates the joint PDF based on the slider weight."""
    rv1 = multivariate_normal(mu1, cov1)
    rv2 = multivariate_normal(mu2, cov2)
    Z = w * rv1.pdf(grid_coords) + (1-w) * rv2.pdf(grid_coords)
    return Z.reshape(T.shape)


# --- 2. Figure & Layout ---
fig = plt.figure(figsize=(12, 8))
# Adjust layout to make room for sliders and titles
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9,
                    top=0.85, wspace=0.2, hspace=0.2)

# Main Joint Distribution Plot (Center)
ax_joint = plt.subplot2grid((5, 5), (1, 1), colspan=3, rowspan=3)
ax_joint.set_xlabel("Time of Day (hrs)", fontweight='bold')
ax_joint.set_ylabel("Average Speed (km/h)", fontweight='bold')

# Marginal Plots (Top and Right)
ax_marg_y = plt.subplot2grid((5, 5), (0, 1), colspan=3, sharex=ax_joint)
ax_marg_x = plt.subplot2grid((5, 5), (1, 4), rowspan=3, sharey=ax_joint)

# Initial State
init_w = 0.5
Z = calculate_mixture_z(init_w)

# Plotting the Heatmap
contour_plot = ax_joint.contourf(T, S, Z, levels=50, cmap='viridis')

# Plotting Marginal p(y) - Time (Line only)
marg_y_data = np.sum(Z, axis=0)
line_y, = ax_marg_y.plot(time_grid, marg_y_data, color='teal', lw=2.5)
ax_marg_y.set_title(r"$p(y = \mathrm{time})$", fontsize=12)

# Plotting Marginal p(x) - Speed (Line only)
marg_x_data = np.sum(Z, axis=1)
line_x, = ax_marg_x.plot(marg_x_data, speed_grid, color='teal', lw=2.5)
ax_marg_x.set_title(r"$p(x = \mathrm{speed})$", fontsize=12, pad=20)

# Clean up marginal axes (remove tick labels to keep it focused)
plt.setp(ax_marg_y.get_xticklabels(), visible=False)
plt.setp(ax_marg_x.get_yticklabels(), visible=False)

# Add Reference Points from your slide
ax_joint.scatter(10, 10, color='red', s=150, marker='*',
                 edgecolors='white', label='10am/10kmh', zorder=5)
ax_joint.scatter(2, 10, color='orange', s=100, marker='X',
                 edgecolors='white', label='2am/10kmh', zorder=5)
ax_joint.legend(loc='upper right', fontsize=9)

# --- 3. Interaction Logic ---
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_w = Slider(ax_slider, 'Rush Hour Weight', 0.0, 1.0, valinit=init_w)


def update(val):
    global contour_plot  # Fixed: Declared first

    w = slider_w.val
    new_Z = calculate_mixture_z(w)

    # Redraw Joint Contour
    for c in contour_plot.collections:
        c.remove()
    contour_plot = ax_joint.contourf(T, S, new_Z, levels=50, cmap='viridis')

    # Update Marginal p(y) Line
    new_marg_y = np.sum(new_Z, axis=0)
    line_y.set_ydata(new_marg_y)
    ax_marg_y.set_ylim(0, np.max(new_marg_y) * 1.1)

    # Update Marginal p(x) Line
    new_marg_x = np.sum(new_Z, axis=1)
    line_x.set_xdata(new_marg_x)
    ax_marg_x.set_xlim(0, np.max(new_marg_x) * 1.1)

    fig.canvas.draw_idle()


slider_w.on_changed(update)

plt.show()
