import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider

# -----------------------------------
# Distributions
# -----------------------------------
mu_init = np.array([-2, 0])
cov_init = np.array([[0.8, 0],
                     [0, 0.8]])

mu1 = np.array([3, 2])
cov1 = np.array([[0.6, 0.2],
                 [0.2, 1]])

mu2 = np.array([3, -2])
cov2 = np.array([[1, -0.3],
                 [-0.3, 0.6]])

# -----------------------------------
# Grid
# -----------------------------------
x = np.linspace(-5, 6, 28)
y = np.linspace(-5, 5, 28)

X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

p_init = multivariate_normal(mu_init, cov_init).pdf(pos)

p_data = (
    0.5 * multivariate_normal(mu1, cov1).pdf(pos)
    + 0.5 * multivariate_normal(mu2, cov2).pdf(pos)
)

# -----------------------------------
# Sampling helpers
# -----------------------------------


def sample_p_data(n):
    mode_choice = np.random.rand(n) < 0.5

    samples = np.zeros((n, 2))

    n1 = np.sum(mode_choice)
    n2 = n - n1

    samples[mode_choice] = np.random.multivariate_normal(mu1, cov1, n1)
    samples[~mode_choice] = np.random.multivariate_normal(mu2, cov2, n2)

    return samples


# -----------------------------------
# Fixed random transport pairs
# -----------------------------------
np.random.seed(0)

n_pairs = 6000
x0_samples = np.random.multivariate_normal(mu_init, cov_init, n_pairs)
z_samples = sample_p_data(n_pairs)

vel_samples = z_samples - x0_samples

# -----------------------------------
# Approximate marginal vector field
# u_t(x) = E[z - x0 | x_t = x]
# -----------------------------------


def compute_marginal_vector_field(t, bandwidth=0.65):

    xt_samples = (1 - t) * x0_samples + t * z_samples

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            grid_point = np.array([X[i, j], Y[i, j]])

            diff = xt_samples - grid_point
            dist2 = np.sum(diff**2, axis=1)

            weights = np.exp(-dist2 / (2 * bandwidth**2))
            weights_sum = np.sum(weights) + 1e-8

            avg_vel = np.sum(weights[:, None] *
                             vel_samples, axis=0) / weights_sum

            U[i, j] = avg_vel[0]
            V[i, j] = avg_vel[1]

    speed = np.sqrt(U**2 + V**2)

    U_norm = U / (speed + 1e-6)
    V_norm = V / (speed + 1e-6)

    return U_norm, V_norm, speed, xt_samples


# -----------------------------------
# Initial field
# -----------------------------------
t0 = 0.0
U_norm, V_norm, speed, xt_samples = compute_marginal_vector_field(t0)

# -----------------------------------
# Plot
# -----------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(bottom=0.18)

# show distributions
ax.contour(X, Y, p_init, levels=8, cmap="Oranges", alpha=0.45)
ax.contour(X, Y, p_data, levels=8, cmap="Blues", alpha=0.45)

# show current transported particles
particle_plot = ax.scatter(
    xt_samples[:500, 0],
    xt_samples[:500, 1],
    s=8,
    alpha=0.35,
    label=r"particles $x_t$"
)

# show vector field
q = ax.quiver(
    X, Y,
    U_norm, V_norm,
    speed,
    scale=45,
    alpha=0.85
)

ax.scatter(*mu_init, s=150, c="black", label=r"$p_{init}$ center")
ax.scatter(
    [mu1[0], mu2[0]],
    [mu1[1], mu2[1]],
    s=150,
    c="red",
    label=r"$p_{data}$ modes"
)

title = ax.set_title(r"Approx. marginal vector field $u_t(x)$, $t=0.00$")

ax.set_xlim([-5, 6])
ax.set_ylim([-5, 5])
ax.set_aspect("equal")
ax.legend(loc="upper left")

# -----------------------------------
# Slider
# -----------------------------------
ax_t = plt.axes([0.2, 0.06, 0.65, 0.03])

slider_t = Slider(
    ax=ax_t,
    label="t",
    valmin=0.0,
    valmax=1.0,
    valinit=t0,
    valstep=0.02
)

# -----------------------------------
# Update
# -----------------------------------


def update(val):

    t = slider_t.val

    U_norm, V_norm, speed, xt_samples = compute_marginal_vector_field(t)

    q.set_UVC(U_norm, V_norm, speed)

    particle_plot.set_offsets(xt_samples[:500])

    title.set_text(rf"Approx. marginal vector field $u_t(x)$, $t={t:.2f}$")

    fig.canvas.draw_idle()


slider_t.on_changed(update)

plt.show()
