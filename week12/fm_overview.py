import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# -----------------------------------
# Create p_init
# -----------------------------------
mu_init = np.array([-2, 0])
cov_init = np.array([[0.8, 0],
                     [0, 0.8]])

# -----------------------------------
# Create p_data: mixture of Gaussians
# -----------------------------------
mu1 = np.array([3, 2])
cov1 = np.array([[0.6, 0.2],
                 [0.2, 1]])

mu2 = np.array([3, -2])
cov2 = np.array([[1, -0.3],
                 [-0.3, 0.6]])

# -----------------------------------
# Grid
# -----------------------------------
x = np.linspace(-5, 6, 40)
y = np.linspace(-5, 5, 40)

X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

p_init = multivariate_normal(mu_init, cov_init).pdf(pos)

p_data = (
    0.5 * multivariate_normal(mu1, cov1).pdf(pos)
    + 0.5 * multivariate_normal(mu2, cov2).pdf(pos)
)

# -----------------------------------
# Helper: sample from p_data
# -----------------------------------


def sample_p_data(n):
    samples = []

    for _ in range(n):
        if np.random.rand() < 0.5:
            z = np.random.multivariate_normal(mu1, cov1)
        else:
            z = np.random.multivariate_normal(mu2, cov2)

        samples.append(z)

    return np.array(samples)


# -----------------------------------
# Approximate marginal vector field u_t(x)
# u_t(x) = E[z - x0 | x_t = x]
# -----------------------------------
np.random.seed(0)

t_field = 0.5
n_pairs = 8000
bandwidth = 0.7

x0_samples = np.random.multivariate_normal(mu_init, cov_init, n_pairs)
z_samples = sample_p_data(n_pairs)

xt_samples = (1 - t_field) * x0_samples + t_field * z_samples
vel_samples = z_samples - x0_samples

U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):

        grid_point = np.array([X[i, j], Y[i, j]])

        diff = xt_samples - grid_point
        dist2 = np.sum(diff**2, axis=1)

        weights = np.exp(-dist2 / (2 * bandwidth**2))
        weights_sum = np.sum(weights) + 1e-8

        avg_vel = np.sum(weights[:, None] * vel_samples, axis=0) / weights_sum

        U[i, j] = avg_vel[0]
        V[i, j] = avg_vel[1]

norm = np.sqrt(U**2 + V**2)
U_norm = U / (norm + 1e-6)
V_norm = V / (norm + 1e-6)

# -----------------------------------
# Random sample movement
# -----------------------------------
n_samples = 40
n_steps = 20

samples_init = np.random.multivariate_normal(mu_init, cov_init, n_samples)
samples_target = sample_p_data(n_samples)

trajectories = []

for p0, p1 in zip(samples_init, samples_target):
    traj = np.array([
        (1 - t) * p0 + t * p1
        for t in np.linspace(0, 1, n_steps)
    ])
    trajectories.append(traj)

# -----------------------------------
# Plot
# -----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# -----------------------------------
# Subplot 1: distributions only
# -----------------------------------
ax = axes[0]

ax.contour(X, Y, p_init, levels=8, cmap='Oranges', alpha=0.8)
ax.contour(X, Y, p_data, levels=8, cmap='Blues', alpha=0.8)

ax.scatter(*mu_init, s=150, c='black', label=r'$p_{init}$ center')
ax.scatter(
    [mu1[0], mu2[0]],
    [mu1[1], mu2[1]],
    s=150,
    c='red',
    label=r'$p_{data}$ modes'
)

ax.set_title("Distributions only")
ax.set_xlim([-5, 6])
ax.set_ylim([-5, 5])
ax.set_aspect('equal')
ax.legend()

# -----------------------------------
# Subplot 2: marginal vector field
# -----------------------------------
ax = axes[1]

ax.contour(X, Y, p_init, levels=8, cmap='Oranges', alpha=0.5)
ax.contour(X, Y, p_data, levels=8, cmap='Blues', alpha=0.5)

ax.quiver(
    X, Y,
    U_norm, V_norm,
    norm,
    scale=80,
    alpha=0.8
)

ax.scatter(*mu_init, s=150, c='black')
ax.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], s=150, c='red')

ax.set_title(r"Approx. marginal vector field $u_t(x)$ at $t=0.5$")
ax.set_xlim([-5, 6])
ax.set_ylim([-5, 5])
ax.set_aspect('equal')

# -----------------------------------
# Subplot 3: random sample movement
# -----------------------------------
ax = axes[2]

ax.contour(X, Y, p_init, levels=8, cmap='Oranges', alpha=0.5)
ax.contour(X, Y, p_data, levels=8, cmap='Blues', alpha=0.5)

for traj in trajectories:
    ax.plot(traj[:, 0], traj[:, 1], alpha=0.6)
    ax.scatter(traj[0, 0], traj[0, 1], c='orange', s=20)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=20)

ax.scatter(*mu_init, s=150, c='black', label=r'$p_{init}$ center')
ax.scatter(
    [mu1[0], mu2[0]],
    [mu1[1], mu2[1]],
    s=150,
    c='red',
    label=r'$p_{data}$ modes'
)

ax.set_title(r"Samples: $x_t=(1-t)x_0+tz$")
ax.set_xlim([-5, 6])
ax.set_ylim([-5, 5])
ax.set_aspect('equal')
ax.legend()

plt.tight_layout()
plt.show()
