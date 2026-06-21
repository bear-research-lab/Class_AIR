import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider, CheckButtons

# -----------------------------------
# distributions
# -----------------------------------
mu_init = np.array([-2, 0])
cov_init = np.array([[0.8, 0], [0, 0.8]])

mu1 = np.array([3, 2])
cov1 = np.array([[0.6, 0.2], [0.2, 1]])

mu2 = np.array([3, -2])
cov2 = np.array([[1, -0.3], [-0.3, 0.6]])

x = np.linspace(-5, 6, 40)
y = np.linspace(-5, 5, 40)

X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

p_init = multivariate_normal(mu_init, cov_init).pdf(pos)

p_data = (
    .5*multivariate_normal(mu1, cov1).pdf(pos)
    + .5*multivariate_normal(mu2, cov2).pdf(pos)
)

# -----------------------------------
# vector field
# -----------------------------------
U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):

        point = np.array([X[i, j], Y[i, j]])

        d1=np.linalg.norm(point-mu1)
        d2=np.linalg.norm(point-mu2)

        target=mu1 if d1 < d2 else mu2

        vec=target-point

        U[i, j]=vec[0]
        V[i, j]=vec[1]

norm=np.sqrt(U**2+V**2)

U=U/(norm+1e-6)
V=V/(norm+1e-6)

# -----------------------------------
# sample generation
# -----------------------------------
np.random.seed(0)


def create_samples(n):

    samples = np.random.multivariate_normal(
        mu_init,
        cov_init,
        n
    )

    targets = []

    for _ in range(n):

        # sample a target mode
        if np.random.rand() < 0.5:
            target = np.random.multivariate_normal(mu1, cov1)
        else:
            target = np.random.multivariate_normal(mu2, cov2)

        targets.append(target)

    return samples, np.array(targets)
samples, target=create_samples(40)

# -----------------------------------
# figure
# -----------------------------------
fig, ax=plt.subplots(figsize=(9, 7))
plt.subplots_adjust(left=.25, bottom=.25)

# distributions
c1=ax.contour(
    X, Y, p_init,
    levels=8,
    cmap='Oranges',
    alpha=.7
)

c2=ax.contour(
    X, Y, p_data,
    levels=8,
    cmap='Blues',
    alpha=.7
)

# vector field
q=ax.quiver(
    X, Y, U, V,
    norm,
    scale=35
)

scatter=ax.scatter([], [], c='blue')

ax.scatter(
    *mu_init,
    s=200,
    c='black'
)

ax.scatter(
    [mu1[0], mu2[0]],
    [mu1[1], mu2[1]],
    s=200,
    c='red'
)

ax.set_xlim([-5, 6])
ax.set_ylim([-5, 5])
ax.set_aspect('equal')

# -----------------------------------
# sliders
# -----------------------------------
ax_t=plt.axes([0.25, 0.12, .65, .03])
slider_t=Slider(
    ax_t,
    't',
    0,
    1,
    valinit=0
)

ax_n=plt.axes([0.25, .06, .65, .03])
slider_n=Slider(
    ax_n,
    'samples',
    5,
    100,
    valinit=40,
    valstep=1
)

# -----------------------------------
# check buttons
# -----------------------------------
rax=plt.axes([0.02, .4, .15, .15])

checks=CheckButtons(
    rax,
    ['field', 'dist'],
    [True, True]
)

# -----------------------------------
# update function
# -----------------------------------


def update(val):

    global samples, target, c1, c2

    n = int(slider_n.val)
    t = slider_t.val

    samples, target = create_samples(n)

    current = (1-t)*samples+t*target

    scatter.set_offsets(current)

    # vector field visibility
    q.set_visible(checks.get_status()[0])

    # remove old contours
    if c1 is not None:
        c1.remove()

    if c2 is not None:
        c2.remove()

    # redraw if distribution enabled
    if checks.get_status()[1]:

        c1 = ax.contour(
            X, Y, p_init,
            levels=8,
            cmap='Oranges',
            alpha=.7
        )

        c2 = ax.contour(
            X, Y, p_data,
            levels=8,
            cmap='Blues',
            alpha=.7
        )

    else:
        c1 = None
        c2 = None

    fig.canvas.draw_idle()

slider_t.on_changed(update)
slider_n.on_changed(update)
checks.on_clicked(update)

update(None)

plt.show()
