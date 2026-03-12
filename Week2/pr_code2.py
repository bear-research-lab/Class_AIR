import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# =========================================================
# Rotation matrix
# =========================================================
def rot(theta_deg):
    th = np.radians(theta_deg)
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [c, -s],
        [s,  c]
    ])


# =========================================================
# Draw arrow helper
# =========================================================
def draw_arrow(ax, start, vec, color="k", label=None, lw=3, alpha=1.0, linestyle="-"):
    ax.arrow(
        start[0], start[1],
        vec[0], vec[1],
        head_width=0.08,
        head_length=0.12,
        length_includes_head=True,
        fc=color,
        ec=color,
        linewidth=lw,
        alpha=alpha,
        linestyle=linestyle
    )

    if label is not None:
        end = start + vec
        ax.text(
            end[0] + 0.06,
            end[1] + 0.06,
            label,
            color=color,
            fontsize=14,
            fontweight="bold"
        )


# =========================================================
# Draw angle arc
# =========================================================
def draw_angle_arc(ax, center, theta_deg, radius=0.5, color="red", lw=2):
    theta_rad = np.radians(theta_deg)

    if theta_deg >= 0:
        arc = np.linspace(0, theta_rad, 100)
    else:
        arc = np.linspace(0, theta_rad, 100)

    x = center[0] + radius * np.cos(arc)
    y = center[1] + radius * np.sin(arc)
    ax.plot(x, y, color=color, linewidth=lw)

    mid = theta_rad / 2.0
    ax.text(
        center[0] + (radius + 0.12) * np.cos(mid),
        center[1] + (radius + 0.12) * np.sin(mid),
        r"$\theta$",
        color=color,
        fontsize=16,
        fontweight="bold"
    )


# =========================================================
# Figure setup
# =========================================================
def setup_axes(ax):
    ax.clear()
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-3.0, 4.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("Coordinate Transformation Between Two 2D Frames",
                 fontsize=16, fontweight="bold")


# =========================================================
# Initial values
# =========================================================
theta0 = 30.0
pbx0 = 1.2
pby0 = 0.8
tx0 = 1.5
ty0 = 1.0
axis_len = 1.0


# =========================================================
# Create figure and sliders
# =========================================================
fig, ax = plt.subplots(figsize=(13, 8))
plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.30)

ax_theta = plt.axes([0.12, 0.20, 0.75, 0.03])
ax_pbx = plt.axes([0.12, 0.15, 0.75, 0.03])
ax_pby = plt.axes([0.12, 0.10, 0.75, 0.03])
ax_tx = plt.axes([0.12, 0.05, 0.75, 0.03])
ax_ty = plt.axes([0.12, 0.00, 0.75, 0.03])

slider_theta = Slider(ax_theta, "theta (deg)", -180,
                      180, valinit=theta0, valstep=1)
slider_pbx = Slider(ax_pbx,   "p_bx",        -2.5,
                    2.5, valinit=pbx0, valstep=0.05)
slider_pby = Slider(ax_pby,   "p_by",        -2.5,
                    2.5, valinit=pby0, valstep=0.05)
slider_tx = Slider(ax_tx,    "t_x",         -2.5,
                   2.5, valinit=tx0,  valstep=0.05)
slider_ty = Slider(ax_ty,    "t_y",         -2.5,
                   2.5, valinit=ty0,  valstep=0.05)


# =========================================================
# Update plot
# =========================================================
def update(_=None):
    setup_axes(ax)

    theta = slider_theta.val
    # point coordinates in frame {b}
    pb = np.array([slider_pbx.val, slider_pby.val])
    # origin of frame {b} in frame {s}
    t = np.array([slider_tx.val, slider_ty.val])

    R = rot(theta)

    # Frame {s}
    origin_s = np.array([0.0, 0.0])
    x_s = np.array([1.0, 0.0])
    y_s = np.array([0.0, 1.0])

    # Frame {b} expressed in {s}
    x_b = R @ np.array([1.0, 0.0])
    y_b = R @ np.array([0.0, 1.0])

    # Same physical point expressed in {s}
    p_s = t + R @ pb

    # -----------------------------------------------------
    # Draw frame {s}
    # -----------------------------------------------------
    draw_arrow(ax, origin_s, axis_len * x_s,
               color="saddlebrown", label=r"$\hat{x}_s$")
    draw_arrow(ax, origin_s, axis_len * y_s,
               color="saddlebrown", label=r"$\hat{y}_s$")
    ax.text(-0.15, -0.30, r"$\{s\}$", color="saddlebrown",
            fontsize=16, fontweight="bold")

    # -----------------------------------------------------
    # Draw frame {b}
    # -----------------------------------------------------
    draw_arrow(ax, t, axis_len * x_b, color="darkgreen", label=r"$\hat{x}_b$")
    draw_arrow(ax, t, axis_len * y_b, color="darkgreen", label=r"$\hat{y}_b$")
    ax.text(t[0] + 0.05, t[1] - 0.30,
            r"$\{b\}$", color="darkgreen", fontsize=16, fontweight="bold")

    # -----------------------------------------------------
    # Draw translation vector t
    # -----------------------------------------------------
    ax.plot([0, t[0]], [0, t[1]], "--", color="dodgerblue", linewidth=3)
    ax.text(t[0] / 2 + 0.05, t[1] / 2 - 0.08, r"$t$",
            color="dodgerblue", fontsize=14, fontweight="bold")

    # -----------------------------------------------------
    # Draw point vectors
    # -----------------------------------------------------
    ax.plot([t[0], p_s[0]], [t[1], p_s[1]],
            "--", color="darkgreen", linewidth=3)
    ax.text(
        (t[0] + p_s[0]) / 2 + 0.05,
        (t[1] + p_s[1]) / 2 + 0.05,
        r"$p_b$",
        color="darkgreen",
        fontsize=14,
        fontweight="bold"
    )

    ax.plot([0, p_s[0]], [0, p_s[1]], "--", color="orangered", linewidth=3)
    ax.text(
        p_s[0] / 2 + 0.05,
        p_s[1] / 2 + 0.05,
        r"$p_s$",
        color="orangered",
        fontsize=14,
        fontweight="bold"
    )

    # -----------------------------------------------------
    # Draw point p
    # -----------------------------------------------------
    ax.scatter(p_s[0], p_s[1], s=140, color="red", edgecolor="black", zorder=5)
    ax.text(p_s[0] + 0.08, p_s[1] + 0.08, r"$p$",
            color="red", fontsize=16, fontweight="bold")

    # -----------------------------------------------------
    # Draw angle theta
    # -----------------------------------------------------
    draw_angle_arc(ax, t, theta, radius=0.55, color="red", lw=2)

    # -----------------------------------------------------
    # Text blocks
    # -----------------------------------------------------
    eq1 = (
        "Using rotation matrix R:\n"
        "p_s = t + R p_b\n\n"
        f"R = [[{R[0, 0]: .2f}, {R[0, 1]: .2f}],\n"
        f"     [{R[1, 0]: .2f}, {R[1, 1]: .2f}]]"
    )

    eq2 = (
        "Coordinates:\n"
        f"p_b = [{pb[0]: .2f}, {pb[1]: .2f}]^T\n"
        f"t   = [{t[0]: .2f}, {t[1]: .2f}]^T\n"
        f"p_s = [{p_s[0]: .2f}, {p_s[1]: .2f}]^T"
    )

    basis_text = (
        "Frame {b} axes written in frame {s}:\n"
        f"x_b = [{x_b[0]: .2f}, {x_b[1]: .2f}]^T\n"
        f"y_b = [{y_b[0]: .2f}, {y_b[1]: .2f}]^T"
    )

    ax.text(
        -3.8, 3.6, eq1,
        fontsize=13,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95)
    )

    ax.text(
        1.55, 3.6, eq2,
        fontsize=13,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95)
    )

    ax.text(
        1.55, -2.55, basis_text,
        fontsize=13,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95)
    )

    fig.canvas.draw_idle()


# =========================================================
# Connect sliders
# =========================================================
slider_theta.on_changed(update)
slider_pbx.on_changed(update)
slider_pby.on_changed(update)
slider_tx.on_changed(update)
slider_ty.on_changed(update)

update()
plt.show()
