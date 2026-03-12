import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# =========================================================
# Homogeneous transform in 2D
# =========================================================
def T2(theta_deg, tx, ty):
    th = np.radians(theta_deg)
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1]
    ])


# =========================================================
# Draw a coordinate frame
# =========================================================
def draw_frame(ax, T, name, axis_len=0.35, xcolor="crimson", ycolor="seagreen"):
    origin = T[:2, 2]
    x_axis = T[:2, :2] @ np.array([axis_len, 0.0])
    y_axis = T[:2, :2] @ np.array([0.0, axis_len])

    ax.arrow(
        origin[0], origin[1], x_axis[0], x_axis[1],
        head_width=0.05, head_length=0.08,
        fc=xcolor, ec=xcolor, linewidth=2, length_includes_head=True
    )
    ax.arrow(
        origin[0], origin[1], y_axis[0], y_axis[1],
        head_width=0.05, head_length=0.08,
        fc=ycolor, ec=ycolor, linewidth=2, length_includes_head=True
    )

    ax.text(origin[0] + 0.03, origin[1] + 0.03,
            f"{{{name}}}", fontsize=13, fontweight="bold")
    ax.text(origin[0] + x_axis[0] + 0.03, origin[1] +
            x_axis[1] + 0.03, f"x_{name}", color=xcolor, fontsize=11)
    ax.text(origin[0] + y_axis[0] + 0.03, origin[1] +
            y_axis[1] + 0.03, f"y_{name}", color=ycolor, fontsize=11)


# =========================================================
# Pretty matrix text
# =========================================================
def fmt_matrix(M, digits=2):
    return (
        f"[[{M[0, 0]: .{digits}f}, {M[0, 1]: .{digits}f}, {M[0, 2]: .{digits}f}],\n"
        f" [{M[1, 0]: .{digits}f}, {M[1, 1]: .{digits}f}, {M[1, 2]: .{digits}f}],\n"
        f" [{M[2, 0]: .{digits}f}, {M[2, 1]: .{digits}f}, {M[2, 2]: .{digits}f}]]"
    )


# =========================================================
# Initial values
# =========================================================
theta1_0 = 30.0
theta2_0 = 40.0
L1_0 = 1.2
L2_0 = 0.9


# =========================================================
# Figure and sliders
# =========================================================
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.28)

ax_theta1 = plt.axes([0.12, 0.17, 0.75, 0.03])
ax_theta2 = plt.axes([0.12, 0.12, 0.75, 0.03])
ax_L1 = plt.axes([0.12, 0.07, 0.75, 0.03])
ax_L2 = plt.axes([0.12, 0.02, 0.75, 0.03])

slider_theta1 = Slider(ax_theta1, "theta1 (deg)", -180,
                       180, valinit=theta1_0, valstep=1)
slider_theta2 = Slider(ax_theta2, "theta2 (deg)", -180,
                       180, valinit=theta2_0, valstep=1)
slider_L1 = Slider(ax_L1,     "L1",            0.4,
                   2.0, valinit=L1_0,     valstep=0.05)
slider_L2 = Slider(ax_L2,     "L2",            0.4,
                   2.0, valinit=L2_0,     valstep=0.05)


# =========================================================
# Update
# =========================================================
def update(_=None):
    ax.clear()
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.2, 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_title("2-Link Serial Chain: Homogeneous Transformation Multiplication",
                 fontsize=16, fontweight="bold")

    theta1 = slider_theta1.val
    theta2 = slider_theta2.val
    L1 = slider_L1.val
    L2 = slider_L2.val

    # Base frame
    T00 = np.eye(3)

    # Transform from frame 0 to frame 1
    # Rotate by theta1, then translate along local x by L1
    T01 = T2(theta1, 0, 0) @ T2(0, L1, 0)

    # Transform from frame 1 to frame 2
    T12 = T2(theta2, 0, 0) @ T2(0, L2, 0)

    # Total transform
    T02 = T01 @ T12

    # Origins
    p0 = T00[:2, 2]
    p1 = T01[:2, 2]
    p2 = T02[:2, 2]

    # Draw links
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=4, label="Link 1")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=4, label="Link 2")

    # Draw joints and end-effector
    ax.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], s=90, zorder=5)
    ax.text(p0[0] - 0.08, p0[1] - 0.15, "Base", fontsize=11)
    ax.text(p1[0] + 0.05, p1[1] + 0.05, "Joint 1", fontsize=11)
    ax.text(p2[0] + 0.05, p2[1] + 0.05, "End effector", fontsize=11)

    # Draw coordinate frames
    draw_frame(ax, T00, "0")
    draw_frame(ax, T01, "1")
    draw_frame(ax, T02, "2")

    # Dashed helper lines
    ax.plot([0, p2[0]], [0, p2[1]], "--", linewidth=2, alpha=0.6)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", linewidth=2, alpha=0.6)

    # End-effector coordinates
    ee_text = (
        f"End-effector position:\n"
        f"p_2 = [{p2[0]: .2f}, {p2[1]: .2f}]^T"
    )

    # Matrix texts
    txt1 = (
        f"T01 =\n{fmt_matrix(T01)}"
    )
    txt2 = (
        f"T12 =\n{fmt_matrix(T12)}"
    )
    txt3 = (
        f"T02 = T01 @ T12 =\n{fmt_matrix(T02)}"
    )
    txt4 = (
        f"theta1 = {theta1:.1f} deg\n"
        f"theta2 = {theta2:.1f} deg\n"
        f"L1 = {L1:.2f}\n"
        f"L2 = {L2:.2f}\n\n"
        f"{ee_text}"
    )

    ax.text(-2.7, 2.2, txt1, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95))
    ax.text(-2.7, 0.45, txt2, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95))
    ax.text(1.05, 2.2, txt3, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95))
    ax.text(1.55, -0.7, txt4, fontsize=12, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95))

    # Main formula
    formula = (
        "Serial chain:\n"
        "T02 = T01 T12\n"
        "p_world = T02 [0, 0, 1]^T"
    )
    ax.text(-0.3, -1.95, formula, fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95))

    ax.legend(loc="upper center")
    fig.canvas.draw_idle()


slider_theta1.on_changed(update)
slider_theta2.on_changed(update)
slider_L1.on_changed(update)
slider_L2.on_changed(update)

update()
plt.show()
