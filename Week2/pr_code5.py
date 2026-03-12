import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =========================================================
# Rotation matrix from roll-pitch-yaw
# Convention:
#   - roll  = rotation about fixed x-axis
#   - pitch = rotation about fixed y-axis
#   - yaw   = rotation about fixed z-axis
#   - Combined rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
#
# This R maps coordinates from body frame to fixed frame:
#   p_fixed = R @ p_body + t
# =========================================================


def rpy_to_rot(roll_deg, pitch_deg, yaw_deg):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    return Rz @ Ry @ Rx


# =========================================================
# Draw a coordinate frame
# origin: 3-vector
# R: 3x3 rotation matrix whose columns are the frame axes
# =========================================================
def draw_frame(ax, origin, R, axis_length=1.0, label_prefix="", alpha=1.0):
    x_axis = R[:, 0] * axis_length
    y_axis = R[:, 1] * axis_length
    z_axis = R[:, 2] * axis_length

    ax.quiver(*origin, *x_axis, color='r', linewidth=2, alpha=alpha)
    ax.quiver(*origin, *y_axis, color='g', linewidth=2, alpha=alpha)
    ax.quiver(*origin, *z_axis, color='b', linewidth=2, alpha=alpha)

    ax.text(*(origin + x_axis), f"{label_prefix}x", color='r', fontsize=10)
    ax.text(*(origin + y_axis), f"{label_prefix}y", color='g', fontsize=10)
    ax.text(*(origin + z_axis), f"{label_prefix}z", color='b', fontsize=10)


# =========================================================
# Initial values
# =========================================================
px0, py0, pz0 = 0.5, 0.3, 0.2       # point in body frame
# body frame orientation relative to fixed frame
roll0, pitch0, yaw0 = 20, 10, 30
tx0, ty0, tz0 = 1.0, 0.5, 0.3       # translation of body origin in fixed frame

# =========================================================
# Figure and 3D axis
# =========================================================
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.08, bottom=0.38)

# =========================================================
# Slider axes
# =========================================================
slider_h = 0.025
slider_gap = 0.035
start_y = 0.30

ax_px = plt.axes([0.12, start_y, 0.75, slider_h])
ax_py = plt.axes([0.12, start_y - slider_gap, 0.75, slider_h])
ax_pz = plt.axes([0.12, start_y - 2 * slider_gap, 0.75, slider_h])

ax_roll = plt.axes([0.12, start_y - 3.5 * slider_gap, 0.75, slider_h])
ax_pitch = plt.axes([0.12, start_y - 4.5 * slider_gap, 0.75, slider_h])
ax_yaw = plt.axes([0.12, start_y - 5.5 * slider_gap, 0.75, slider_h])

ax_tx = plt.axes([0.12, start_y - 7.0 * slider_gap, 0.75, slider_h])
ax_ty = plt.axes([0.12, start_y - 8.0 * slider_gap, 0.75, slider_h])
ax_tz = plt.axes([0.12, start_y - 9.0 * slider_gap, 0.75, slider_h])

# =========================================================
# Create sliders
# =========================================================
s_px = Slider(ax_px, 'p_x (body)', -2.0, 2.0, valinit=px0)
s_py = Slider(ax_py, 'p_y (body)', -2.0, 2.0, valinit=py0)
s_pz = Slider(ax_pz, 'p_z (body)', -2.0, 2.0, valinit=pz0)

s_roll = Slider(ax_roll,  'roll (deg)',  -180.0, 180.0, valinit=roll0)
s_pitch = Slider(ax_pitch, 'pitch (deg)', -180.0, 180.0, valinit=pitch0)
s_yaw = Slider(ax_yaw,   'yaw (deg)',   -180.0, 180.0, valinit=yaw0)

s_tx = Slider(ax_tx, 't_x', -3.0, 3.0, valinit=tx0)
s_ty = Slider(ax_ty, 't_y', -3.0, 3.0, valinit=ty0)
s_tz = Slider(ax_tz, 't_z', -3.0, 3.0, valinit=tz0)


# =========================================================
# Update function
# =========================================================
def update(val):
    ax.cla()

    # Read slider values
    p_body = np.array([s_px.val, s_py.val, s_pz.val])
    roll = s_roll.val
    pitch = s_pitch.val
    yaw = s_yaw.val
    t = np.array([s_tx.val, s_ty.val, s_tz.val])

    # Rotation from body frame to fixed frame
    R = rpy_to_rot(roll, pitch, yaw)

    # Point transformed into fixed frame
    p_fixed = R @ p_body + t

    # Fixed frame
    R_fixed = np.eye(3)
    origin_fixed = np.zeros(3)

    # Body frame origin in fixed coordinates
    origin_body = t

    # Draw fixed frame
    draw_frame(ax, origin_fixed, R_fixed, axis_length=1.0,
               label_prefix="F_", alpha=1.0)

    # Draw body frame
    draw_frame(ax, origin_body, R, axis_length=0.8,
               label_prefix="B_", alpha=1.0)

    # Draw point in fixed frame
    ax.scatter(*p_fixed, color='magenta', s=80, label='Point (in fixed frame)')

    # Draw point in body frame visualization:
    # start from body origin and move along body axes
    ax.plot(
        [origin_body[0], p_fixed[0]],
        [origin_body[1], p_fixed[1]],
        [origin_body[2], p_fixed[2]],
        'm--',
        linewidth=2
    )

    # Draw body origin and fixed origin
    ax.scatter(*origin_fixed, color='k', s=40)
    ax.scatter(*origin_body, color='orange', s=60, label='Body origin')

    # Text info
    info = (
        f"p_body = [{p_body[0]:.2f}, {p_body[1]:.2f}, {p_body[2]:.2f}]^T\n"
        f"t      = [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]^T\n"
        f"p_fixed = R @ p_body + t\n"
        f"        = [{p_fixed[0]:.2f}, {p_fixed[1]:.2f}, {p_fixed[2]:.2f}]^T"
    )
    ax.text2D(0.02, 0.98, info, transform=ax.transAxes, fontsize=10, va='top')

    # Axis settings
    ax.set_title("3D Coordinate Transformation: Body Frame to Fixed Frame")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc='upper right')

    # Make the axes look balanced
    lim = 3.5
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # Better viewing angle
    ax.view_init(elev=25, azim=45)

    plt.draw()


# =========================================================
# Connect sliders
# =========================================================
s_px.on_changed(update)
s_py.on_changed(update)
s_pz.on_changed(update)

s_roll.on_changed(update)
s_pitch.on_changed(update)
s_yaw.on_changed(update)

s_tx.on_changed(update)
s_ty.on_changed(update)
s_tz.on_changed(update)

# Initial draw
update(None)

plt.show()
