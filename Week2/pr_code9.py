import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R


def draw_quaternion_frame(ax_plot, axis_x, axis_y, axis_z, angle_deg):
    """Draws a coordinate frame rotated by a quaternion via Axis-Angle."""
    ax_plot.clear()

    ax_plot.set_xlim([-1.5, 1.5])
    ax_plot.set_ylim([-1.5, 1.5])
    ax_plot.set_zlim([-1.5, 1.5])
    ax_plot.set_xlabel('Global X')
    ax_plot.set_ylabel('Global Y')
    ax_plot.set_zlabel('Global Z')

    # 1. Normalize the rotation axis vector
    axis = np.array([axis_x, axis_y, axis_z], dtype=float)
    norm = np.linalg.norm(axis)

    if norm == 0:
        axis = np.array([0, 0, 1])  # Default to Z if magnitude is 0
    else:
        axis = axis / norm

    # 2. Create the rotation using Axis-Angle
    angle_rad = np.radians(angle_deg)
    rot_vec = axis * angle_rad
    rot = R.from_rotvec(rot_vec)

    # 3. Extract the exact Quaternion (x, y, z, w) for display
    quat = rot.as_quat()
    qx, qy, qz, qw = quat

    # Title showing the mathematical mapping
    title = (f"Quaternion Rotation (One single spin!)\n"
             f"Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}], Angle: {angle_deg:.0f}°\n"
             f"Resulting Quaternion [x, y, z, w]:\n"
             f"[{qx:.3f},  {qy:.3f},  {qz:.3f},  {qw:.3f}]")
    ax_plot.set_title(title, pad=15, fontweight='bold')

    # Draw faint global reference axes
    ax_plot.quiver(0, 0, 0, 1.5, 0, 0, color='gray',
                   alpha=0.3, arrow_length_ratio=0.1)
    ax_plot.quiver(0, 0, 0, 0, 1.5, 0, color='gray',
                   alpha=0.3, arrow_length_ratio=0.1)
    ax_plot.quiver(0, 0, 0, 0, 0, 1.5, color='gray',
                   alpha=0.3, arrow_length_ratio=0.1)

    # DRAW THE CUSTOM ROTATION AXIS (Magenta dashed line)
    ax_plot.plot([0, axis[0]*1.5], [0, axis[1]*1.5], [0, axis[2]*1.5],
                 color='m', linestyle='--', linewidth=3, label='Rotation Axis')

    # Get local axes from the rotation matrix
    rot_matrix = rot.as_matrix()
    x_local = rot_matrix[:, 0]
    y_local = rot_matrix[:, 1]
    z_local = rot_matrix[:, 2]

    # Draw the rotated local frame
    ax_plot.quiver(0, 0, 0, x_local[0], x_local[1], x_local[2],
                   color='r', linewidth=4, arrow_length_ratio=0.15, label='Local X')
    ax_plot.quiver(0, 0, 0, y_local[0], y_local[1], y_local[2],
                   color='g', linewidth=4, arrow_length_ratio=0.15, label='Local Y')
    ax_plot.quiver(0, 0, 0, z_local[0], z_local[1], z_local[2],
                   color='b', linewidth=4, arrow_length_ratio=0.15, label='Local Z')

    ax_plot.legend(loc='upper left', fontsize='small')
    ax_plot.view_init(elev=25, azim=45)


# --- Setup Figure ---
fig = plt.figure(figsize=(10, 8))
ax_main = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(bottom=0.35)

# --- Create Sliders ---
ax_axis_x = fig.add_axes([0.15, 0.20, 0.7, 0.03])
ax_axis_y = fig.add_axes([0.15, 0.15, 0.7, 0.03])
ax_axis_z = fig.add_axes([0.15, 0.10, 0.7, 0.03])
ax_angle = fig.add_axes([0.15, 0.05, 0.7, 0.03])

# The axis sliders define the 3D vector (doesn't need to be normalized, the code handles it)
slider_x = Slider(ax_axis_x, 'Axis X', -1.0, 1.0, valinit=1.0)
slider_y = Slider(ax_axis_y, 'Axis Y', -1.0, 1.0, valinit=1.0)
slider_z = Slider(ax_axis_z, 'Axis Z', -1.0, 1.0, valinit=1.0)
slider_angle = Slider(ax_angle, 'Angle (°)', -180.0, 180.0, valinit=90.0)


def update(val):
    draw_quaternion_frame(ax_main, slider_x.val,
                          slider_y.val, slider_z.val, slider_angle.val)
    fig.canvas.draw_idle()


slider_x.on_changed(update)
slider_y.on_changed(update)
slider_z.on_changed(update)
slider_angle.on_changed(update)

# Initial draw
update(None)
plt.show()
