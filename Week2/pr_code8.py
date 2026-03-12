import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R


def draw_coordinate_frame(ax, yaw, pitch, roll, case_name):
    """Draws a 3D coordinate frame and displays the Yaw - Roll math."""
    ax.clear()

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel('Global Z')

    # Calculate the math rule for +90 pitch (Intrinsic ZYX)
    math_rule_value = yaw - roll

    # Display the angles and the mathematical proof
    title = (f"{case_name}\n"
             f"Yaw={yaw:.0f}°, Pitch={pitch:.0f}°, Roll={roll:.0f}°\n"
             f"Math: (Yaw - Roll) = {math_rule_value:.0f}°")
    ax.set_title(title, pad=10)

    # Draw faint global reference axes
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='gray',
              alpha=0.3, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='gray',
              alpha=0.3, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='gray',
              alpha=0.3, arrow_length_ratio=0.1)

    # THE FIX: Uppercase 'ZYX' for Intrinsic (local) moving axes
    rot_matrix = R.from_euler(
        'ZYX', [yaw, pitch, roll], degrees=True).as_matrix()

    # Extract local axes
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]
    z_axis = rot_matrix[:, 2]

    # Draw local frame
    ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r',
              linewidth=4, arrow_length_ratio=0.15, label='Local X (Roll)')
    ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g',
              linewidth=4, arrow_length_ratio=0.15, label='Local Y (Pitch)')
    ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b',
              linewidth=4, arrow_length_ratio=0.15, label='Local Z (Yaw)')

    ax.legend(loc='upper left', fontsize='small')
    ax.view_init(elev=25, azim=45)


# --- Setup Figure ---
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# THE UI FIX: Give the subplots plenty of room away from the sliders and main title
fig.subplots_adjust(bottom=0.35, top=0.75)

fixed_pitch = 90.0
init_yaw1, init_roll1 = 30.0, 0.0
init_yaw2, init_roll2 = 60.0, 30.0

# --- Create 4 Independent Sliders ---
ax_yaw1 = fig.add_axes([0.15, 0.20, 0.3, 0.03])
ax_roll1 = fig.add_axes([0.15, 0.15, 0.3, 0.03])
ax_yaw2 = fig.add_axes([0.60, 0.20, 0.3, 0.03])
ax_roll2 = fig.add_axes([0.60, 0.15, 0.3, 0.03])

slider_yaw1 = Slider(ax_yaw1, 'Case 1 Yaw', -180.0, 180.0, valinit=init_yaw1)
slider_roll1 = Slider(ax_roll1, 'Case 1 Roll', -
                      180.0, 180.0, valinit=init_roll1)
slider_yaw2 = Slider(ax_yaw2, 'Case 2 Yaw', -180.0, 180.0, valinit=init_yaw2)
slider_roll2 = Slider(ax_roll2, 'Case 2 Roll', -
                      180.0, 180.0, valinit=init_roll2)


def update(val):
    draw_coordinate_frame(ax1, slider_yaw1.val,
                          fixed_pitch, slider_roll1.val, "Case 1")
    draw_coordinate_frame(ax2, slider_yaw2.val,
                          fixed_pitch, slider_roll2.val, "Case 2")
    fig.canvas.draw_idle()


slider_yaw1.on_changed(update)
slider_roll1.on_changed(update)
slider_yaw2.on_changed(update)
slider_roll2.on_changed(update)

update(None)

# Set the title high up so it doesn't overlap
fig.suptitle("Proving Gimbal Lock at +90° Pitch (Intrinsic Z-Y-X)\nIf (Yaw - Roll) is identical, the orientation is identical.",
             fontsize=15, fontweight='bold', y=0.95)
plt.show()
