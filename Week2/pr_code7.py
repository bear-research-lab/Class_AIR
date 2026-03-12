import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# Rotation matrices
# ============================================================


def Rx(roll_deg):
    r = np.radians(roll_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)]
    ])


def Ry(pitch_deg):
    p = np.radians(pitch_deg)
    return np.array([
        [np.cos(p), 0, np.sin(p)],
        [0,         1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])


def Rz(yaw_deg):
    y = np.radians(yaw_deg)
    return np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1]
    ])


# ============================================================
# Draw a coordinate frame
# ============================================================

def draw_frame(ax, R, origin, label, axis_len=0.8, alpha=1.0, linewidth=2.5):
    origin = np.array(origin).reshape(3)

    x_axis = R @ np.array([axis_len, 0, 0])
    y_axis = R @ np.array([0, axis_len, 0])
    z_axis = R @ np.array([0, 0, axis_len])

    # x-axis (red)
    ax.quiver(
        origin[0], origin[1], origin[2],
        x_axis[0], x_axis[1], x_axis[2],
        color='r', linewidth=linewidth, alpha=alpha
    )

    # y-axis (green)
    ax.quiver(
        origin[0], origin[1], origin[2],
        y_axis[0], y_axis[1], y_axis[2],
        color='g', linewidth=linewidth, alpha=alpha
    )

    # z-axis (blue)
    ax.quiver(
        origin[0], origin[1], origin[2],
        z_axis[0], z_axis[1], z_axis[2],
        color='b', linewidth=linewidth, alpha=alpha
    )

    ax.text(origin[0], origin[1], origin[2], label, fontsize=11, weight='bold')


# ============================================================
# Update plot
# ============================================================

def update_plot(*args):
    roll = roll_var.get()
    pitch = pitch_var.get()
    yaw = yaw_var.get()

    ax.cla()

    # Original frame
    R0 = np.eye(3)

    # ------------------------------------------------------------
    # Intrinsic rotation: Yaw -> Pitch -> Roll
    #
    # Step 1: yaw about current z-axis
    # Step 2: pitch about new y-axis
    # Step 3: roll about new x-axis
    #
    # Combined matrix:
    #   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # ------------------------------------------------------------
    R_yaw = Rz(yaw)
    R_yaw_pitch = Rz(yaw) @ Ry(pitch)
    R_ypr = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    # Draw four frames at different origins for comparison
    draw_frame(ax, R0,             origin=[0.0, 0.0, 0.0], label='Original')
    draw_frame(ax, R_yaw,          origin=[2.0, 0.0, 0.0], label='Yaw only')
    draw_frame(ax, R_yaw_pitch,    origin=[4.0, 0.0, 0.0], label='Yaw+Pitch')
    draw_frame(ax, R_ypr,          origin=[
               6.0, 0.0, 0.0], label='Yaw+Pitch+Roll')

    # Global reference line
    ax.plot([0, 7], [0, 0], [0, 0], 'k--', alpha=0.25)

    # Axis settings
    ax.set_xlim(-1, 7.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f"Intrinsic Rotation (Yaw → Pitch → Roll)\n"
        f"Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, Roll={roll:.1f}°"
    )

    ax.set_box_aspect([8.5, 5, 5])
    ax.view_init(elev=25, azim=-60)
    canvas.draw()


# ============================================================
# Tkinter GUI
# ============================================================

root = tk.Tk()
root.title("3D Intrinsic Yaw-Pitch-Roll Coordinate Frame Visualizer")
root.geometry("1200x800")

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Matplotlib figure
fig = plt.Figure(figsize=(10, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')

canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Slider frame
control_frame = ttk.Frame(main_frame)
control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

roll_var = tk.DoubleVar(value=0.0)
pitch_var = tk.DoubleVar(value=0.0)
yaw_var = tk.DoubleVar(value=0.0)

# Yaw slider
ttk.Label(control_frame, text="Yaw (deg)").grid(row=0, column=0, sticky="w")
yaw_slider = tk.Scale(
    control_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
    resolution=1, variable=yaw_var, command=update_plot, length=350
)
yaw_slider.grid(row=0, column=1, padx=10, pady=5)

# Pitch slider
ttk.Label(control_frame, text="Pitch (deg)").grid(row=1, column=0, sticky="w")
pitch_slider = tk.Scale(
    control_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
    resolution=1, variable=pitch_var, command=update_plot, length=350
)
pitch_slider.grid(row=1, column=1, padx=10, pady=5)

# Roll slider
ttk.Label(control_frame, text="Roll (deg)").grid(row=2, column=0, sticky="w")
roll_slider = tk.Scale(
    control_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
    resolution=1, variable=roll_var, command=update_plot, length=350
)
roll_slider.grid(row=2, column=1, padx=10, pady=5)


def reset_values():
    yaw_var.set(0.0)
    pitch_var.set(0.0)
    roll_var.set(0.0)
    update_plot()


reset_button = ttk.Button(control_frame, text="Reset", command=reset_values)
reset_button.grid(row=0, column=2, rowspan=3, padx=20)

update_plot()
root.mainloop()
