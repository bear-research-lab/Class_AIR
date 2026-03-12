import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------
# 1. Forward kinematics function
# ----------------------------
def forward_kinematics_2link(theta1_deg, theta2_deg, L1=1.0, L2=0.8):
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)

    # Base
    p0 = np.array([0.0, 0.0])

    # End of link 1
    p1 = np.array([
        L1 * np.cos(theta1),
        L1 * np.sin(theta1)
    ])

    # End of link 2
    p2 = p1 + np.array([
        L2 * np.cos(theta1 + theta2),
        L2 * np.sin(theta1 + theta2)
    ])

    return p0, p1, p2


# ----------------------------
# 2. GUI Application
# ----------------------------
class FKDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2-Link Planar Robot Forward Kinematics Demo")
        self.root.geometry("1000x700")

        # Default values
        self.theta1_var = tk.DoubleVar(value=30.0)
        self.theta2_var = tk.DoubleVar(value=45.0)
        self.L1_var = tk.DoubleVar(value=1.0)
        self.L2_var = tk.DoubleVar(value=0.8)

        # Main layout
        self.control_frame = ttk.Frame(root, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.plot_frame = ttk.Frame(root, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_controls()
        self.build_plot()
        self.update_plot()

    def build_controls(self):
        ttk.Label(
            self.control_frame,
            text="Forward Kinematics Demo",
            font=("Arial", 16, "bold")
        ).pack(pady=(0, 15))

        # theta1
        ttk.Label(self.control_frame, text="theta1 (deg)").pack(anchor="w")
        theta1_scale = tk.Scale(
            self.control_frame,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.theta1_var,
            length=250,
            command=lambda _: self.update_plot()
        )
        theta1_scale.pack(pady=(0, 10))

        # theta2
        ttk.Label(self.control_frame, text="theta2 (deg)").pack(anchor="w")
        theta2_scale = tk.Scale(
            self.control_frame,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            resolution=1,
            variable=self.theta2_var,
            length=250,
            command=lambda _: self.update_plot()
        )
        theta2_scale.pack(pady=(0, 10))

        # L1
        ttk.Label(self.control_frame, text="L1").pack(anchor="w")
        l1_scale = tk.Scale(
            self.control_frame,
            from_=0.5,
            to=2.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            variable=self.L1_var,
            length=250,
            command=lambda _: self.update_plot()
        )
        l1_scale.pack(pady=(0, 10))

        # L2
        ttk.Label(self.control_frame, text="L2").pack(anchor="w")
        l2_scale = tk.Scale(
            self.control_frame,
            from_=0.5,
            to=2.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            variable=self.L2_var,
            length=250,
            command=lambda _: self.update_plot()
        )
        l2_scale.pack(pady=(0, 10))

        # Text output
        ttk.Label(
            self.control_frame,
            text="Numerical Result",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", pady=(20, 5))

        self.result_label = ttk.Label(
            self.control_frame,
            text="",
            justify="left",
            font=("Courier", 10)
        )
        self.result_label.pack(anchor="w", pady=(0, 10))

        ttk.Label(
            self.control_frame,
            text="Forward Kinematics Equations",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", pady=(10, 5))

        eq_text = (
            "x = L1*cos(theta1) + L2*cos(theta1 + theta2)\n"
            "y = L1*sin(theta1) + L2*sin(theta1 + theta2)"
        )
        ttk.Label(
            self.control_frame,
            text=eq_text,
            justify="left",
            font=("Courier", 10)
        ).pack(anchor="w")

    def build_plot(self):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        theta1_deg = self.theta1_var.get()
        theta2_deg = self.theta2_var.get()
        L1 = self.L1_var.get()
        L2 = self.L2_var.get()

        p0, p1, p2 = forward_kinematics_2link(theta1_deg, theta2_deg, L1, L2)

        self.ax.clear()

        # Plot links
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                     linewidth=4, label="Link 1")
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     linewidth=4, label="Link 2")

        # Plot joints
        self.ax.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], s=80)

        # Labels
        self.ax.text(p0[0], p0[1], "  Base", fontsize=10)
        self.ax.text(p1[0], p1[1], "  Joint 2", fontsize=10)
        self.ax.text(p2[0], p2[1], "  End Effector", fontsize=10)

        # Axes
        self.ax.axhline(0, linewidth=1)
        self.ax.axvline(0, linewidth=1)

        # Figure settings
        max_range = L1 + L2 + 0.3
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)
        self.ax.set_title("2-Link Planar Robot: Forward Kinematics")
        self.ax.legend()

        # Numeric output
        theta1 = np.radians(theta1_deg)
        theta2 = np.radians(theta2_deg)
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

        result_text = (
            f"theta1 = {theta1_deg:.1f} deg\n"
            f"theta2 = {theta2_deg:.1f} deg\n"
            f"L1     = {L1:.1f}\n"
            f"L2     = {L2:.1f}\n\n"
            f"End-effector:\n"
            f"x = {x:.3f}\n"
            f"y = {y:.3f}"
        )
        self.result_label.config(text=result_text)

        self.canvas.draw()


# ----------------------------
# 3. Run app
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FKDemoApp(root)
    root.mainloop()
