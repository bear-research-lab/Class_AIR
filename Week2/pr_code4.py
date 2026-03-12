# import numpy as np 
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# import CheckButtons  # ========================================================= # Fixed body dimensions # ========================================================= TORSO = 1.2 UPPER_ARM = 0.55 FOREARM = 0.50 THIGH = 0.75 SHIN = 0.75 SHOULDER_WIDTH = 0.45 HIP_WIDTH = 0.30 HEAD_RADIUS = 0.18 # ========================================================= # 2D homogeneous transformation matrix # ========================================================= def T2D(theta_deg, tx=0.0, ty=0.0): theta = np.radians(theta_deg) c, s = np.cos(theta), np.sin(theta) return np.array([ [c, -s, tx], [s, c, ty], [0, 0, 1] ]) def get_position(T): return T[:2, 2] def draw_frame(ax, T, name="", axis_len=0.08): origin = T[:2, 2] x_axis = T[:2, :2] @ np.array([axis_len, 0.0]) y_axis = T[:2, :2] @ np.array([0.0, axis_len]) ax.arrow( origin[0], origin[1], x_axis[0], x_axis[1], head_width=0.02, length_includes_head=True ) ax.arrow( origin[0], origin[1], y_axis[0], y_axis[1], head_width=0.02, length_includes_head=True ) if name: ax.text(origin[0] + 0.02, origin[1] + 0.02, name, fontsize=9) # ========================================================= # Forward kinematics for one 2-link limb # ========================================================= def fk_two_link(base_T, theta1_deg, theta2_deg, L1, L2): """ base_T : base transform of shoulder/hip theta1_deg : first joint angle theta2_deg : second joint angle L1, L2 : link lengths """ T_joint1 = base_T @ T2D(theta1_deg, 0, 0) T_link1_end = T_joint1 @ T2D(0, L1, 0) T_joint2 = T_link1_end @ T2D(theta2_deg, 0, 0) T_link2_end = T_joint2 @ T2D(0, L2, 0) return { "base_T": base_T, "joint1_T": T_joint1, "joint2_T": T_joint2, "p0": get_position(base_T), "p1": get_position(T_link1_end), "p2": get_position(T_link2_end) } # ========================================================= # Main drawing function # ========================================================= def draw_humanoid( ax, torso_angle=90, left_shoulder=40, left_elbow=30, right_shoulder=-40, right_elbow=-30, left_hip=-110, left_knee=25, right_hip=-70, right_knee=-25, show_frames=False ): ax.clear() # ----------------------------------------------------- # Root / pelvis # ----------------------------------------------------- T_root = np.eye(3) # Torso T_torso = T_root @ T2D(torso_angle, 0, 0) T_neck = T_torso @ T2D(0, TORSO, 0) pelvis = get_position(T_root) neck = get_position(T_neck) # ----------------------------------------------------- # Correct left-right offsets: # left/right separation should be in x, not y # ----------------------------------------------------- T_left_shoulder_base = T_neck @ T2D(0, -SHOULDER_WIDTH / 2, 0) T_right_shoulder_base = T_neck @ T2D(0, SHOULDER_WIDTH / 2, 0) T_left_hip_base = T_root @ T2D(0, -HIP_WIDTH / 2, 0) T_right_hip_base = T_root @ T2D(0, HIP_WIDTH / 2, 0) # ----------------------------------------------------- # Arms # ----------------------------------------------------- left_arm = fk_two_link( T_left_shoulder_base, left_shoulder, left_elbow, UPPER_ARM, FOREARM ) right_arm = fk_two_link( T_right_shoulder_base, right_shoulder, right_elbow, UPPER_ARM, FOREARM ) # ----------------------------------------------------- # Legs # ----------------------------------------------------- left_leg = fk_two_link( T_left_hip_base, left_hip, left_knee, THIGH, SHIN ) right_leg = fk_two_link( T_right_hip_base, right_hip, right_knee, THIGH, SHIN ) # ----------------------------------------------------- # Draw torso # ----------------------------------------------------- ax.plot( [pelvis[0], neck[0]], [pelvis[1], neck[1]], linewidth=5 ) # Shoulder line p_ls = get_position(T_left_shoulder_base) p_rs = get_position(T_right_shoulder_base) ax.plot( [p_ls[0], p_rs[0]], [p_ls[1], p_rs[1]], linewidth=3 ) # Hip line p_lh = get_position(T_left_hip_base) p_rh = get_position(T_right_hip_base) ax.plot( [p_lh[0], p_rh[0]], [p_lh[1], p_rh[1]], linewidth=3 ) # ----------------------------------------------------- # Draw head # ----------------------------------------------------- head_center = neck + np.array([0, HEAD_RADIUS * 1.6]) head_circle = plt.Circle( head_center, HEAD_RADIUS, fill=False, linewidth=3 ) ax.add_patch(head_circle) # ----------------------------------------------------- # Draw limbs # ----------------------------------------------------- def draw_limb(limb): p0, p1, p2 = limb["p0"], limb["p1"], limb["p2"] ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=4) ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=4) ax.scatter( [p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], s=50 ) draw_limb(left_arm) draw_limb(right_arm) draw_limb(left_leg) draw_limb(right_leg) # ----------------------------------------------------- # Optional coordinate frames # ----------------------------------------------------- if show_frames: draw_frame(ax, T_root, "Root") draw_frame(ax, T_torso, "Torso") draw_frame(ax, T_neck, "Neck") draw_frame(ax, left_arm["base_T"], "L arm base") draw_frame(ax, left_arm["joint1_T"], "L shoulder") draw_frame(ax, left_arm["joint2_T"], "L elbow") draw_frame(ax, right_arm["base_T"], "R arm base") draw_frame(ax, right_arm["joint1_T"], "R shoulder") draw_frame(ax, right_arm["joint2_T"], "R elbow") draw_frame(ax, left_leg["base_T"], "L leg base") draw_frame(ax, left_leg["joint1_T"], "L hip") draw_frame(ax, left_leg["joint2_T"], "L knee") draw_frame(ax, right_leg["base_T"], "R leg base") draw_frame(ax, right_leg["joint1_T"], "R hip") draw_frame(ax, right_leg["joint2_T"], "R knee") # ----------------------------------------------------- # Labels # ----------------------------------------------------- ax.text(pelvis[0] + 0.03, pelvis[1], "Pelvis", fontsize=10) ax.text(neck[0] + 0.03, neck[1], "Neck", fontsize=10) # ----------------------------------------------------- # Plot setup # ----------------------------------------------------- ax.set_aspect("equal") ax.grid(True) ax.set_xlim(-2.0, 2.0) ax.set_ylim(-1.8, 2.4) ax.set_title("2D Human-Shaped Robot Using Forward Kinematics") def main(): fig, ax = plt.subplots(figsize=(7, 9)) plt.subplots_adjust(left=0.28, bottom=0.42) # Initial values init_vals = { "torso_angle": 90, "left_shoulder": 40, "left_elbow": 30, "right_shoulder": -40, "right_elbow": -30, "left_hip": -110, "left_knee": 25, "right_hip": -70, "right_knee": -25, } # Initial draw draw_humanoid(ax, **init_vals, show_frames=False) # ===================================================== # Sliders # ===================================================== slider_h = 0.025 slider_gap = 0.035 x0 = 0.28 w = 0.62 y_top = 0.33 ax_torso = plt.axes([x0, y_top - 0 * slider_gap, w, slider_h]) ax_ls = plt.axes([x0, y_top - 1 * slider_gap, w, slider_h]) ax_le = plt.axes([x0, y_top - 2 * slider_gap, w, slider_h]) ax_rs = plt.axes([x0, y_top - 3 * slider_gap, w, slider_h]) ax_re = plt.axes([x0, y_top - 4 * slider_gap, w, slider_h]) ax_lh = plt.axes([x0, y_top - 5 * slider_gap, w, slider_h]) ax_lk = plt.axes([x0, y_top - 6 * slider_gap, w, slider_h]) ax_rh = plt.axes([x0, y_top - 7 * slider_gap, w, slider_h]) ax_rk = plt.axes([x0, y_top - 8 * slider_gap, w, slider_h]) s_torso = Slider(ax_torso, "torso", 60, 120, valinit=init_vals["torso_angle"], valstep=1) s_ls = Slider(ax_ls, "L shoulder", -180, 180, valinit=init_vals["left_shoulder"], valstep=1) s_le = Slider(ax_le, "L elbow", -180, 180, valinit=init_vals["left_elbow"], valstep=1) s_rs = Slider(ax_rs, "R shoulder", -180, 180, valinit=init_vals["right_shoulder"], valstep=1) s_re = Slider(ax_re, "R elbow", -180, 180, valinit=init_vals["right_elbow"], valstep=1) s_lh = Slider(ax_lh, "L hip", -180, 0, valinit=init_vals["left_hip"], valstep=1) s_lk = Slider(ax_lk, "L knee", -180, 180, valinit=init_vals["left_knee"], valstep=1) s_rh = Slider(ax_rh, "R hip", -180, 0, valinit=init_vals["right_hip"], valstep=1) s_rk = Slider(ax_rk, "R knee", -180, 180, valinit=init_vals["right_knee"], valstep=1) # Check button for show_frames check_ax = plt.axes([0.05, 0.05, 0.18, 0.08]) check = CheckButtons(check_ax, ["Show frames"], [False]) def update(_): show_frames = check.get_status()[0] draw_humanoid( ax, torso_angle=s_torso.val, left_shoulder=s_ls.val, left_elbow=s_le.val, right_shoulder=s_rs.val, right_elbow=s_re.val, left_hip=s_lh.val, left_knee=s_lk.val, right_hip=s_rh.val, right_knee=s_rk.val, show_frames=show_frames ) fig.canvas.draw_idle() s_torso.on_changed(update) s_ls.on_changed(update) s_le.on_changed(update) s_rs.on_changed(update) s_re.on_changed(update) s_lh.on_changed(update) s_lk.on_changed(update) s_rh.on_changed(update) s_rk.on_changed(update) check.on_clicked(update) print("Forward kinematics idea:") print("- Link lengths are fixed.") print("- Only joint angles change.") print("- Each arm and leg is a kinematic chain.") print("- The final hand/foot position is computed by multiplying transforms along the chain.") plt.show() if __name__ == "__main__": main()import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, CheckButtons

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# =========================================================
# Fixed body dimensions
# =========================================================
TORSO = 1.2
UPPER_ARM = 0.55
FOREARM = 0.50
THIGH = 0.75
SHIN = 0.75

SHOULDER_WIDTH = 0.45
HIP_WIDTH = 0.30
HEAD_RADIUS = 0.18


# =========================================================
# 2D homogeneous transformation matrix
# =========================================================
def T2D(theta_deg, tx=0.0, ty=0.0):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1]
    ])


def get_position(T):
    return T[:2, 2]


def draw_frame(ax, T, name="", axis_len=0.08):
    origin = T[:2, 2]
    x_axis = T[:2, :2] @ np.array([axis_len, 0.0])
    y_axis = T[:2, :2] @ np.array([0.0, axis_len])

    ax.arrow(origin[0], origin[1], x_axis[0], x_axis[1],
             head_width=0.02, length_includes_head=True)
    ax.arrow(origin[0], origin[1], y_axis[0], y_axis[1],
             head_width=0.02, length_includes_head=True)

    if name:
        ax.text(origin[0] + 0.02, origin[1] + 0.02, name, fontsize=9)


# =========================================================
# Forward kinematics for one 2-link limb
# =========================================================
def fk_two_link(base_T, theta1_deg, theta2_deg, L1, L2):

    T_joint1 = base_T @ T2D(theta1_deg, 0, 0)
    T_link1_end = T_joint1 @ T2D(0, L1, 0)

    T_joint2 = T_link1_end @ T2D(theta2_deg, 0, 0)
    T_link2_end = T_joint2 @ T2D(0, L2, 0)

    return {
        "base_T": base_T,
        "joint1_T": T_joint1,
        "joint2_T": T_joint2,
        "p0": get_position(base_T),
        "p1": get_position(T_link1_end),
        "p2": get_position(T_link2_end)
    }


# =========================================================
# Draw humanoid
# =========================================================
def draw_humanoid(ax,
                  torso_angle=90,
                  left_shoulder=40,
                  left_elbow=30,
                  right_shoulder=-40,
                  right_elbow=-30,
                  left_hip=-110,
                  left_knee=25,
                  right_hip=-70,
                  right_knee=-25,
                  show_frames=False):

    ax.clear()

    T_root = np.eye(3)

    T_torso = T_root @ T2D(torso_angle)
    T_neck = T_torso @ T2D(0, TORSO, 0)

    pelvis = get_position(T_root)
    neck = get_position(T_neck)

    # Correct left-right placement
    T_left_shoulder_base = T_neck @ T2D(0, -SHOULDER_WIDTH/2, 0)
    T_right_shoulder_base = T_left_shoulder_base

    T_left_hip_base = T_root @ T2D(0, -HIP_WIDTH/2, 0)
    T_right_hip_base = T_root @ T2D(0, HIP_WIDTH/2, 0)

    left_arm = fk_two_link(T_left_shoulder_base, left_shoulder, left_elbow,
                           UPPER_ARM, FOREARM)

    right_arm = fk_two_link(T_right_shoulder_base, right_shoulder, right_elbow,
                            UPPER_ARM, FOREARM)

    left_leg = fk_two_link(T_left_hip_base, left_hip, left_knee,
                           THIGH, SHIN)

    right_leg = fk_two_link(T_right_hip_base, right_hip, right_knee,
                            THIGH, SHIN)

    ax.plot([pelvis[0], neck[0]], [pelvis[1], neck[1]], linewidth=5)

    p_ls = get_position(T_left_shoulder_base)
    p_rs = get_position(T_right_shoulder_base)
    ax.plot([p_ls[0], p_rs[0]], [p_ls[1], p_rs[1]], linewidth=3)

    p_lh = get_position(T_left_hip_base)
    p_rh = get_position(T_right_hip_base)
    ax.plot([p_lh[0], p_rh[0]], [p_lh[1], p_rh[1]], linewidth=3)

    head_center = neck + np.array([0, HEAD_RADIUS * 1.6])
    circle = plt.Circle(head_center, HEAD_RADIUS, fill=False, linewidth=3)
    ax.add_patch(circle)

    def draw_limb(limb):
        p0, p1, p2 = limb["p0"], limb["p1"], limb["p2"]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=4)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=4)
        ax.scatter([p0[0], p1[0], p2[0]],
                   [p0[1], p1[1], p2[1]], s=50)

    draw_limb(left_arm)
    draw_limb(right_arm)
    draw_limb(left_leg)
    draw_limb(right_leg)

    if show_frames:
        draw_frame(ax, T_root, "Root")
        draw_frame(ax, T_torso, "Torso")
        draw_frame(ax, T_neck, "Neck")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.8, 2.4)
    ax.set_title("2D Humanoid Forward Kinematics")


# =========================================================
# Main GUI
# =========================================================
def main():

    fig, ax = plt.subplots(figsize=(7, 9))
    plt.subplots_adjust(bottom=0.35)

    params = {
        "torso_angle": 90,
        "left_shoulder": 40,
        "left_elbow": 30,
        "right_shoulder": -40,
        "right_elbow": -30,
        "left_hip": -110,
        "left_knee": 25,
        "right_hip": -70,
        "right_knee": -25
    }

    draw_humanoid(ax, **params)

    sliders = {}

    slider_names = [
        ("torso_angle", 60, 120),
        ("left_shoulder", -180, 180),
        ("left_elbow", -180, 180),
        ("right_shoulder", -180, 180),
        ("right_elbow", -180, 180),
        ("left_hip", -180, 0),
        ("left_knee", -180, 180),
        ("right_hip", -180, 0),
        ("right_knee", -180, 180)
    ]

    for i, (name, minv, maxv) in enumerate(slider_names):
        ax_slider = plt.axes([0.25, 0.30-i*0.03, 0.65, 0.02])
        sliders[name] = Slider(ax_slider, name, minv,
                               maxv, valinit=params[name])

    check_ax = plt.axes([0.02, 0.05, 0.15, 0.1])
    check = CheckButtons(check_ax, ["show_frames"], [False])

    def update(val):

        draw_humanoid(
            ax,
            torso_angle=sliders["torso_angle"].val,
            left_shoulder=sliders["left_shoulder"].val,
            left_elbow=sliders["left_elbow"].val,
            right_shoulder=sliders["right_shoulder"].val,
            right_elbow=sliders["right_elbow"].val,
            left_hip=sliders["left_hip"].val,
            left_knee=sliders["left_knee"].val,
            right_hip=sliders["right_hip"].val,
            right_knee=sliders["right_knee"].val,
            show_frames=check.get_status()[0]
        )

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    check.on_clicked(update)

    plt.show()


if __name__ == "__main__":
    main()
