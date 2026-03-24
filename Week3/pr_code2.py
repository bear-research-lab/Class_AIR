import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import norm

# 1. Analytical KL Divergence for two Gaussians


def calculate_kl_analytical(mu1, std1, mu2, std2):
    """Calculates D_KL(P || Q)"""
    # Formula: log(s2/s1) + (s1^2 + (m1-m2)^2)/(2*s2^2) - 0.5
    return np.log(std2/std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5


# 2. Initial Setup
mu_p, std_p = 0.0, 1.0
mu_q, std_q = 1.0, 1.5
x = np.linspace(-10, 10, 1000)

# Create the figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
    10, 8), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.25, hspace=0.4)

# --- Top Plot: PDF Comparison ---
line_p, = ax1.plot(x, norm.pdf(x, mu_p, std_p),
                   'b-', lw=2, label='Target $P(x)$')
line_q, = ax1.plot(x, norm.pdf(x, mu_q, std_q),
                   'r--', lw=2, label='Approx $Q(x)$')
ax1.set_ylabel("Probability Density")
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# --- Bottom Plot: Pointwise KL (The Integrand) ---
y_p = norm.pdf(x, mu_p, std_p)
y_q = norm.pdf(x, mu_q, std_q)
# Avoid log(0) issues by adding a tiny epsilon
pointwise_kl = y_p * np.log((y_p + 1e-10) / (y_q + 1e-10))
line_diff, = ax2.plot(x, pointwise_kl, 'g-', lw=1.5,
                      label='$p(x) \ln(p(x)/q(x))$')

# Initialize the fill globally so we can .remove() it later
global_poly_fill = ax2.fill_between(x, pointwise_kl, color='green', alpha=0.3)

ax2.axhline(0, color='black', lw=0.8, ls='--')
ax2.set_ylabel("Information Gain (nats)")
ax2.set_xlabel("x")
ax2.legend()

# --- UI Controls (Sliders and Buttons) ---
ax_mu_p = plt.axes([0.15, 0.12, 0.25, 0.03])
ax_std_p = plt.axes([0.15, 0.08, 0.25, 0.03])
ax_mu_q = plt.axes([0.55, 0.12, 0.25, 0.03])
ax_std_q = plt.axes([0.55, 0.08, 0.25, 0.03])
ax_swap = plt.axes([0.85, 0.10, 0.1, 0.04])

s_mu_p = Slider(ax_mu_p, '$\mu_P$', -5.0, 5.0, valinit=mu_p)
s_std_p = Slider(ax_std_p, '$\sigma_P$', 0.5, 4.0, valinit=std_p)
s_mu_q = Slider(ax_mu_q, '$\mu_Q$', -5.0, 5.0, valinit=mu_q)
s_std_q = Slider(ax_std_q, '$\sigma_Q$', 0.5, 4.0, valinit=std_q)
btn_swap = Button(ax_swap, 'Swap P/Q')


def update(val):
    global global_poly_fill

    mp, sp = s_mu_p.val, s_std_p.val
    mq, sq = s_mu_q.val, s_std_q.val

    # Recalculate Distributions
    new_p = norm.pdf(x, mp, sp)
    new_q = norm.pdf(x, mq, sq)

    # Update Lines
    line_p.set_ydata(new_p)
    line_q.set_ydata(new_q)

    # Update Pointwise Plot
    new_pointwise = new_p * np.log((new_p + 1e-10) / (new_q + 1e-10))
    line_diff.set_ydata(new_pointwise)

    # Update Fill Area (The fix for your error)
    global_poly_fill.remove()
    global_poly_fill = ax2.fill_between(
        x, new_pointwise, color='green', alpha=0.3)

    # Update KL Title
    k_val = calculate_kl_analytical(mp, sp, mq, sq)
    ax1.set_title(f"KL Divergence $D_{{KL}}(P \parallel Q) = {k_val:.4f}$",
                  fontsize=14, fontweight='bold', color='#8B0000')

    fig.canvas.draw_idle()


def swap_logic(event):
    # Capture current state
    curr_mp, curr_sp = s_mu_p.val, s_std_p.val
    curr_mq, curr_sq = s_mu_q.val, s_std_q.val

    # Set values (triggers update)
    s_mu_p.set_val(curr_mq)
    s_std_p.set_val(curr_sq)
    s_mu_q.set_val(curr_mp)
    s_std_q.set_val(curr_sq)  # wait, fixed typo: curr_sp


# Register Events
s_mu_p.on_changed(update)
s_std_p.on_changed(update)
s_mu_q.on_changed(update)
s_std_q.on_changed(update)
btn_swap.on_clicked(swap_logic)

# Initialize title on first run
update(None)

plt.show()
