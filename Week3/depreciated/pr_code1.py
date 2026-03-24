import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Expert policy
#    State x should move toward 0.
#    Expert action: u = -0.8 * x
# ============================================================


def expert_policy(x):
    return -0.8 * x


# ============================================================
# 2. Environment dynamics
#    Next state = current state + action
# ============================================================
def step(x, u):
    return x + u


# ============================================================
# 3. Create expert dataset
#    Expert demonstrations start near the training region.
# ============================================================
np.random.seed(0)

num_demo_traj = 50
horizon = 15

X_train = []
U_train = []

for _ in range(num_demo_traj):
    x = np.random.uniform(-1.0, 1.0)  # expert mostly sees states near center
    for _ in range(horizon):
        u = expert_policy(x)
        X_train.append([x])
        U_train.append(u)
        x = step(x, u)

X_train = np.array(X_train)   # shape (N, 1)
U_train = np.array(U_train)   # shape (N,)

# ============================================================
# 4. Train a simple linear behavior cloning policy
#    u = w*x + b
# ============================================================
X_design = np.hstack([X_train, np.ones((len(X_train), 1))])  # [x, 1]
params = np.linalg.lstsq(X_design, U_train, rcond=None)[0]
w, b = params

print("Learned BC policy:")
print(f"u = {w:.4f} * x + {b:.4f}")

# ------------------------------------------------------------
# To simulate an imperfect learned policy, we intentionally
# add a small bias. This represents model error / dataset shift.
# Even tiny error can accumulate over time.
# ------------------------------------------------------------
bias_error = 0.08


def bc_policy(x):
    return w * x + b + bias_error


# ============================================================
# 5. Rollout expert vs BC from an initial test state
#    Start a bit outside the main training region to show how
#    off-distribution errors get worse.
# ============================================================
test_horizon = 25
x0 = 1.2

expert_states = [x0]
bc_states = [x0]

x_exp = x0
x_bc = x0

for _ in range(test_horizon):
    u_exp = expert_policy(x_exp)
    x_exp = step(x_exp, u_exp)
    expert_states.append(x_exp)

    u_bc = bc_policy(x_bc)
    x_bc = step(x_bc, u_bc)
    bc_states.append(x_bc)

expert_states = np.array(expert_states)
bc_states = np.array(bc_states)

# ============================================================
# 6. Plot trajectories
# ============================================================
t = np.arange(test_horizon + 1)

plt.figure(figsize=(8, 5))
plt.plot(t, expert_states, label="Expert rollout", linewidth=2)
plt.plot(t, bc_states, label="Behavior cloning rollout", linewidth=2)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Time step")
plt.ylabel("State x")
plt.title("Compounding Error in Behavior Cloning")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 7. Plot absolute error from expert trajectory
# ============================================================
traj_error = np.abs(bc_states - expert_states)

plt.figure(figsize=(8, 5))
plt.plot(t, traj_error, linewidth=2)
plt.xlabel("Time step")
plt.ylabel("|BC state - Expert state|")
plt.title("Error grows over time (compounding error)")
plt.grid(True)
plt.show()
