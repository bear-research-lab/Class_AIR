import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate data from a Gaussian distribution
# -----------------------------
np.random.seed(42)

true_mu = 5.0
true_sigma = 2.0
n_samples = 100

data = np.random.normal(loc=true_mu, scale=true_sigma, size=n_samples)

# -----------------------------
# 2. Solve MLE for Gaussian parameters
#    For a normal distribution:
#    mu_MLE = sample mean
#    sigma_MLE = sqrt((1/N) * sum((x_i - mu)^2))
# -----------------------------
mu_mle = np.mean(data)
sigma_mle = np.sqrt(np.mean((data - mu_mle) ** 2))

print(f"True mean      = {true_mu:.4f}")
print(f"True std       = {true_sigma:.4f}")
print(f"Estimated mean = {mu_mle:.4f}")
print(f"Estimated std  = {sigma_mle:.4f}")

# -----------------------------
# 3. Define Gaussian PDF using MLE parameters
# -----------------------------
x = np.linspace(min(data) - 2, max(data) + 2, 500)
pdf_mle = (1 / (sigma_mle * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x - mu_mle) / sigma_mle) ** 2
)

# Optional: true PDF for comparison
pdf_true = (1 / (true_sigma * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x - true_mu) / true_sigma) ** 2
)

# -----------------------------
# 4. Visualization
# -----------------------------
plt.figure(figsize=(8, 5))

# Histogram of sampled data (density=True makes it comparable to PDF)
plt.hist(data, bins=15, density=True, alpha=0.6, label="Sampled data histogram")

# Fitted Gaussian from MLE
plt.plot(x, pdf_mle, linewidth=2, label="MLE fitted Gaussian")

# Optional: true Gaussian
plt.plot(x, pdf_true, linestyle="--", linewidth=2, label="True Gaussian")

# Show sampled points on x-axis
plt.scatter(data, np.zeros_like(data), alpha=0.7, marker='x', label="Sampled points")

plt.title("Gaussian Samples and MLE-Fitted Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()