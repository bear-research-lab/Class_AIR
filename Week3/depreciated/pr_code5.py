import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. 데이터 생성 (100개의 점)
np.random.seed(42)
x = np.sort(np.random.rand(100))
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, 100)
x_plot = np.linspace(0, 1, 1000).reshape(-1, 1)

# 2. 두 가지 모델 설정
# 적은 파라미터 (단순한 곡선)
model_simple = make_pipeline(PolynomialFeatures(3), LinearRegression())
# 많은 파라미터 (아주 복잡하고 '각진' 곡선 - 과적합)
model_complex = make_pipeline(PolynomialFeatures(30), LinearRegression())

# 3. 학습 및 예측
model_simple.fit(x.reshape(-1, 1), y)
y_simple = model_simple.predict(x_plot)

model_complex.fit(x.reshape(-1, 1), y)
y_complex = model_complex.predict(x_plot)

# 4. 시각화
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='gray', alpha=0.5, label='100 Data Points')

plt.plot(x_plot, y_simple, color='teal', lw=3,
         label='Fewer Parameters (Smooth Trend)')
plt.plot(x_plot, y_complex, color='crimson', lw=2,
         linestyle='--', label='More Parameters (Sharp/Overfit)')

plt.title("Model Capacity vs. Function Shape", fontsize=14)
plt.ylim(-2, 2)
plt.legend()
plt.show()
