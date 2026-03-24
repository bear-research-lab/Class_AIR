import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. 데이터 생성: 복잡한 '해안선' (Sine wave 기반)
np.random.seed(42)
n_samples = 30
x = np.sort(np.random.rand(n_samples))
y = np.sin(1.5 * np.pi * x) + np.random.normal(0, 0.1, n_samples)
x_plot = np.linspace(0, 1, 500).reshape(-1, 1)

# 2. 세 가지 다른 용량(Degree)의 모델 정의
degrees = [1, 4, 15]  # 1: 직선(Under), 4: 적절(Best), 15: 각진 형태(Over)
titles = ['Underfitting (Straight Fence)',
          'Balanced (Smooth Manifold)', 'Overfitting (Sharp/Pointy)']
colors = ['orange', 'teal', 'crimson']

plt.figure(figsize=(18, 5))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)

    # 다항 회귀 모델 생성 및 학습
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x_plot)

    # 시각화
    ax.scatter(x, y, color='black', s=40, label='Cat Data (Coastline)')
    ax.plot(x_plot, y_pred, color=colors[i],
            lw=3, label=f'Model (Capacity: {degree})')

    # '바다' 영역 비유를 위한 배경 색칠 (Over-generalization 시각화)
    # 모델 주위로 '확률 밀도'가 퍼져 있다고 가정하고 채우기
    ax.fill_between(x_plot.flatten(), y_pred - 0.2,
                    y_pred + 0.2, color=colors[i], alpha=0.1)

    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower left')

    if i == 0:
        ax.annotate('Over-generalization:\nFences include too much "Sea"',
                    xy=(0.5, 0), xytext=(0.1, 0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    if i == 2:
        ax.annotate('Memorization:\nSharp/Pointy spikes',
                    xy=(0.85, 0.5), xytext=(0.5, 1.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
plt.show()
