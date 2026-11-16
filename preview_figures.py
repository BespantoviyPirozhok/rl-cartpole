"""
preview_figures.py
Синтетические кривые для предварительной вёрстки.
⚠️ Заменить на реальные результаты после train_rl.py.

Запуск: python preview_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})
os.makedirs("results", exist_ok=True)


def smooth(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def sigmoid_curve(n, start, end, center, k, noise, seed):
    rng = np.random.default_rng(seed)
    s = 1 / (1 + np.exp(-k * (np.arange(n) - center)))
    base = start + (end - start) * s
    return base + rng.normal(0, noise, n)


# ---------- CartPole: DQN ≈ 450, PPO ≈ 500 ----------
n_cp = 1000
ppo_cp = sigmoid_curve(n_cp, 20, 500, 300, 0.020, 18, 1)
dqn_cp = sigmoid_curve(n_cp, 20, 460, 460, 0.012, 55, 2)
dqn_cp += np.sin(np.arange(n_cp) / 25) * 15  # добавляем «нестабильность» DQN

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(len(smooth(ppo_cp))), smooth(ppo_cp), label="PPO", color="tab:orange")
ax.plot(np.arange(len(smooth(dqn_cp))), smooth(dqn_cp), label="DQN", color="tab:blue")
ax.axhline(475, ls="--", color="gray", lw=1, label="Порог сходимости = 475")
ax.set_xlabel("Эпизод"); ax.set_ylabel("Средняя награда за эпизод")
ax.set_title("CartPole-v1: средняя награда (скользящее среднее, окно = 100)")
ax.set_ylim(0, 520); ax.legend(loc="lower right")
fig.savefig("results/fig_cartpole.pdf"); plt.close(fig)

# ---------- FrozenLake: DQN ≈ 0.78, PPO ≈ 0.68 ----------
n_fl = 800
dqn_fl = sigmoid_curve(n_fl, 0.02, 0.78, 300, 0.024, 0.10, 3)
ppo_fl = sigmoid_curve(n_fl, 0.02, 0.68, 430, 0.018, 0.08, 4)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(len(smooth(dqn_fl))), smooth(dqn_fl), label="DQN", color="tab:blue")
ax.plot(np.arange(len(smooth(ppo_fl))), smooth(ppo_fl), label="PPO", color="tab:orange")
ax.axhline(0.8, ls="--", color="gray", lw=1, label="Порог сходимости = 0.8")
ax.set_xlabel("Эпизод"); ax.set_ylabel("Доля успешных эпизодов")
ax.set_title("FrozenLake-v1 (8×8): доля успешных эпизодов (окно = 100)")
ax.set_ylim(0, 1.0); ax.legend(loc="lower right")
fig.savefig("results/fig_frozenlake.pdf"); plt.close(fig)

print("✓ results/fig_cartpole.pdf")
print("✓ results/fig_frozenlake.pdf")