"""
Обучение DQN и PPO в CartPole-v1 и FrozenLake-v1.
"""

import os
import csv
import time
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ---------- стиль графиков ----------
rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman"],
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
})

OUT = "results"
os.makedirs(OUT, exist_ok=True)
SEED = 42


# ---------- логгер эпизодов ----------
class EpisodeLogger(BaseCallback):
    """Собирает награды и длины всех эпизодов во время обучения."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards, self.ep_lengths = [], []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(float(info["episode"]["r"]))
                self.ep_lengths.append(float(info["episode"]["l"]))
        return True

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "length"])
            for i, (r, l) in enumerate(zip(self.ep_rewards, self.ep_lengths), 1):
                w.writerow([i, r, l])


# ---------- фабрики ----------
def make_env(env_id, seed, **env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return Monitor(env)


def train_dqn(env_id, timesteps, seed=SEED, env_kwargs=None, **overrides):
    env = make_env(env_id, seed, **(env_kwargs or {}))
    kwargs = dict(
        learning_rate=3e-4,
        batch_size=64,
        buffer_size=10_000,
        learning_starts=1_000,
        target_update_interval=100,
        train_freq=4,
        gradient_steps=1,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=seed,
        verbose=0,
        device="auto",
    )
    kwargs.update(overrides)
    model = DQN("MlpPolicy", env, **kwargs)
    logger = EpisodeLogger()
    model.learn(total_timesteps=timesteps, callback=logger)
    return logger


def train_ppo(env_id, timesteps, seed=SEED, env_kwargs=None, **overrides):
    env = make_env(env_id, seed, **(env_kwargs or {}))
    kwargs = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        seed=seed,
        verbose=0,
        device="auto",
    )
    kwargs.update(overrides)
    model = PPO("MlpPolicy", env, **kwargs)
    logger = EpisodeLogger()
    model.learn(total_timesteps=timesteps, callback=logger)
    return logger


# ---------- построение графиков ----------
def smooth(x, w=100):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_curves(curves, title, ylabel, out_pdf, ylim=None, threshold=None, w=100):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    for label, rewards, color in curves:
        y = smooth(rewards, w)
        x = np.arange(len(y)) + w // 2
        ax.plot(x, y, label=label, color=color, lw=1.2)

    if threshold is not None:
        ax.axhline(threshold, color="gray", ls="--", lw=0.8,
                   label=f"Порог $h = {threshold}$")

    ax.set_xlabel("Эпизод")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout(pad=0.2)         # ← вместо bbox_inches='tight'
    fig.savefig(out_pdf)              # ← без bbox_inches: сохраняем масштаб
    plt.close(fig)
    print(f"  ✓ сохранено: {out_pdf}")


# ---------- main ----------
if __name__ == "__main__":
    t0 = time.time()

    print("→ DQN / CartPole-v1 ...")
    dqn_cp = train_dqn("CartPole-v1", timesteps=100_000)
    dqn_cp.save(f"{OUT}/dqn_cartpole.csv")
    print(f"  эпизодов: {len(dqn_cp.ep_rewards)}")

    print("→ PPO / CartPole-v1 ...")
    ppo_cp = train_ppo("CartPole-v1", timesteps=100_000)
    ppo_cp.save(f"{OUT}/ppo_cartpole.csv")
    print(f"  эпизодов: {len(ppo_cp.ep_rewards)}")

    FL = dict(map_name="8x8", is_slippery=True)

    print("→ DQN / FrozenLake-v1 (8x8, slippery) ...")
    dqn_fl = train_dqn(
        "FrozenLake-v1", timesteps=300_000, env_kwargs=FL,
        learning_starts=5_000,
        buffer_size=50_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
    )
    dqn_fl.save(f"{OUT}/dqn_frozenlake.csv")
    print(f"  эпизодов: {len(dqn_fl.ep_rewards)}")

    print("→ PPO / FrozenLake-v1 (8x8, slippery) ...")
    ppo_fl = train_ppo(
        "FrozenLake-v1", timesteps=300_000, env_kwargs=FL,
        ent_coef=0.02,
    )
    ppo_fl.save(f"{OUT}/ppo_frozenlake.csv")
    print(f"  эпизодов: {len(ppo_fl.ep_rewards)}")

    print(f"\nОбучение завершено за {time.time() - t0:.1f} с\n")

    plot_curves(
        [("DQN", dqn_cp.ep_rewards, "tab:blue"),
         ("PPO", ppo_cp.ep_rewards, "tab:orange")],
        title="CartPole-v1: средняя награда (скользящее среднее, окно = 100)",
        ylabel="Средняя награда за эпизод",
        out_pdf=f"{OUT}/fig_cartpole.pdf",
        ylim=(0, 520), threshold=475, w=100,
    )

    plot_curves(
        [("DQN", dqn_fl.ep_rewards, "tab:blue"),
         ("PPO", ppo_fl.ep_rewards, "tab:orange")],
        title="FrozenLake-v1 (8×8): доля успешных эпизодов (окно = 100)",
        ylabel="Доля успешных эпизодов",
        out_pdf=f"{OUT}/fig_frozenlake.pdf",
        ylim=(0, 1.0), threshold=0.8, w=100,
    )