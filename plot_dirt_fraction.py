import numpy as np
import matplotlib.pyplot as plt
from utils import *

def plot_dirt_fraction(dirt_fractions, episode_boundaries=None, save_path=None):
    """
    Plot dirt fraction over timesteps and count time spent in each category.

    Args:
        dirt_fractions:     list or 1-D array of dirtFraction values (one per timestep)
        episode_boundaries: list of timestep indices where episodes START (e.g. [0, 1000, 2000])
        save_path:          where to save the figure
    """
    df = np.asarray(dirt_fractions, dtype=float)
    timesteps = np.arange(len(df))

    high = int(np.sum(df >= 0.32))
    mid  = int(np.sum((df >= 0.20) & (df < 0.32)))
    low  = int(np.sum(df < 0.20))
    total = len(df)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: time-series with coloured background bands
    ax1.axhspan(0.32, max(1.0, df.max() + 0.05), alpha=0.12, color="#d62728", label="df ≥ 0.32")
    ax1.axhspan(0.20, 0.32,                       alpha=0.12, color="#ff7f0e", label="0.20 ≤ df < 0.32")
    ax1.axhspan(0.0,  0.20,                        alpha=0.12, color="#2ca02c", label="df < 0.20")
    ax1.axhline(0.32, color="#d62728", linestyle="--", linewidth=0.9)
    ax1.axhline(0.20, color="#ff7f0e", linestyle="--", linewidth=0.9)
    ax1.plot(timesteps, df, color="black", linewidth=0.6, alpha=0.85)

    if episode_boundaries:
        for i, b in enumerate(episode_boundaries):
            ax1.axvline(b, color="steelblue", linestyle=":", linewidth=0.7, alpha=0.6,
                        label="episode start" if i == 0 else None)
        # mark initial df per episode
        for b in episode_boundaries:
            if b < len(df):
                ax1.scatter(b, df[b], color="steelblue", s=20, zorder=5)

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Dirt Fraction")
    ax1.set_title("Dirt Fraction Over Time")
    ax1.set_ylim(0, max(1.0, df.max() + 0.05))
    ax1.legend(loc="upper right", fontsize=8)

    # Right: bar chart of category counts
    labels = ["df ≥ 0.32", "0.20 ≤ df < 0.32", "df < 0.20"]
    counts = [high, mid, low]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax2.bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}\n({100 * count / total:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax2.set_ylabel("Timestep Count")
    ax2.set_title("Timesteps per Dirt Fraction Category")

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    # plt.show()

    # ── console summary ────────────────────────────────────────────────────────
    if episode_boundaries:
        print("\nInitial dirt fraction per episode:")
        for i, b in enumerate(episode_boundaries):
            if b < len(df):
                print(f"  Episode {i + 1} (t={b:6d}): {df[b]:.4f}")

    print(f"\nTimestep counts (total={total}):")
    print(f"  df >= 0.32          : {high:6d}  ({100 * high / total:5.1f}%)")
    print(f"  0.20 <= df < 0.32   : {mid:6d}  ({100 * mid  / total:5.1f}%)")
    print(f"  df < 0.20           : {low:6d}  ({100 * low  / total:5.1f}%)")

    return high, mid, low


# ── Usage ─────────────────────────────────────────────────────────────────────
#
# Episode boundaries are every 1000 steps, so compute them directly:
#
#   EPISODE_LEN = 1000
#
# Option A – Python for-loop (evaluation):
#
#   dirt_fractions = []
#   obs, state = env.reset(key)
#   for t in range(num_eval_steps):
#       actions = ...
#       obs, state, rewards, done, info = env.step_env(key, state, actions)
#       dirt_fractions.append(float(info["dirtFraction"]))
#
#   episode_boundaries = list(range(0, len(dirt_fractions), EPISODE_LEN))
#   plot_dirt_fraction(dirt_fractions, episode_boundaries)
#
#
# Option B – jax.lax.scan (info["dirtFraction"] has shape (num_steps,)):
#
#   dirt_fractions = np.array(info["dirtFraction"])
#   episode_boundaries = list(range(0, len(dirt_fractions), EPISODE_LEN))
#   plot_dirt_fraction(dirt_fractions, episode_boundaries)
