"""
Load an .npz file containing rollout states and h-values, then plot:
  1. A heatmap of state visitation counts (how often each (x, y) bin was visited).
  2. A heatmap of the mean max-h value in each visited bin.

Usage:
    python plot_heatmap.py path/to/rollout_data.npz [--resolution 50]

Expected .npz keys (flat arrays from a State pytree):
    dubins_x      : (num_rollouts, rollout_length)
    dubins_y      : (num_rollouts, rollout_length)
    dubins_v      : (num_rollouts, rollout_length)
    dubins_theta  : (num_rollouts, rollout_length)
    obs_alpha     : (num_rollouts, rollout_length, num_obstacles)
    obs_forward   : (num_rollouts, rollout_length, num_obstacles)
    hs            : (num_rollouts, rollout_length, h_dim)
"""

import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from environments.dubins import get_environment_parameters


def load_data(path: str):
    """Load the .npz and return flat x, y, v, theta arrays and hs."""
    data = jnp.load(path, allow_pickle=True)
    # print(list(data.keys()))
    states = data['states'].item()
    hs = data['hs']

    xs = states.dubins_state.x.ravel()       # flatten across rollouts & timesteps
    ys = states.dubins_state.y.ravel()
    vs = states.dubins_state.v.ravel()
    thetas = states.dubins_state.theta.ravel()

    # hs shape: (num_rollouts, rollout_length, h_dim) → take max across h_dim
    hs_raw = hs
    hs_max = hs_raw.max(axis=-1).ravel()  # max over the h components

    return xs, ys, vs, thetas, hs_max


def plot_heatmaps(xs, ys, hs, x_min, x_max, y_min, y_max, resolution, save=False):
    """Create visitation-count and mean-h heatmaps."""
    x_edges = np.linspace(x_min, x_max, resolution + 1)
    y_edges = np.linspace(y_min, y_max, resolution + 1)

    # --- Visitation count ---
    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    counts = counts.T  # orient so that y is the vertical axis

    # --- Mean max-h per bin ---
    h_sum, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=hs)
    h_sum = h_sum.T
    with np.errstate(invalid="ignore"):
        h_mean = np.where(counts > 0, h_sum / counts, np.nan)

    # Mid-points for labeling
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # -- Visitation count heatmap --
    ax = axes[0]
    counts_masked = np.ma.masked_where(counts == 0, counts)
    im0 = ax.pcolormesh(
        x_edges, y_edges, counts_masked,
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=counts.max()),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("State Visitation Count (log scale)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im0, ax=ax, label="visits")

    # -- Mean h-value heatmap --
    ax = axes[1]
    im1 = ax.pcolormesh(
        x_edges, y_edges, h_mean,
        cmap="RdYlGn_r",  # red = high h (unsafe), green = low h (safe)
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mean max h(x) per Bin  (>0 ⇒ unsafe)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im1, ax=ax, label="mean max h")

    plt.tight_layout()
    if save:
        plt.savefig("heatmap.png", dpi=200)
        print("Saved heatmap.png")
    plt.show()


def plot_distributions(vs, thetas, save=False):
    """Plot histograms of velocity and theta distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    vs_np = np.asarray(vs)

    # -- Velocity distribution (full) --
    ax = axes[0]
    ax.hist(vs_np, bins=80, color="#2196F3", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Velocity (v)")
    ax.set_ylabel("Count")
    ax.set_title("Velocity Distribution (full)")
    ax.axvline(float(np.mean(vs_np)), color="red", linestyle="--", linewidth=1.5, label=f"mean = {float(np.mean(vs_np)):.3f}")
    ax.legend()

    # -- Velocity distribution (zoomed, outliers removed) --
    percentiles = [ 25, 75 ]
    p_lo, p_hi = np.percentile(vs_np, percentiles)
    vs_core = vs_np[(vs_np >= p_lo) & (vs_np <= p_hi)]
    ax = axes[1]
    ax.hist(vs_core, bins=80, color="#2196F3", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Velocity (v)")
    ax.set_ylabel("Count")
    ax.set_title(f"Velocity Distribution ({percentiles[0]}-{percentiles[1]} pctl)")
    ax.axvline(float(np.mean(vs_core)), color="red", linestyle="--", linewidth=1.5, label=f"mean = {float(np.mean(vs_core)):.3f}")
    ax.legend()

    # -- Theta distribution --
    thetas = (thetas + np.pi) % (2 * np.pi) - np.pi
    ax = axes[2]
    ax.hist(np.asarray(thetas), bins=80, color="#FF9800", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Theta (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Theta Distribution")
    ax.axvline(float(np.mean(thetas)), color="red", linestyle="--", linewidth=1.5, label=f"mean = {float(np.mean(thetas)):.3f}")
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig("distributions.png", dpi=200)
        print("Saved distributions.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot state-visitation heatmap from rollout .npz")
    parser.add_argument("npz_path", type=str, help="Path to the .npz file")
    parser.add_argument("--resolution", type=int, default=50,
                        help="Number of bins per axis (default: 50)")
    parser.add_argument("--save", action="store_true",
                        help="Save the plot to heatmap.png")
    args = parser.parse_args()

    # Load space bounds from parameters
    params = get_environment_parameters("basic")
    x_min = float(params.x_min)
    x_max = float(params.x_max)
    y_min = float(params.y_min)
    y_max = float(params.y_max)

    xs, ys, vs, thetas, hs = load_data(args.npz_path)

    print(f"Loaded {len(xs):,} data points from {args.npz_path}")
    print(f"Space bounds: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")
    print(f"Grid resolution: {args.resolution} × {args.resolution}")
    print(f"Velocity range: [{float(vs.min()):.3f}, {float(vs.max()):.3f}]")
    print(f"Theta range:    [{float(thetas.min()):.3f}, {float(thetas.max()):.3f}]")

    plot_heatmaps(xs, ys, hs, x_min, x_max, y_min, y_max, args.resolution, save=args.save)
    plot_distributions(vs, thetas, save=args.save)


if __name__ == "__main__":
    main()
