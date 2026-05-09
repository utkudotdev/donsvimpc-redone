"""Plot results saved by run_horizon_experiment.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser(description="Plot horizon experiment results.")
    parser.add_argument(
        "data",
        type=Path,
        help="Path to .npz produced by run_horizon_experiment.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("horizon_experiment.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    data = np.load(args.data, allow_pickle=True)

    horizons = data["horizons"]
    cbf_collide_rate = 100.0 * data["cbf_collided"].mean(axis=1)
    ncbf_collide_rate = 100.0 * data["ncbf_collided"].mean(axis=1)
    cbf_reach_rate = 100.0 * data["cbf_reached"].mean(axis=1)
    ncbf_reach_rate = 100.0 * data["ncbf_reached"].mean(axis=1)

    env = str(data["env"])
    num_tasks = int(data["num_tasks"])
    num_steps = int(data["num_steps"])
    goal_radius = float(data["goal_radius"])
    goal_dwell_steps = int(data["goal_dwell_steps"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(horizons, cbf_collide_rate, "o-", label="specification-based CBF")
    ax.plot(horizons, ncbf_collide_rate, "s-", label="NCBF")
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(h)) for h in horizons])
    ax.set_xlabel("MPPI horizon")
    ax.set_ylabel("trajectories with safety violation (%)")
    ax.set_ylim(0, 100)
    ax.set_title("safety violations")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(horizons, cbf_reach_rate, "o-", label="specification-based CBF")
    ax.plot(horizons, ncbf_reach_rate, "s-", label="NCBF")
    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(h)) for h in horizons])
    ax.set_xlabel("MPPI horizon")
    ax.set_ylabel(
        f"trajectories reaching goal (within {goal_radius} for "
        f"{goal_dwell_steps} steps, no prior collision) (%)"
    )
    ax.set_ylim(0, 100)
    ax.set_title("goal reaching")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"'{env}' ({num_tasks} tasks, {num_steps} steps)")
    fig.tight_layout()

    print(f"Saving plot to {args.out}")
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
