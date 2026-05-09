"""Compare hand-rolled CBF vs learned NCBF safety performance as MPPI horizon grows.

For each horizon H in {4, 8, 16, 32, 64}, runs both controllers on the same set of
100 random (start, goal) tasks in a fixed environment and reports how many
trajectories had at least one safety violation (h(x) > 0) according to the true h.
"""

from functools import partial
from pathlib import Path
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dynamics.environment_dynamics import Parameters, State, step_state
from dynamics.dubins_dynamics import DubinsState
from dynamics.obstacle_dynamics import ObstacleState
from controllers.mppi import (
    MPPIDynamicParameters,
    MPPIParameters,
    MPPIState,
    mppi_compute_action,
)
from safety import cbf
from tasks.dubins import compute_h_vector, make_goal_reaching_task
from environments.dubins import ENVIRONMENTS, make_environment
from networks.ncbf import load_checkpoint, NCBF, NCBFNetwork


HORIZONS = [4, 8, 16, 32]
NUM_TASKS = 100
NUM_STEPS = 120
DT = 0.05
NUM_ROLLOUTS = 256
TEMP = 1.0
VARIANCES = [2.0, 2.0]
CBF_ALPHA = 0.92
VIO_COST = 10_000.0
SAMPLE_MARGIN = 0.0  # keep starts/goals away from walls


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Compare CBF and NCBF safety vs. MPPI horizon."
    )
    parser.add_argument(
        "--ncbf",
        type=Path,
        required=True,
        help="Path to NCBF checkpoint directory.",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=sorted(ENVIRONMENTS.keys()),
        help="Environment name.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("horizon_experiment.png"),
        help="Where to save the resulting plot.",
    )
    return parser.parse_args()


def _make_initial_state(x, y, theta, params: Parameters) -> State:
    num_obstacles = params.obstacle_params.radius.shape[0]
    return State(
        dubins_state=DubinsState(
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            v=jnp.array(0.0),
            theta=jnp.asarray(theta),
        ),
        obstacle_state=ObstacleState(
            alpha=jnp.zeros(num_obstacles),
            forward=jnp.ones(num_obstacles, dtype=bool),
        ),
    )


def sample_safe_point(key, params: Parameters):
    """Rejection-sample an (x, y) within bounds whose true h < 0."""
    x_lo = float(params.x_min) + SAMPLE_MARGIN
    x_hi = float(params.x_max) - SAMPLE_MARGIN
    y_lo = float(params.y_min) + SAMPLE_MARGIN
    y_hi = float(params.y_max) - SAMPLE_MARGIN

    for _ in range(1000):
        key, sub = jax.random.split(key)
        xy = jax.random.uniform(
            sub,
            (2,),
            minval=jnp.array([x_lo, y_lo]),
            maxval=jnp.array([x_hi, y_hi]),
        )
        candidate = _make_initial_state(xy[0], xy[1], 0.0, params)
        h = jnp.max(compute_h_vector(candidate, params))
        if float(h) < -SAMPLE_MARGIN:
            return key, float(xy[0]), float(xy[1])
    raise RuntimeError("Failed to sample a safe point in 1000 tries.")


def generate_tasks(key, params: Parameters, num_tasks: int):
    """Generate (initial_state, goal) pairs as batched pytrees of length num_tasks."""
    starts_x, starts_y, starts_theta = [], [], []
    goals = []
    for _ in range(num_tasks):
        key, x_s, y_s = sample_safe_point(key, params)
        key, x_g, y_g = sample_safe_point(key, params)
        key, theta_key = jax.random.split(key)
        theta = float(jax.random.uniform(theta_key, (), minval=-jnp.pi, maxval=jnp.pi))
        starts_x.append(x_s)
        starts_y.append(y_s)
        starts_theta.append(theta)
        goals.append([x_g, y_g, 0.0])

    num_obstacles = params.obstacle_params.radius.shape[0]
    initial_states = State(
        dubins_state=DubinsState(
            x=jnp.array(starts_x),
            y=jnp.array(starts_y),
            v=jnp.zeros(num_tasks),
            theta=jnp.array(starts_theta),
        ),
        obstacle_state=ObstacleState(
            alpha=jnp.zeros((num_tasks, num_obstacles)),
            forward=jnp.ones((num_tasks, num_obstacles), dtype=bool),
        ),
    )
    return initial_states, jnp.array(goals)


def make_simulator(use_ncbf: bool, ncbf_network: NCBFNetwork | None):
    """Returns a jit-compiled simulator that runs MPPI for `num_steps` and reports
    whether the trajectory ever violated the true h."""

    if use_ncbf:
        h_fn = NCBF(h_fn=compute_h_vector, ncbf_network=ncbf_network)
    else:
        h_fn = compute_h_vector

    @partial(jax.jit, static_argnames=("horizon", "num_steps"))
    def simulate(
        initial_state: State,
        goal: jnp.ndarray,
        params: Parameters,
        mppi_key: jnp.ndarray,
        horizon: int,
        num_steps: int,
    ):
        task_cost, task_term, _ = make_goal_reaching_task(goal)

        compute_cbf_violation = cbf.cbf_violation(h_fn, DT)
        cost_fn, terminal_cost_fn = cbf.embed_cbf_violation(
            compute_cbf_violation, task_cost, task_term, CBF_ALPHA, VIO_COST
        )

        mppi_state = MPPIState(
            actions=jnp.zeros((horizon, 2)),
            key=mppi_key,
        )
        mppi_params = MPPIParameters(num_rollouts=NUM_ROLLOUTS, num_iters=1)
        mppi_dyn = MPPIDynamicParameters(
            temp=jnp.array(TEMP), variance=jnp.array(VARIANCES)
        )

        def step(carry, _):
            state, mppi_state = carry
            opt_actions, mppi_state, _ = mppi_compute_action(
                state,
                params,
                cost_fn,
                terminal_cost_fn,
                mppi_state,
                mppi_params,
                mppi_dyn,
                DT,
            )
            action = opt_actions[0]
            h_now = jnp.max(compute_h_vector(state, params))
            violated = h_now > 0.0
            next_state = step_state(state, action, params, DT)
            return (next_state, mppi_state), violated

        (final_state, _), violations = jax.lax.scan(
            step, (initial_state, mppi_state), None, length=num_steps
        )
        final_h = jnp.max(compute_h_vector(final_state, params))
        return jnp.logical_or(violations.any(), final_h > 0.0)

    return simulate


def run_for_horizon(simulate, initial_states, goals, params, mppi_keys, horizon):
    num_tasks = goals.shape[0]
    results = np.zeros(num_tasks, dtype=bool)
    for i in range(num_tasks):
        init_i = jax.tree.map(lambda leaf: leaf[i], initial_states)
        violated = simulate(init_i, goals[i], params, mppi_keys[i], horizon, NUM_STEPS)
        results[i] = bool(violated)
    return results


def main():
    args = get_arguments()

    key = jax.random.key(seed=args.seed)
    env_key, task_key, mppi_key = jax.random.split(key, 3)

    print(f"Building environment '{args.env}'.")
    params = make_environment(args.env, key=env_key)

    print(f"Loading NCBF network from {args.ncbf}.")
    ncbf_network: NCBFNetwork = load_checkpoint(args.ncbf)[0]

    print(f"Sampling {NUM_TASKS} random tasks.")
    initial_states, goals = generate_tasks(task_key, params, NUM_TASKS)

    mppi_keys = jax.random.split(mppi_key, NUM_TASKS)

    sim_cbf = make_simulator(use_ncbf=False, ncbf_network=None)
    sim_ncbf = make_simulator(use_ncbf=True, ncbf_network=ncbf_network)

    cbf_rates = []
    ncbf_rates = []
    for H in HORIZONS:
        print(f"\nHorizon H={H}")
        print("  Running CBF...")
        cbf_violated = run_for_horizon(
            sim_cbf, initial_states, goals, params, mppi_keys, H
        )
        print("  Running NCBF...")
        ncbf_violated = run_for_horizon(
            sim_ncbf, initial_states, goals, params, mppi_keys, H
        )
        cbf_rate = 100.0 * cbf_violated.mean()
        ncbf_rate = 100.0 * ncbf_violated.mean()
        cbf_rates.append(cbf_rate)
        ncbf_rates.append(ncbf_rate)
        print(f"  CBF collisions:  {cbf_violated.sum()}/{NUM_TASKS} ({cbf_rate:.1f}%)")
        print(
            f"  NCBF collisions: {ncbf_violated.sum()}/{NUM_TASKS} ({ncbf_rate:.1f}%)"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(HORIZONS, cbf_rates, "o-", label="specification-based CBF")
    ax.plot(HORIZONS, ncbf_rates, "s-", label="NCBF")
    ax.set_xscale("log", base=2)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("MPPI horizon")
    ax.set_ylabel("trajectories with safety violation (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Safety vs. horizon on '{args.env}' ({NUM_TASKS} tasks, {NUM_STEPS} steps)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    print(f"\nSaving plot to {args.out}")
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
