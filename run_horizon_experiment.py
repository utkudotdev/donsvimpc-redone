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
GOAL_RADIUS = 0.3  # robot is "near goal" if within this xy distance
GOAL_DWELL_STEPS = 10  # consecutive near-goal steps required to count as reached


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
        default=Path("horizon_experiment.npz"),
        help="Where to save the raw experiment results.",
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
            state, mppi_state, ever_collided, near_streak, max_streak = carry
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
            ever_collided = jnp.logical_or(ever_collided, violated)

            d = state.dubins_state
            near_goal = (
                (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2 < GOAL_RADIUS**2
            )
            counts_for_goal = jnp.logical_and(near_goal, jnp.logical_not(ever_collided))
            near_streak = jnp.where(counts_for_goal, near_streak + 1, 0)
            max_streak = jnp.maximum(max_streak, near_streak)

            next_state = step_state(state, action, params, DT)
            return (
                next_state, mppi_state, ever_collided, near_streak, max_streak,
            ), None

        init_carry = (
            initial_state, mppi_state,
            jnp.array(False), jnp.array(0), jnp.array(0),
        )
        (final_state, _, ever_collided, _, max_streak), _ = jax.lax.scan(
            step, init_carry, None, length=num_steps
        )
        final_h = jnp.max(compute_h_vector(final_state, params))
        collided = jnp.logical_or(ever_collided, final_h > 0.0)
        reached = max_streak >= GOAL_DWELL_STEPS
        return collided, reached

    return simulate


def run_for_horizon(simulate, initial_states, goals, params, mppi_keys, horizon):
    num_tasks = goals.shape[0]
    collided = np.zeros(num_tasks, dtype=bool)
    reached = np.zeros(num_tasks, dtype=bool)
    for i in range(num_tasks):
        init_i = jax.tree.map(lambda leaf: leaf[i], initial_states)
        c, r = simulate(init_i, goals[i], params, mppi_keys[i], horizon, NUM_STEPS)
        collided[i] = bool(c)
        reached[i] = bool(r)
    return collided, reached


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

    horizons = np.array(HORIZONS, dtype=np.int32)
    cbf_collided = np.zeros((len(HORIZONS), NUM_TASKS), dtype=bool)
    cbf_reached = np.zeros_like(cbf_collided)
    ncbf_collided = np.zeros_like(cbf_collided)
    ncbf_reached = np.zeros_like(cbf_collided)

    for h_idx, H in enumerate(HORIZONS):
        print(f"\nHorizon H={H}")
        print("  Running CBF...")
        cbf_collided[h_idx], cbf_reached[h_idx] = run_for_horizon(
            sim_cbf, initial_states, goals, params, mppi_keys, H
        )
        print("  Running NCBF...")
        ncbf_collided[h_idx], ncbf_reached[h_idx] = run_for_horizon(
            sim_ncbf, initial_states, goals, params, mppi_keys, H
        )

        print(
            f"  CBF :  collisions {cbf_collided[h_idx].sum()}/{NUM_TASKS}   "
            f"reached {cbf_reached[h_idx].sum()}/{NUM_TASKS}"
        )
        print(
            f"  NCBF:  collisions {ncbf_collided[h_idx].sum()}/{NUM_TASKS}   "
            f"reached {ncbf_reached[h_idx].sum()}/{NUM_TASKS}"
        )

    print(f"\nSaving results to {args.out}")
    np.savez(
        args.out,
        horizons=horizons,
        cbf_collided=cbf_collided,
        cbf_reached=cbf_reached,
        ncbf_collided=ncbf_collided,
        ncbf_reached=ncbf_reached,
        env=args.env,
        num_tasks=NUM_TASKS,
        num_steps=NUM_STEPS,
        goal_radius=GOAL_RADIUS,
        goal_dwell_steps=GOAL_DWELL_STEPS,
    )


if __name__ == "__main__":
    main()
