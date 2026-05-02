from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

from dynamics.environment_dynamics import Parameters, State, step_state
from controllers.mppi import (
    MPPIDynamicParameters,
    MPPIParameters,
    MPPIState,
    mppi_compute_action,
    mppi_rollout,
)
from dynamics.obstacle_dynamics import ObstacleState
from dynamics.dubins_dynamics import DubinsState

from safety import cbf
from tasks.dubins import compute_h_vector, make_goal_reaching_task
from environments.dubins import ENVIRONMENTS, get_environment_parameters
from environments.discovery import discover_env_name

from networks.ncbf import load_checkpoint, NCBF, NCBFNetwork

import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser(description="Run Dubin's car.")
    parser.add_argument(
        "--ncbf",
        type=Path,
        default=None,
        help="Path to NCBF checkpoint. Providing this will use the NCBF safety layer. ",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=sorted(ENVIRONMENTS.keys()),
        help="Environment name. Defaults to the env from the NCBF checkpoint metadata, or 'basic' if no checkpoint is given.",
    )

    return parser.parse_args()


def make_mppi_controller(
    horizon: int,
    num_rollouts: int,
    temp: float,
    variances: list[float],
    key: jnp.ndarray,
):
    mppi_state = MPPIState(
        actions=jnp.full((horizon, 2), fill_value=0.0),
        key=key,
    )
    mppi_params = MPPIParameters(
        num_rollouts=num_rollouts,
        num_iters=1,
    )
    mppi_dynamic_params = MPPIDynamicParameters(
        temp=jnp.array(temp),
        variance=jnp.array(variances),
    )

    return mppi_state, mppi_params, mppi_dynamic_params


@partial(
    jax.jit,
    static_argnames=(
        "cost_fn",
        "terminal_cost_fn",
        "collision_checker",
        "mppi_params",
        "num_steps",
    ),
)
def run_simulation(
    initial_state: State,
    params: Parameters,
    cost_fn,
    terminal_cost_fn,
    collision_checker,
    mppi_state: MPPIState,
    mppi_params: MPPIParameters,
    mppi_dynamic_params: MPPIDynamicParameters,
    dt: float,
    num_steps: int,
):
    def step(carry, _):
        state, mppi_state = carry

        optimized_actions, mppi_state, rollouts = mppi_compute_action(
            state,
            params,
            cost_fn,
            terminal_cost_fn,
            mppi_state,
            mppi_params,
            mppi_dynamic_params,
            dt,
        )
        action = optimized_actions[0]

        _, opt_traj = mppi_rollout(
            state, optimized_actions, params, cost_fn, terminal_cost_fn, dt
        )

        h_vector = collision_checker(state, params)
        violated = (h_vector > 0.0).any()

        def _print_violation(operand):
            h, s = operand
            jax.debug.print("h(x) = {h}", h=h)
            jax.debug.print("x = {s}", s=s)

        jax.lax.cond(violated, _print_violation, lambda _: None, (h_vector, state))

        next_state = step_state(state, action, params, dt)

        outputs = {
            "state": state,
            "rollouts": rollouts,
            "opt_traj": opt_traj,
            "violated": violated,
        }
        return (next_state, mppi_state), outputs

    (final_state, _), outputs = jax.lax.scan(
        step, (initial_state, mppi_state), None, length=num_steps
    )

    # Append the final state so the trajectory has length num_steps + 1.
    states = jax.tree.map(
        lambda pre, fin: jnp.concatenate([pre, fin[None]], axis=0),
        outputs["state"],
        final_state,
    )

    return states, outputs["rollouts"], outputs["opt_traj"], outputs["violated"]


def visualize(
    params: Parameters,
    states: State,
    rollouts: State,
    opt_trajs: State,
    violated: jnp.ndarray,
    goal: jnp.ndarray,
    num_rollouts: int,
    dt: float,
    num_steps: int,
):
    xs = np.asarray(states.dubins_state.x)
    ys = np.asarray(states.dubins_state.y)
    thetas = np.asarray(states.dubins_state.theta)

    obs_pos = jax.vmap(jax.vmap(ObstacleState.position), in_axes=(0, None))(
        states.obstacle_state, params.obstacle_params
    )
    obs_xs = np.asarray(obs_pos[..., 0])
    obs_ys = np.asarray(obs_pos[..., 1])

    rollout_xs = np.asarray(rollouts.dubins_state.x)
    rollout_ys = np.asarray(rollouts.dubins_state.y)
    opt_rollout_xs = np.asarray(opt_trajs.dubins_state.x)
    opt_rollout_ys = np.asarray(opt_trajs.dubins_state.y)
    violation_steps = [int(i) for i, v in enumerate(np.asarray(violated)) if v]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(params.x_min, params.x_max)
    ax.set_ylim(params.y_min, params.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.plot(float(goal[0]), float(goal[1]), "g*", markersize=18, label="goal")
    (trail,) = ax.plot([], [], "-", color="tab:blue", linewidth=1.5, label="executed")

    rollout_lines = []
    for _ in range(num_rollouts):
        (line,) = ax.plot([], [], "-", color="tab:orange", alpha=0.05, linewidth=0.6)
        rollout_lines.append(line)

    (opt_rollout_line,) = ax.plot(
        [], [], "-", color="tab:green", alpha=1.0, linewidth=2.0
    )

    (car_heading,) = ax.plot([], [], "-", color="black", linewidth=2.5)
    (car_body,) = ax.plot([], [], "o", color="tab:blue", markersize=8)

    obstacle_bodies = []
    num_obstacles = obs_xs.shape[1]
    radii = np.asarray(params.obstacle_params.radius)
    for k in range(num_obstacles):
        body = patches.Circle(
            (obs_xs[0, k], obs_ys[0, k]),
            radius=float(radii[k]),
            color="tab:red",
            alpha=0.5,
            label="obstacle" if k == 0 else None,
        )
        ax.add_patch(body)
        obstacle_bodies.append(body)

    (violations,) = ax.plot([], [], "o", color="tab:red", markersize=4)

    title = ax.set_title("")
    ax.legend(loc="upper left")

    def update(i):
        trail.set_data(xs[: i + 1], ys[: i + 1])
        car_body.set_data([xs[i]], [ys[i]])
        for k in range(num_obstacles):
            obstacle_bodies[k].set_center((obs_xs[i, k], obs_ys[i, k]))

        heading_length = 0.3
        dx = heading_length * np.cos(thetas[i])
        dy = heading_length * np.sin(thetas[i])
        car_heading.set_data([xs[i], xs[i] + dx], [ys[i], ys[i] + dy])

        if i < rollout_xs.shape[0]:
            for k, line in enumerate(rollout_lines):
                line.set_data(rollout_xs[i, k], rollout_ys[i, k])

        if i < opt_rollout_xs.shape[0]:
            opt_rollout_line.set_data(opt_rollout_xs[i], opt_rollout_ys[i])

        violations.set_data(
            [xs[s] for s in violation_steps], [ys[s] for s in violation_steps]
        )

        title.set_text(f"step {i}/{num_steps}   pos=({xs[i]:+.2f}, {ys[i]:+.2f})")

        return [
            trail,
            car_body,
            car_heading,
            title,
            *obstacle_bodies,
            *rollout_lines,
            opt_rollout_line,
        ]

    _anim = FuncAnimation(
        fig, update, frames=xs.shape[0], interval=dt * 1000, blit=False, repeat=False
    )
    plt.show()


def main():
    args = get_arguments()

    ncbf_path: Path | None = args.ncbf
    use_ncbf = ncbf_path is not None

    env_name = args.env if args.env is not None else discover_env_name(ncbf_path)
    params: Parameters = get_environment_parameters(env_name)

    dubins_state = DubinsState(
        x=jnp.array(1.0),
        y=jnp.array(1.0),
        v=jnp.array(0.0),
        theta=jnp.array(0.0),
    )

    num_obstacles = params.obstacle_params.radius.shape[0]
    obs_states = ObstacleState(
        alpha=jnp.zeros(num_obstacles),
        forward=jnp.ones(num_obstacles, dtype=bool),
    )

    initial_state = State(dubins_state=dubins_state, obstacle_state=obs_states)
    dt = 0.05

    horizon = 12
    num_rollouts = 256
    temp = 1.0
    variances = [2.0, 2.0]

    mppi_state, mppi_params, mppi_dynamic_params = make_mppi_controller(
        horizon, num_rollouts, temp, variances, jax.random.key(seed=0)
    )

    # goal = jnp.array([7.0, 3.5, 0.0])
    goal = jnp.array([7.0, 1.0, 0.0])
    task_cost_fn, task_terminal_cost_fn, _task_done_fn = make_goal_reaching_task(goal)

    collision_checker = compute_h_vector
    h_fn = compute_h_vector
    if use_ncbf:
        print(f"Loading NCBF network from {ncbf_path}")
        ncbf_network: NCBFNetwork = load_checkpoint(ncbf_path)[0]
        h_fn = NCBF(h_fn=h_fn, ncbf_network=ncbf_network)

    cbf_alpha = 0.92
    vio_cost = 10_000.0
    compute_cbf_violation = cbf.cbf_violation(h_fn, dt)
    cost_fn, terminal_cost_fn = cbf.embed_cbf_violation(
        compute_cbf_violation, task_cost_fn, task_terminal_cost_fn, cbf_alpha, vio_cost
    )

    num_steps = 120

    states, rollouts, opt_trajs, violated = run_simulation(
        initial_state,
        params,
        cost_fn,
        terminal_cost_fn,
        collision_checker,
        mppi_state,
        mppi_params,
        mppi_dynamic_params,
        dt,
        num_steps,
    )

    visualize(
        params,
        states,
        rollouts,
        opt_trajs,
        violated,
        goal,
        mppi_params.num_rollouts,
        dt,
        num_steps,
    )


if __name__ == "__main__":
    main()
