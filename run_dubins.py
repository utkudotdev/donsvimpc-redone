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
from environments.dubins import get_environment_parameters

from networks.ncbf import load_checkpoint, NCBF, NCBFNetwork

import argparse
from pathlib import Path

def get_arguments():
    parser = argparse.ArgumentParser(description="Run Dubin's car.")
    parser.add_argument('--ncbf', type=Path, default=None, help="Path to NCBF checkpoint. Providing this will use the NCBF safety layer. ")

    return parser.parse_args()

def make_mppi_controller(horizon: int, num_rollouts: int, temp: float, variances: list[float]):
    mppi_state = MPPIState(
        actions=jnp.full((horizon, 2), fill_value=0.0),
        key=jax.random.key(seed=0),
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

def main():
    args = get_arguments()
    
    ncbf_path: Path | None = args.ncbf
    use_ncbf = ncbf_path is not None 
    
    # Load parameters from environment
    params: Parameters = get_environment_parameters("basic")

    # Define environment initial states   
    dubins_state = DubinsState(
        x=jnp.array(0.5),
        y=jnp.array(2.75),
        v=jnp.array(0.0),
        theta=jnp.array(0.0),
    )

    obs_state_1 = ObstacleState(alpha=jnp.array(0.0), forward=jnp.array(True))
    obs_state_2 = ObstacleState(alpha=jnp.array(0.0), forward=jnp.array(True))
    obs_state_3 = ObstacleState(alpha=jnp.array(0.0), forward=jnp.array(True))
    obs_states = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), obs_state_1, obs_state_2, obs_state_3
    )

    state = State(dubins_state=dubins_state, obstacle_state=obs_states)
    dt = 0.05 # NOTE: Could include in the state

    # Define MPPI controller
    horizon = 20
    num_rollouts = 256
    temp = 1.0
    variances = [ 1.0, 1.0 ]

    
    mppi_state, mppi_params, mppi_dynamic_params = make_mppi_controller(
        horizon, num_rollouts, temp, variances) # NOTE: use eqx.module to merge mppi_params and mppi_dynamics params

    # Load task-specific cost functions
    goal = jnp.array([7.0, 3.5, 0.0])

    task_cost_fn, task_terminal_cost_fn, _task_done_fn = make_goal_reaching_task(goal)

    # Create safety filter (this project this is done by augmenting cost fn)

    collision_checker = compute_h_vector # Happen to be the same in this case
    h_fn = compute_h_vector # Dubin's car specific safety filter
    if use_ncbf:
        print(f'Loading NCBF network from {ncbf_path}')
        ncbf_network: NCBFNetwork = load_checkpoint(ncbf_path)[0]
        h_fn = NCBF(h_fn=h_fn, ncbf_network=ncbf_network) # NCBF is max ( h, V_network )

    cbf_alpha = 0.92
    vio_cost = 10_000.0
    compute_cbf_violation = cbf.cbf_violation(h_fn, dt)
    cost_fn, terminal_cost_fn = cbf.embed_cbf_violation(
        compute_cbf_violation, task_cost_fn, task_terminal_cost_fn, cbf_alpha, vio_cost)
    

    num_steps = 120



    # --- GRID VISUALIZATION ---
    grid_x = jnp.linspace(params.x_min, params.x_max, 50)
    grid_y = jnp.linspace(params.y_min, params.y_max, 50)
    X, Y = jnp.meshgrid(grid_x, grid_y)

    @jax.jit
    def eval_grid_point(x, y):
        s = State(
            dubins_state=DubinsState(
                x=x, y=y, v=jnp.array(0.1), theta=jnp.array(jnp.pi / 2.0)
            ),
            obstacle_state=state.obstacle_state,
        )
        a = jnp.array([0.0, 0.0])  # zero turn rate, small forward velocity
        h_val = jnp.max(h_fn(s, params))
        cbf_viol = compute_cbf_violation(s, a, params, alpha=cbf_alpha)
        return h_val, cbf_viol

    vec_eval = jax.vmap(jax.vmap(eval_grid_point, in_axes=(0, 0)), in_axes=(0, 0))
    H_vals, CBF_viols = vec_eval(X, Y)

    fig_grid, ax_grid = plt.subplots(1, 2, figsize=(12, 5))
    c1 = ax_grid[0].contourf(X, Y, H_vals, levels=30, cmap="coolwarm")
    ax_grid[0].set_title("h(x, y) [>0 is unsafe]")
    fig_grid.colorbar(c1, ax=ax_grid[0])

    c2 = ax_grid[1].contourf(X, Y, CBF_viols, levels=30, cmap="inferno")
    ax_grid[1].set_title("CBF Violation (theta=pi/2, v=0.1)")
    fig_grid.colorbar(c2, ax=ax_grid[1])
    plt.show()
    # --------------------------

    xs, ys, thetas = (
        [float(state.dubins_state.x)],
        [float(state.dubins_state.y)],
        [float(state.dubins_state.theta)],
    )

    obs_pos = jax.vmap(ObstacleState.position)(state.obstacle_state, params.obstacle_params)
    obs_xs, obs_ys = [np.asarray(obs_pos[:, 0])], [np.asarray(obs_pos[:, 1])]

    rollout_xs, rollout_ys = [], []  # per-step (num_rollouts, horizon)
    opt_rollout_xs, opt_rollout_ys = [], []  # per-step (horizon, )

    violation_steps = set()

    for which_step in range(num_steps):
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
        # print(f'First={action}, Min={jnp.min(optimized_actions)}, Max={jnp.max(optimized_actions)}')

        # Generate full optimal trajectory
        _, trajs = mppi_rollout(
            state, optimized_actions, params, cost_fn, terminal_cost_fn, dt
        )
        opt_rollout_xs.append(np.asarray(trajs.dubins_state.x))
        opt_rollout_ys.append(np.asarray(trajs.dubins_state.y))

        # Store counterfactual rollouts
        rollout_xs.append(np.asarray(rollouts.dubins_state.x))
        rollout_ys.append(np.asarray(rollouts.dubins_state.y))

        # Advance simulation
        h_vector = collision_checker(state, params)
        if (h_vector > 0.0).any():
            violation_steps.add(which_step)
            print(f"h(x) = {h_vector}")

        state = step_state(state, action, params, dt)
        xs.append(float(state.dubins_state.x))
        ys.append(float(state.dubins_state.y))
        thetas.append(float(state.dubins_state.theta))

        obs_pos = jax.vmap(ObstacleState.position)(state.obstacle_state, params.obstacle_params)
        obs_xs.append(np.asarray(obs_pos[:, 0]))
        obs_ys.append(np.asarray(obs_pos[:, 1]))

    xs = np.array(xs)
    ys = np.array(ys)
    thetas = np.array(thetas)
    obs_xs = np.array(obs_xs)
    obs_ys = np.array(obs_ys)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(params.x_min, params.x_max)
    ax.set_ylim(params.y_min, params.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.plot(float(goal[0]), float(goal[1]), "g*", markersize=18, label="goal")
    (trail,) = ax.plot([], [], "-", color="tab:blue", linewidth=1.5, label="executed")

    rollout_lines = []
    for _ in range(mppi_params.num_rollouts):
        (line,) = ax.plot([], [], "-", color="tab:orange", alpha=0.05, linewidth=0.6)
        rollout_lines.append(line)

    (line,) = ax.plot([], [], "-", color="tab:green", alpha=1.0, linewidth=2.0)
    opt_rollout_line = line

    # --- VISUALIZATION UPDATE FOR DUBINS CAR ---
    # Draw a line originating from the car's center to indicate its current heading (theta)
    (car_heading,) = ax.plot([], [], "-", color="black", linewidth=2.5)
    # The dot representing the car's center position
    (car_body,) = ax.plot([], [], "o", color="tab:blue", markersize=8)

    # --- VISUALIZATION UPDATE FOR OBSTACLE ---
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

        # --- DUBINS CAR HEADING CALCULATION ---
        # Length of the directional pointer
        heading_length = 0.3

        # Calculate where the pointer should end based on current theta
        dx = heading_length * np.cos(thetas[i])
        dy = heading_length * np.sin(thetas[i])

        # Draw the pointer from the car's center outward
        car_heading.set_data([xs[i], xs[i] + dx], [ys[i], ys[i] + dy])

        if i < len(rollout_xs):
            rx = rollout_xs[i]
            ry = rollout_ys[i]
            for k, line in enumerate(rollout_lines):
                line.set_data(rx[k], ry[k])

        if i < len(opt_rollout_xs):
            rx = opt_rollout_xs[i]
            ry = opt_rollout_ys[i]
            opt_rollout_line.set_data(rx, ry)

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
        fig, update, frames=len(xs), interval=dt * 1000, blit=False, repeat=False
    )
    plt.show()


if __name__ == "__main__":
    main()
