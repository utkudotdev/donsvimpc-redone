import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

from environment.environment_dynamics import Parameters, State, step_state
from controllers.mppi import (
    MPPIDynamicParameters,
    MPPIParameters,
    MPPIState,
    mppi_compute_action,
    mppi_rollout
)
from environment.obstacle_dynamics import ObstacleParameters, ObstacleState
from environment.dubins_dynamics import DubinsParameters, DubinsState

def main():
    dubins_params = DubinsParameters(
        turn_rate_min=jnp.array(-1.0),
        turn_rate_max=jnp.array(1.0),
        velocity_min=jnp.array(-1.0),
        velocity_max=jnp.array(1.0)
    )
    obs_params = ObstacleParameters(
        radius=jnp.array(0.3),
        speed=jnp.array(-0.5),
        start_point=jnp.array([+0.5, 1.0]),
        end_point=jnp.array([-0.5, 1.0]),
    )
    x_min, x_max = -0.5, 0.5
    y_min, y_max = 0.0, 2.0
    params = Parameters(
        dubins_params=dubins_params,
        obstacle_params=obs_params,
        x_min=jnp.array(x_min),
        x_max=jnp.array(x_max),
        y_min=jnp.array(y_min),
        y_max=jnp.array(y_max),
    )

    state = State(
        dubins_state=DubinsState(
            x=jnp.array(0.0),
            y=jnp.array(0.5),
            theta=jnp.array(0.0),
        ),
        obstacle_state=ObstacleState(alpha=jnp.array(0.5), forward=jnp.array(True)),
    )

    goal = jnp.array([ 0, 1.50 ])
    
    num_steps = 120
    dt = 0.05

    horizon = 40

    mppi_state = MPPIState(
        actions=jnp.full((horizon, 2), fill_value=0.0),
        key=jax.random.key(seed=0),
    )
    mppi_params = MPPIParameters(
        num_rollouts=256,
        num_iters=1,
    )
    mppi_dynamic_params = MPPIDynamicParameters(
        temp=jnp.array(0.10),
        variance=jnp.array([0.01, 0.01]),
    )

    def task_cost_fn(s: State, a: jnp.ndarray) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        ctrl = jnp.sum(a**2)
        return pos_err + 0.01 * ctrl

    def task_terminal_cost_fn(s: State) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        return pos_err

    def compute_h_vector(s: State, p: Parameters):
        h_boundary = jnp.max(jnp.array([ 
            s.dubins_state.x - p.x_max, 
            p.x_min - s.dubins_state.x,
            s.dubins_state.y - p.y_max, 
            p.y_min - s.dubins_state.y]))
         
        dubins_position = jnp.array([ s.dubins_state.x, s.dubins_state.y ])
        obstacle_position = s.obstacle_state.position(p.obstacle_params)
        
        signed_distance = jnp.linalg.norm(dubins_position - obstacle_position) - p.obstacle_params.radius
        
        h_obstacles = -signed_distance 

        return jnp.array([
            h_obstacles, 
            h_boundary
        ])

    def compute_cbf_violation(s: State, a: jnp.ndarray, p: Parameters, alpha: jnp.ndarray):
        h = jnp.max(compute_h_vector(s, p))
        s_prime = step_state(s, a, p, dt)
        h_prime = jnp.max(compute_h_vector(s_prime, p))
        return h_prime + (alpha - 1) * h
        
    def cost_fn(s: State, a: jnp.ndarray, p: Parameters):
        h_violation = compute_cbf_violation(s, a, p, alpha=0.95)
        cbf_cost = jnp.where(h_violation > 0, 100 * h_violation, 0.0)
        task_cost = task_cost_fn(s, a)
        return task_cost + 10 * cbf_cost    
        
    def terminal_cost_fn(s: State, p: Parameters):
        return task_terminal_cost_fn(s)

    # --- GRID VISUALIZATION ---
    grid_x = jnp.linspace(x_min, x_max, 50)
    grid_y = jnp.linspace(y_min, y_max, 50)
    X, Y = jnp.meshgrid(grid_x, grid_y)

    @jax.jit
    def eval_grid_point(x, y):
        s = State(
            dubins_state=DubinsState(x=x, y=y, theta=jnp.array(jnp.pi / 2.0)),
            obstacle_state=state.obstacle_state
        )
        a = jnp.array([0.0, 0.1]) # zero turn rate, small forward velocity
        h_val = jnp.max(compute_h_vector(s, params))
        cbf_viol = compute_cbf_violation(s, a, params, alpha=jnp.array(0.95))
        return h_val, cbf_viol

    vec_eval = jax.vmap(jax.vmap(eval_grid_point, in_axes=(0, 0)), in_axes=(0, 0))
    H_vals, CBF_viols = vec_eval(X, Y)

    fig_grid, ax_grid = plt.subplots(1, 2, figsize=(12, 5))
    c1 = ax_grid[0].contourf(X, Y, H_vals, levels=30, cmap='coolwarm')
    ax_grid[0].set_title("h(x, y) [>0 is unsafe]")
    fig_grid.colorbar(c1, ax=ax_grid[0])
    
    c2 = ax_grid[1].contourf(X, Y, CBF_viols, levels=30, cmap='inferno')
    ax_grid[1].set_title("CBF Violation (theta=pi/2, v=0.1)")
    fig_grid.colorbar(c2, ax=ax_grid[1])
    plt.show()
    # --------------------------

    xs, ys, thetas = [float(state.dubins_state.x)], [float(state.dubins_state.y)], [float(state.dubins_state.theta)]
    
    obs_pos = state.obstacle_state.position(obs_params)
    obs_xs, obs_ys = [float(obs_pos[0])], [float(obs_pos[1])]
    
    rollout_xs, rollout_ys = [], []  # per-step (num_rollouts, horizon)
    opt_rollout_xs, opt_rollout_ys = [], [] # per-step (horizon, )
    
    for _ in range(num_steps):
        optimized_actions, mppi_state, rollouts = mppi_compute_action(
            state,
            params,
            cost_fn,
            terminal_cost_fn,
            mppi_state,
            mppi_params,
            mppi_dynamic_params,
            dt
        )
        action = optimized_actions[0]
        print(f'First={action}, Min={jnp.min(optimized_actions)}, Max={jnp.max(optimized_actions)}')

        # Generate full optimal trajectory
        _, trajs = mppi_rollout(state, optimized_actions, params, cost_fn, terminal_cost_fn, dt)
        opt_rollout_xs.append(np.asarray(trajs.dubins_state.x))
        opt_rollout_ys.append(np.asarray(trajs.dubins_state.y))

        # Store counterfactual rollouts
        rollout_xs.append(np.asarray(rollouts.dubins_state.x))
        rollout_ys.append(np.asarray(rollouts.dubins_state.y))

        # Advance simulation
        state = step_state(state, action, params, dt)
        xs.append(float(state.dubins_state.x))
        ys.append(float(state.dubins_state.y))
        thetas.append(float(state.dubins_state.theta))
        
        obs_pos = state.obstacle_state.position(obs_params)
        obs_xs.append(float(obs_pos[0]))
        obs_ys.append(float(obs_pos[1]))

    xs = np.array(xs)
    ys = np.array(ys)
    thetas = np.array(thetas)
    obs_xs = np.array(obs_xs)
    obs_ys = np.array(obs_ys)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
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
    obstacle_body = patches.Circle((0, 0), radius=float(obs_params.radius), color="tab:red", alpha=0.5, label="obstacle")
    ax.add_patch(obstacle_body)
    
    title = ax.set_title("")
    ax.legend(loc="upper left")

    def update(i):
        trail.set_data(xs[: i + 1], ys[: i + 1])
        car_body.set_data([xs[i]], [ys[i]])
        obstacle_body.set_center((obs_xs[i], obs_ys[i]))
        
        # --- DUBINS CAR HEADING CALCULATION ---
        # Length of the directional pointer
        heading_length = 0.3 

        # Calculate where the pointer should end based on current theta
        dx = heading_length * np.cos(thetas[i])
        dz = heading_length * np.sin(thetas[i])
        
        # Draw the pointer from the car's center outward
        car_heading.set_data([xs[i], xs[i] + dx], [ys[i], ys[i] + dz])

        if i < len(rollout_xs):
            rx = rollout_xs[i]
            rz = rollout_ys[i]
            for k, line in enumerate(rollout_lines):
                line.set_data(rx[k], rz[k])
        
        if i < len(opt_rollout_xs):
            rx = opt_rollout_xs[i]
            rz = opt_rollout_ys[i]
            opt_rollout_line.set_data(rx, rz)

        title.set_text(f"step {i}/{num_steps}   pos=({xs[i]:+.2f}, {ys[i]:+.2f})")
        
        return [trail, car_body, car_heading, obstacle_body, title, *rollout_lines, opt_rollout_line]

    _anim = FuncAnimation(
        fig, update, frames=len(xs), interval=dt * 1000, blit=False, repeat=False
    )
    plt.show()

if __name__ == "__main__":
    main()
