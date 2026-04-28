import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
        radius=jnp.array(1.5),
        speed=jnp.array(0.5),
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
        obstacle_state=ObstacleState(alpha=jnp.array(0.0), forward=jnp.array(True)),
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
        temp=jnp.array(0.1),
        variance=jnp.array([0.01, 0.01]),
    )

    def cost_fn(s: State, a: jnp.ndarray) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        ctrl = jnp.sum(a**2)
        return pos_err + 0.01 * ctrl


    def terminal_cost_fn(s: State) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        return pos_err

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
    (obstacle_body,) = ax.plot([], [], "o", color="tab:red", markersize=15, label="obstacle")
    
    title = ax.set_title("")
    ax.legend(loc="upper left")

    def update(i):
        trail.set_data(xs[: i + 1], ys[: i + 1])
        car_body.set_data([xs[i]], [ys[i]])
        obstacle_body.set_data([obs_xs[i]], [obs_ys[i]])
        
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
