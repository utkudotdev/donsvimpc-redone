import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from environment.environment_dynamics import Parameters, State, step_state
from environment.mppi import (
    MPPIDynamicParameters,
    MPPIParameters,
    MPPIState,
    mppi_compute_action,
    mppi_rollout
)
from environment.obstacle_dynamics import ObstacleParameters, ObstacleState
from environment.quadrotor_dynamics import (
    GRAVITY,
    QuadrotorParameters,
    QuadrotorState,
)


def main():
    quad_params = QuadrotorParameters(
        mass=jnp.array(0.01),
        rotor_dist=jnp.array(0.07),
        moi=jnp.array(0.00016499),
        rho=jnp.array(1.0),
        rotor_size=jnp.array(0.02),
        thrust_min=jnp.array(-5.0),
        thrust_max=jnp.array(5.0),
    )
    obs_params = ObstacleParameters(
        radius=jnp.array(0.0),
        speed=jnp.array(0.0),
        start_point=jnp.array([0.0, 0.0]),
        end_point=jnp.array([1.0, 0.0]),
    )
    x_min, x_max = -1.0, 6.0
    y_min, y_max = 0.0, 3.0
    params = Parameters(
        quadrotor_params=quad_params,
        obstacle_params=obs_params,
        x_min=jnp.array(x_min),
        x_max=jnp.array(x_max),
        y_min=jnp.array(y_min),
        y_max=jnp.array(y_max),
    )

    state = State(
        quadrotor_state=QuadrotorState(
            x=jnp.array(0.0),
            z=jnp.array(0.5),
            theta=jnp.array(0.0),
            vx=jnp.array(0.0),
            vz=jnp.array(0.0),
            w=jnp.array(0.0),
        ),
        obstacle_state=ObstacleState(alpha=jnp.array(0.0), forward=jnp.array(True)),
    )

    goal = jnp.array([4.0, 2.0])

    dt = 0.05
    horizon = 25
    num_steps = 120

    hover = 0.5 * float(quad_params.mass) * GRAVITY
    mppi_state = MPPIState(
        actions=jnp.full((horizon, 2), hover),
        key=jax.random.key(seed=0),
    )
    mppi_params = MPPIParameters(
        num_rollouts=256,
        num_iters=1,
    )
    mppi_dynamic_params = MPPIDynamicParameters(
        temp=jnp.array(2048.0),
        variance=jnp.array([0.01, 0.01]),
        dt=jnp.array(dt),
    )
    
    # x z theta vx vz w
    # Q = [30.0, 30.0, 50.0, 10.0, 10.0, 10.0]
    # QN = [30.0, 30.0, 50.0, 10.0, 10.0, 10.0]
    # R = [100.0, 100.0]
    # goal_state = [10.0, 1.0, 0.0, 1.0, 0.0, 0.0]


    def cost_fn(s: State, a: jnp.ndarray) -> jnp.ndarray:
        q = s.quadrotor_state
        pos_err = (q.x - goal[0]) ** 2 + (q.z - goal[1]) ** 2
        vel = q.vx**2 + q.vz**2
        ctrl = jnp.sum(a**2)
        angle = q.theta**2
        angular_vel = q.w**2
        return 100 * pos_err + 20.0 * angle # + 0.01 * vel #+ 10.0 * angular_vel + 100.0 * ctrl

    def terminal_cost_fn(s: State) -> jnp.ndarray:
        q = s.quadrotor_state
        pos_err = (q.x - goal[0]) ** 2 + (q.z - goal[1]) ** 2
        vel = q.vx**2 + q.vz**2
        angle = q.theta**2
        angular_vel = q.w**2
        return 100 * pos_err + 50.0 * angle + 50.0 * vel# + 10.0 * angular_vel

    xs, zs, thetas = [float(state.quadrotor_state.x)], [float(state.quadrotor_state.z)], [float(state.quadrotor_state.theta)]
    rollout_xs, rollout_zs = [], []  # per-step (num_rollouts, horizon)
    opt_rollout_xs, opt_rollout_zs = [], [] # per-step (horizon, )

    for _ in range(num_steps):
        optimized_actions, mppi_state, rollouts = mppi_compute_action(
            state,
            params,
            cost_fn,
            terminal_cost_fn,
            mppi_state,
            mppi_params,
            mppi_dynamic_params,
        )
        action = optimized_actions[0]
        print(f'First={action-hover}, Min={jnp.min(optimized_actions) - hover}, Max={jnp.max(optimized_actions) - hover}')

        # Generate full optimal trajectory
        _, trajs = mppi_rollout(state, optimized_actions, params, cost_fn, terminal_cost_fn, mppi_dynamic_params.dt)
        opt_rollout_xs.append(np.asarray(trajs.quadrotor_state.x))
        opt_rollout_zs.append(np.asarray(trajs.quadrotor_state.z))

        # Store counterfactual rollouts
        rollout_xs.append(np.asarray(rollouts.quadrotor_state.x))
        rollout_zs.append(np.asarray(rollouts.quadrotor_state.z))

        # Advance simulation
        state = step_state(state, action, params, dt)
        xs.append(float(state.quadrotor_state.x))
        zs.append(float(state.quadrotor_state.z))
        thetas.append(float(state.quadrotor_state.theta))

    for i in range(1, len(optimized_actions)):
        action = optimized_actions[i]
        state = step_state(state, action, params, dt)
        xs.append(float(state.quadrotor_state.x))
        zs.append(float(state.quadrotor_state.z))
        thetas.append(float(state.quadrotor_state.theta))



    xs = np.array(xs)
    zs = np.array(zs)
    thetas = np.array(thetas)

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
    # --- VISUALIZATION UPDATE ---
    # Add a line to represent the body of the quadrotor to visualize tilt
    (quad_body,) = ax.plot([], [], "-", color="black", linewidth=4)
    # Keep the dot to represent the center of mass
    (quad,) = ax.plot([], [], "o", color="tab:blue", markersize=8)
    
    title = ax.set_title("")
    ax.legend(loc="upper left")

    def update(i):
        trail.set_data(xs[: i + 1], zs[: i + 1])
        quad.set_data([xs[i]], [zs[i]])
        
        # --- DRONE TILT CALCULATION ---
        # Set a visual width for the drone's body
        drone_width = quad_params.rotor_dist * 2 + quad_params.rotor_size

        # Calculate the X and Z offsets from the center based on theta
        # If your theta definition is shifted by 90 degrees, swap sin/cos or add np.pi/2 to thetas[i]
        dx = (drone_width / 2.0) * np.cos(thetas[i])
        dz = (drone_width / 2.0) * np.sin(thetas[i])
        
        # Update the line segment representing the drone body
        quad_body.set_data([xs[i] - dx, xs[i] + dx], [zs[i] - dz, zs[i] + dz])

        if i < len(rollout_xs):
            rx = rollout_xs[i]
            rz = rollout_zs[i]
            for k, line in enumerate(rollout_lines):
                line.set_data(rx[k], rz[k])
        
        if i < len(opt_rollout_xs):
            rx = opt_rollout_xs[i]
            rz = opt_rollout_zs[i]
            opt_rollout_line.set_data(rx, rz)

        title.set_text(f"step {i}/{num_steps}   pos=({xs[i]:+.2f}, {zs[i]:+.2f})")
        
        # Make sure to return quad_body so FuncAnimation knows to update it
        return [trail, quad, quad_body, title, *rollout_lines, opt_rollout_line]

    _anim = FuncAnimation(
        # frames = num_steps + 1
        fig, update, frames=len(xs), interval=dt * 1000, blit=False, repeat=False
    )
    plt.show()

if __name__ == "__main__":
    main()
