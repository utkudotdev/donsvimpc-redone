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
)
from environment.obstacle_dynamics import ObstacleParameters, ObstacleState
from environment.quadrotor_dynamics import (
    GRAVITY,
    QuadrotorParameters,
    QuadrotorState,
)


def main():
    quad_params = QuadrotorParameters(
        mass=jnp.array(1.0),
        rotor_dist=jnp.array(0.1),
        moi=jnp.array(0.01),
        rho=jnp.array(0.0),
        rotor_size=jnp.array(0.1),
        thrust_min=jnp.array(0.0),
        thrust_max=jnp.array(20.0),
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
        key=jax.random.PRNGKey(0),
    )
    mppi_params = MPPIParameters(
        num_rollouts=256,
        num_iters=2,
    )
    mppi_dynamic_params = MPPIDynamicParameters(
        temp=jnp.array(1.0),
        variance=jnp.array([4.0, 4.0]),
        dt=jnp.array(dt),
    )

    def cost_fn(s: State, a: jnp.ndarray) -> jnp.ndarray:
        q = s.quadrotor_state
        pos_err = (q.x - goal[0]) ** 2 + (q.z - goal[1]) ** 2
        vel = q.vx**2 + q.vz**2
        ctrl = jnp.sum(a**2)
        return 5.0 * pos_err + 0.05 * vel + 1e-3 * ctrl

    def terminal_cost_fn(s: State) -> jnp.ndarray:
        q = s.quadrotor_state
        return 50.0 * ((q.x - goal[0]) ** 2 + (q.z - goal[1]) ** 2)

    xs, zs, thetas = [float(state.quadrotor_state.x)], [float(state.quadrotor_state.z)], [float(state.quadrotor_state.theta)]
    rollout_xs, rollout_zs = [], []  # per-step (num_rollouts, horizon)

    for _ in range(num_steps):
        action, mppi_state, rollouts = mppi_compute_action(
            state,
            params,
            cost_fn,
            terminal_cost_fn,
            mppi_state,
            mppi_params,
            mppi_dynamic_params,
        )
        rollout_xs.append(np.asarray(rollouts.quadrotor_state.x))
        rollout_zs.append(np.asarray(rollouts.quadrotor_state.z))

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
    rollout_lines = ax.plot(
        np.zeros((mppi_params.num_rollouts, 0)).T,
        np.zeros((mppi_params.num_rollouts, 0)).T,
    )
    rollout_lines = []
    for _ in range(mppi_params.num_rollouts):
        (line,) = ax.plot([], [], "-", color="tab:orange", alpha=0.05, linewidth=0.6)
        rollout_lines.append(line)
    (quad,) = ax.plot([], [], "o", color="tab:blue", markersize=10)
    title = ax.set_title("")
    ax.legend(loc="upper left")

    def update(i):
        trail.set_data(xs[: i + 1], zs[: i + 1])
        quad.set_data([xs[i]], [zs[i]])
        if i < len(rollout_xs):
            rx = rollout_xs[i]
            rz = rollout_zs[i]
            for k, line in enumerate(rollout_lines):
                line.set_data(rx[k], rz[k])
        title.set_text(f"step {i}/{num_steps}   pos=({xs[i]:+.2f}, {zs[i]:+.2f})")
        return [trail, quad, title, *rollout_lines]

    _anim = FuncAnimation(
        fig, update, frames=num_steps + 1, interval=dt * 1000, blit=False, repeat=False
    )
    plt.show()


if __name__ == "__main__":
    main()
