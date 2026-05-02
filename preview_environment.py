import argparse

import jax
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from dynamics.obstacle_dynamics import ObstacleState, step_obstacle
from environments.dubins import ENVIRONMENTS, get_environment_parameters


def get_arguments():
    parser = argparse.ArgumentParser(description="Preview a Dubins environment.")
    parser.add_argument(
        "--env",
        type=str,
        default="basic",
        choices=sorted(ENVIRONMENTS.keys()),
        help="Environment name from environments/dubins.py.",
    )
    parser.add_argument("--num-steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.05)
    return parser.parse_args()


def rollout_obstacles(
    initial_state: ObstacleState, params, dt: float, num_steps: int
) -> ObstacleState:
    step_all = jax.vmap(step_obstacle, in_axes=(0, 0, None))

    def body(state, _):
        next_state = step_all(state, params, dt)
        return next_state, state

    final_state, traj = jax.lax.scan(body, initial_state, None, length=num_steps)
    return jax.tree.map(
        lambda pre, fin: jnp.concatenate([pre, fin[None]], axis=0), traj, final_state
    )


def main():
    args = get_arguments()
    params = get_environment_parameters(args.env)
    obstacle_params = params.obstacle_params

    num_obstacles = obstacle_params.radius.shape[0]
    initial_obs_state = ObstacleState(
        alpha=jnp.zeros(num_obstacles),
        forward=jnp.ones(num_obstacles, dtype=bool),
    )

    states = rollout_obstacles(initial_obs_state, obstacle_params, args.dt, args.num_steps)

    positions = jax.vmap(jax.vmap(ObstacleState.position), in_axes=(0, None))(
        states, obstacle_params
    )
    obs_xs = np.asarray(positions[..., 0])
    obs_ys = np.asarray(positions[..., 1])
    radii = np.asarray(obstacle_params.radius)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(float(params.x_min), float(params.x_max))
    ax.set_ylim(float(params.y_min), float(params.y_max))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"environment: {args.env}")

    obstacle_bodies = []
    for k in range(num_obstacles):
        body = patches.Circle(
            (obs_xs[0, k], obs_ys[0, k]),
            radius=float(radii[k]),
            color="tab:red",
            alpha=0.5,
        )
        ax.add_patch(body)
        obstacle_bodies.append(body)

    def update(i):
        for k, body in enumerate(obstacle_bodies):
            body.set_center((obs_xs[i, k], obs_ys[i, k]))
        return obstacle_bodies

    _anim = FuncAnimation(
        fig,
        update,
        frames=obs_xs.shape[0],
        interval=args.dt * 1000,
        blit=False,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
