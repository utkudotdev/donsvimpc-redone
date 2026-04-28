from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class ObstacleParameters:
    radius: jnp.ndarray
    speed: jnp.ndarray
    start_point: jnp.ndarray
    end_point: jnp.ndarray


@register_dataclass
@dataclass
class ObstacleState:
    x: jnp.ndarray
    z: jnp.ndarray
    vx: jnp.ndarray
    vz: jnp.ndarray

    @property
    def position(self) -> jnp.ndarray:
        return jnp.array([self.x, self.z])

    @property
    def velocity(self) -> jnp.ndarray:
        return jnp.array([self.vx, self.vz])


@partial(jax.jit, static_argnames=("num_substeps",))
@checkify.checkify
def step_obstacle(
    state: ObstacleState, params: ObstacleParameters, dt: float, num_substeps: int = 10
):
    """Linear obstacle dynamics"""
    path_delta = params.end_point - params.start_point
    path_norm = jnp.linalg.norm(path_delta)
    checkify.check(jnp.all(path_norm) > 1e-5, "path norm")

    path_norm_vec = path_delta / path_norm
    twod_obstacle_velocity = state.velocity * path_norm_vec

    new_position = state.position + dt * twod_obstacle_velocity

    # 2. Boundary Detection using Projection
    # Vector from start to current position
    start_to_obs = obstacles - obstacle_paths[:, 0]

    # Dot product: (A dot B) / |B|^2 gives the progress ratio 't'
    # path_norms is |B|, so we square it for |B|^2
    progress = jnp.sum(start_to_obs * path_deltas, axis=1) / (
        jnp.squeeze(path_norms) ** 2 + 1e-8
    )

    # 3. Identify who crossed the line
    past_end = progress > 1.0
    past_start = progress < 0.0
    crossed = past_end | past_start

    # 4. Snap positions to the boundaries if they overshot
    # We use [:, None] to broadcast (N,) booleans to (N, 2) coordinates
    obstacles = jnp.where(past_end[:, None], obstacle_paths[:, 1], obstacles)
    obstacles = jnp.where(past_start[:, None], obstacle_paths[:, 0], obstacles)

    # 5. Flip the scalar velocity for those who crossed
    obstacle_velocities = jnp.where(crossed, -obstacle_velocities, obstacle_velocities)

    return obstacles, obstacle_velocities
