from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
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
    alpha: jnp.ndarray
    forward: jnp.ndarray  # bool array; traced through jit

    def position(self, params: ObstacleParameters) -> jnp.ndarray:
        return self.alpha * params.end_point + (1 - self.alpha) * params.start_point

    def velocity(self, params: ObstacleParameters) -> jnp.ndarray:
        path_delta = params.end_point - params.start_point
        path_norm = jnp.linalg.norm(path_delta)
        direction = path_delta / path_norm
        sign = jnp.where(self.forward, 1.0, -1.0)
        return direction * params.speed * sign


@partial(jax.jit, static_argnames=("num_substeps",))
def step_obstacle(
    state: ObstacleState,
    params: ObstacleParameters,
    dt: float,
    num_substeps: int = 10,
) -> ObstacleState:
    """Linear obstacle dynamics: ping-pong along the segment start_point → end_point."""

    path_delta = params.end_point - params.start_point
    path_norm = jnp.linalg.norm(path_delta)

    sub_dt = dt / num_substeps
    dalpha = sub_dt * params.speed / path_norm

    def substep(s: ObstacleState, _) -> tuple[ObstacleState, None]:
        step = jnp.where(s.forward, dalpha, -dalpha)
        new_alpha = s.alpha + step
        out_of_bounds = (new_alpha < 0) | (new_alpha > 1)
        new_forward = jnp.where(out_of_bounds, ~s.forward, s.forward)
        new_alpha = jnp.clip(new_alpha, 0.0, 1.0)
        return ObstacleState(alpha=new_alpha, forward=new_forward), None

    next_state, _ = jax.lax.scan(substep, state, xs=None, length=num_substeps)
    return next_state
