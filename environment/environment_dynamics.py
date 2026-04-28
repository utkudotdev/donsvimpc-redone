from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .obstacle_dynamics import ObstacleParameters, ObstacleState, step_obstacle
from .quadrotor_dynamics import QuadrotorParameters, QuadrotorState, step_quadrotor


@register_dataclass
@dataclass
class State:
    quadrotor_state: QuadrotorState
    obstacle_state: ObstacleState


@register_dataclass
@dataclass
class Parameters:
    quadrotor_params: QuadrotorParameters
    obstacle_params: ObstacleParameters

    x_min: jnp.ndarray
    x_max: jnp.ndarray
    y_min: jnp.ndarray
    y_max: jnp.ndarray


@partial(jax.jit, static_argnames=("num_substeps",))
def step_state(
    state: State,
    action: jnp.ndarray,
    params: Parameters,
    dt: float,
    num_substeps: int = 10,
) -> State:

    quadrotor_state = step_quadrotor(
        state.quadrotor_state, action, params.quadrotor_params, dt, num_substeps
    )

    obstacle_state = step_obstacle(
        state.obstacle_state, params.obstacle_params, dt, num_substeps
    )

    return State(quadrotor_state, obstacle_state)
