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


@partial(jax.jit, static_argnames=("num_substeps",))
def step_state(
    state: State,
    action: jnp.ndarray,
    params: Parameters,
    dt: float,
    num_substeps: int = 10,
) -> State:

    quadrotor_state = step_quadrotor(
        state.quadrotor_state, action, params.quadrotor_params, num_substeps
    )

    obstacle_state = step_state(state.obstacle_state, params.quadrotor_params, dt)

    return State(quadrotor_state, obstacle_state)
