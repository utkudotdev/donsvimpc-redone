from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .obstacle_dynamics import ObstacleParameters, ObstacleState, step_obstacle
from .quadrotor_dynamics import QuadrotorParameters, QuadrotorState, step_quadrotor
from .dubins_dynamics import DubinsParameters, DubinsState, step_dubins


@register_dataclass
@dataclass
class State:
    # quadrotor_state: QuadrotorState
    dubins_state: DubinsState
    obstacle_state: ObstacleState


@register_dataclass
@dataclass
class Parameters:
    # quadrotor_params: QuadrotorParameters
    dubins_params: DubinsParameters
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

    # quadrotor_state = step_quadrotor(
    #     state.quadrotor_state, action, params.quadrotor_params, dt, num_substeps
    # )

    dubins_state = step_dubins(
        state.dubins_state, action, params.dubins_params, dt, num_substeps
    )

    obstacle_state = jax.vmap(step_obstacle, in_axes=(0, 0, None, None))(
        state.obstacle_state, params.obstacle_params, dt, num_substeps
    )

    return State(
        # quadrotor_state,
        dubins_state,
        obstacle_state)


def make_relative_dubins_state(s: State, p: Parameters) -> jnp.ndarray:
    """
    Input to neural-CBF contains the state. To make the neural-CBF more
    general, we pass in a 'relative state'.

    """
    obstacle_relative_pos = (
        jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params)
        - s.dubins_state.position()
    )
    boundary_max_relative_pos = (
        jnp.array([p.x_max, p.y_max]) - s.dubins_state.position()
    )
    boundary_min_relative_pos = s.dubins_state.position() - jnp.array(
        [p.x_min, p.y_min]
    )

    obstacle_abs_vel = jax.vmap(ObstacleState.velocity)(
        s.obstacle_state, p.obstacle_params
    )

    # TODO: if we randomize car dynamics we would have to include that here
    # Right now, dynamics and velocity constraints are baked into NCBF
    return jnp.concatenate(
        [
            obstacle_relative_pos.flatten(),
            boundary_max_relative_pos,
            boundary_min_relative_pos,
            jnp.atleast_1d(s.dubins_state.v),
            jnp.atleast_1d(s.dubins_state.theta),
            obstacle_abs_vel.flatten(),
        ]
    )