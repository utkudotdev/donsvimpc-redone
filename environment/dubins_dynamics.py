from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

DUBINS_ACTION_DIM = 2

@register_dataclass
@dataclass
class DubinsParameters:
    turn_rate_min: jnp.ndarray
    turn_rate_max: jnp.ndarray
    velocity_min: jnp.ndarray
    velocity_max: jnp.ndarray


@register_dataclass
@dataclass
class DubinsState:
    x: jnp.ndarray
    y: jnp.ndarray
    theta: jnp.ndarray

@partial(jax.jit, static_argnames=("num_substeps",))
def step_dubins(
    state: DubinsState,
    action: jnp.ndarray,
    params: DubinsParameters,
    dt: float,
    num_substeps: int = 10,
) -> DubinsState:
    """Propagate the quadrotor `num_substeps` times with substep `dt / num_substeps`.

    `action = [v, theta_dot]` are commanded thrusts for the two rotors.
    """
    sub_dt = dt / num_substeps
    v = jnp.clip(action[0], params.velocity_min, params.velocity_max)
    theta_dot = jnp.clip(action[1], params.turn_rate_min, params.turn_rate_max)

    def substep(s: DubinsState, _) -> tuple[DubinsState, None]:
        x_d = v * jnp.cos(s.theta)
        y_d = v * jnp.sin(s.theta)
        
        return (
            DubinsState(
                x=s.x + x_d * sub_dt,
                y=s.y + y_d * sub_dt,
                theta=s.theta + theta_dot * sub_dt,
            ),
            None,
        )

    next_state, _ = jax.lax.scan(substep, state, xs=None, length=num_substeps)
    return next_state

