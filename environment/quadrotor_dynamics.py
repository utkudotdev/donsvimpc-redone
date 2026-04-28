from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

GRAVITY = 9.81
GROUND_EFFECT_DENOM_MIN = 0.1
QUADROTOR_ACTION_DIM = 2


@register_dataclass
@dataclass
class QuadrotorParameters:
    mass: jnp.ndarray
    rotor_dist: jnp.ndarray  # half the distance between rotors
    moi: jnp.ndarray
    rho: jnp.ndarray  # ground effect coefficient
    rotor_size: jnp.ndarray  # rotor propeller length
    thrust_min: jnp.ndarray
    thrust_max: jnp.ndarray


@register_dataclass
@dataclass
class QuadrotorState:
    x: jnp.ndarray
    z: jnp.ndarray
    theta: jnp.ndarray
    vx: jnp.ndarray
    vz: jnp.ndarray
    w: jnp.ndarray


@partial(jax.jit, static_argnames=("num_substeps",))
def step_quadrotor(
    state: QuadrotorState,
    action: jnp.ndarray,
    params: QuadrotorParameters,
    dt: float,
    num_substeps: int = 10,
) -> QuadrotorState:
    """Propagate the quadrotor `num_substeps` times with substep `dt / num_substeps`.

    `action = [F1_in, F2_in]` are commanded thrusts for the two rotors.
    """
    sub_dt = dt / num_substeps
    f1_in = action[0]
    f2_in = action[1]

    def substep(s: QuadrotorState, _) -> tuple[QuadrotorState, None]:
        z1 = s.z - params.rotor_dist * jnp.sin(s.theta)
        z2 = s.z + params.rotor_dist * jnp.sin(s.theta)

        denom_1 = jnp.maximum(
            GROUND_EFFECT_DENOM_MIN,
            1 - params.rho * (params.rotor_size / (4 * z1)) ** 2,
        )
        denom_2 = jnp.maximum(
            GROUND_EFFECT_DENOM_MIN,
            1 - params.rho * (params.rotor_size / (4 * z2)) ** 2,
        )

        f1 = f1_in / denom_1
        f2 = f2_in / denom_2

        f = jnp.clip(f1 + f2, params.thrust_min, params.thrust_max)
        tau = params.rotor_dist * (f2 - f1)

        x_dd = -f * jnp.sin(s.theta) / params.mass
        z_dd = f * jnp.cos(s.theta) / params.mass - GRAVITY
        theta_dd = tau / params.moi

        return (
            QuadrotorState(
                x=s.x + s.vx * sub_dt,
                z=s.z + s.vz * sub_dt,
                theta=s.theta + s.w * sub_dt,
                vx=s.vx + x_dd * sub_dt,
                vz=s.vz + z_dd * sub_dt,
                w=s.w + theta_dd * sub_dt,
            ),
            None,
        )

    next_state, _ = jax.lax.scan(substep, state, xs=None, length=num_substeps)
    return next_state
