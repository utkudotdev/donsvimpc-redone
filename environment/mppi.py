from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .environment_dynamics import Parameters, State, step_state
from .quadrotor_dynamics import QUADROTOR_ACTION_DIM


@register_dataclass
@dataclass
class MPPIState:
    actions: jnp.ndarray
    key: jax.Array


@register_dataclass
@dataclass(frozen=True)
class MPPIParameters:
    num_rollouts: int
    num_iters: int
    knot_scale: int = 5


@register_dataclass
@dataclass
class MPPIDynamicParameters:
    temp: jnp.ndarray
    variance: jnp.ndarray
    dt: jnp.ndarray


def mppi_rollout(
    state: State,
    actions: jnp.ndarray,
    params: Parameters,
    cost_fn,
    terminal_cost_fn,
    dt: float,
):
    """Roll out `actions` from `state`. Returns (total_cost, state_trajectory)."""

    def body(s, a):
        s_next = step_state(s, a, params, dt)
        return s_next, (cost_fn(s, a), s_next)

    final_state, (costs, traj) = jax.lax.scan(body, state, actions)
    return jnp.sum(costs) + terminal_cost_fn(final_state), traj


@partial(jax.jit, static_argnames=("cost_fn", "terminal_cost_fn", "mppi_params"))
def mppi_compute_action(
    state: State,
    params: Parameters,
    cost_fn,
    terminal_cost_fn,
    mppi_state: MPPIState,
    mppi_params: MPPIParameters,
    mppi_dynamic_params: MPPIDynamicParameters,
):
    """Compute next action of MPPI controller.

    Returns (action, new_mppi_state, rollouts) where `rollouts` is a batched
    `State` pytree with leading dims (num_rollouts, horizon) holding the state
    trajectory of every sampled rollout used in the final iteration.
    """

    horizon = mppi_state.actions.shape[0]
    std = jnp.sqrt(mppi_dynamic_params.variance)

    key, subkey = jax.random.split(mppi_state.key)

    # --- KNOT / ZERO-ORDER HOLD LOGIC ---
    # Calculate how many knots we need to cover the full horizon
    num_knots = (horizon + mppi_params.knot_scale - 1) // mppi_params.knot_scale
    
    # Sample noise for the knots only
    knot_noise = (
        jax.random.normal(
            subkey, (mppi_params.num_rollouts, num_knots, QUADROTOR_ACTION_DIM)
        )
        * std
    )
    
    # Expand the knots to the full horizon using a zero-order hold (repeat)
    noise = jnp.repeat(knot_noise, mppi_params.knot_scale, axis=1)
    
    # Trim back down to the exact horizon length 
    # (needed if horizon is not perfectly divisible by knot_scale)
    noise = noise[:, :horizon, :]
    # ------------------------------------

    def rollout(actions):
        return mppi_rollout(
            state, actions, params, cost_fn, terminal_cost_fn, mppi_dynamic_params.dt
        )

    def iter_step(nominal, _):
        perturbed = nominal[None] + noise
        costs, trajs = jax.vmap(rollout)(perturbed)
        beta = jnp.min(costs)
        weights = jnp.exp(-(costs - beta) / mppi_dynamic_params.temp)
        weights = weights / jnp.sum(weights)

        # Debugging
        ess = 1.0 / jnp.sum(weights ** 2)
        jax.debug.print("ESS={ess}", ess=ess)
        ########

        update = jnp.sum(weights[:, None, None] * noise, axis=0)
        return nominal + update, trajs

    optimized, trajs_per_iter = jax.lax.scan(
        iter_step, mppi_state.actions, None, length=mppi_params.num_iters
    )
    rollouts = jax.tree.map(lambda x: x[-1], trajs_per_iter)

    # action = optimized[0]
    shifted = jnp.concatenate([optimized[1:], optimized[-1:]], axis=0)
    return optimized, MPPIState(actions=shifted, key=key), rollouts
