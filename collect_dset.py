from dynamics.obstacle_dynamics import ObstacleState, from_many
from dynamics.environment_dynamics import State, Parameters, step_state
from dynamics.dubins_dynamics import DubinsState
from controllers.mppi import (
    MPPIDynamicParameters,
    MPPIParameters,
    MPPIState,
    mppi_compute_action,
)
import jax
import jax.numpy as jnp
from tasks.dubins import make_goal_reaching_task, compute_h_vector
from safety import cbf
from environments.dubins import get_environment_parameters


def sample_start_state(key: jnp.ndarray, p: Parameters) -> State:
    N_OBSTACLES = len(p.obstacle_params.start_point)
    obstacles_key, x_key, y_key, v_key, theta_key = jax.random.split(key, 5)

    obstacle_states = []
    for key in jax.random.split(obstacles_key, N_OBSTACLES):
        alpha_key, forward_key = jax.random.split(key)
        state = ObstacleState(
            alpha=jax.random.uniform(alpha_key, minval=0, maxval=1.0),
            forward=jax.random.uniform(forward_key) > 0.5,
        )
        obstacle_states.append(state)

    return State(
        dubins_state=DubinsState(
            x=jax.random.uniform(x_key, minval=p.x_min, maxval=p.x_max),
            y=jax.random.uniform(y_key, minval=p.y_min, maxval=p.y_max),
            v=jax.random.uniform(
                v_key,
                minval=p.dubins_params.velocity_min,
                maxval=p.dubins_params.velocity_max,
            ),
            theta=jax.random.uniform(theta_key, minval=0, maxval=2 * jnp.pi),
        ),
        obstacle_state=from_many(*obstacle_states),
    )


def get_mppi_controller(horizon: int, num_rollouts: int, temp: float, variance: list):
    mppi_state = MPPIState(
        actions=jnp.full((horizon, 2), fill_value=0.0),
        key=jax.random.key(seed=0),
    )
    mppi_params = MPPIParameters(
        num_rollouts=num_rollouts,
        num_iters=1,
    )
    mppi_dynamic_params = MPPIDynamicParameters(
        temp=jnp.array(temp),
        variance=jnp.array(variance),
    )
    return mppi_state, mppi_params, mppi_dynamic_params


def rollout_state_with_mppi(
    state: State,
    params: Parameters,
    dt: float,
    max_rollout_length: int,
    cost_fn,
    terminal_cost_fn,
    h_fn,
    horizon: int,
    num_rollouts: int,
    temp: float,
    variance: list,
):
    mppi_state, mppi_params, mppi_dynamics_params = get_mppi_controller(
        horizon, num_rollouts, temp, variance
    )

    def _step(carry, _):
        state, mppi_state = carry
        optimized_actions, mppi_state, _ = mppi_compute_action(
            state,
            params,
            cost_fn,
            terminal_cost_fn,
            mppi_state,
            mppi_params,
            mppi_dynamics_params,
            dt,
        )
        action = optimized_actions[0]
        new_state = step_state(state, action, params, dt)

        return (
            (new_state, mppi_state),
            state,
        )

    _, states = jax.lax.scan(
        _step, init=(state, mppi_state), xs=None, length=max_rollout_length
    )

    hs = jax.vmap(h_fn, in_axes=(0, None))(states, params)

    return (states, hs)


NUM_ROLLOUTS = 10000
NUM_ROLLOUTS_PER_BATCH = 1024
MAX_ROLLOUT_LENGTH = 256
DT = 0.05

MPPI_HORIZON = 20
MPPI_NUM_ROLLOUTS = 128
MPPI_TEMP = 1.0
MPPI_VARIANCE = [1.0, 1.0]


def main():
    key = jax.random.key(seed=0)
    parameters = get_environment_parameters("basic")

    rollout_keys = jax.random.split(key, NUM_ROLLOUTS)

    def _inner(key: jnp.ndarray):
        start_key, goal_key = jax.random.split(key)
        start_state = sample_start_state(start_key, parameters)
        goal = jax.random.uniform(
            goal_key,
            shape=(3,),
            minval=jnp.array(
                [
                    parameters.x_min,
                    parameters.y_min,
                    parameters.dubins_params.velocity_min,
                ]
            ),
            maxval=jnp.array(
                [
                    parameters.x_max,
                    parameters.y_max,
                    parameters.dubins_params.velocity_max,
                ]
            ),
        )

        cbf_alpha = 0.92
        vio_cost = 1000.0
        cost_fn, terminal_cost_fn, _ = make_goal_reaching_task(goal)
        h_vio_fn = cbf.cbf_violation(compute_h_vector, DT)
        cost_fn, terminal_cost_fn = cbf.embed_cbf_violation(
            h_vio_fn, cost_fn, terminal_cost_fn, cbf_alpha, vio_cost
        )

        states, hs = rollout_state_with_mppi(
            start_state,
            parameters,
            DT,
            MAX_ROLLOUT_LENGTH,
            cost_fn,
            terminal_cost_fn,
            compute_h_vector,
            MPPI_HORIZON,
            MPPI_NUM_ROLLOUTS,
            MPPI_TEMP,
            MPPI_VARIANCE,
        )

        return (states, hs)

    states, hs = jax.lax.map(_inner, rollout_keys, batch_size=NUM_ROLLOUTS_PER_BATCH)
    jnp.savez("dset.npz", states=states, hs=hs)


if __name__ == "__main__":
    main()
