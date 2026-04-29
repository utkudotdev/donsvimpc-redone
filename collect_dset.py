from environment.obstacle_dynamics import ObstacleParameters, ObstacleState, from_many
from environment.environment_dynamics import State, Parameters, step_state
from environment.dubins_dynamics import DubinsParameters, DubinsState
from controllers.mppi import MPPIDynamicParameters, MPPIParameters, MPPIState, mppi_compute_action
import jax
import jax.numpy as jnp
import functools as ft

def get_dubins_parameters() -> DubinsParameters:
    return DubinsParameters(
        turn_rate_min=jnp.array(-1.0),
        turn_rate_max=jnp.array(1.0),
        velocity_min=jnp.array(-1.0),
        velocity_max=jnp.array(1.0),
        acceleration_min=jnp.array(-2.0),
        acceleration_max=jnp.array(2.0),
    )


def get_obstacle_parameters() -> ObstacleParameters:
    return from_many(
        ObstacleParameters(
            radius=jnp.array(1.5),
            speed=jnp.array(0.0),
            start_point=jnp.array([4.0, 4.0]),
            end_point=jnp.array([0.0, 0.0]),
        ),
        ObstacleParameters(
            radius=jnp.array(0.5),
            speed=jnp.array(0.6),
            start_point=jnp.array([5.7, 4.0]),
            end_point=jnp.array([5.7, 2.25]),
        ),
        ObstacleParameters(
            radius=jnp.array(0.5),
            speed=jnp.array(0.0),
            start_point=jnp.array([5.7, 4.0]),
            end_point=jnp.array([5.7, 0.0]),
        ),
    )


def get_parameters() -> Parameters:
    x_min, x_max = 0.0, 8.0
    y_min, y_max = 2.25, 4.0
    return Parameters(
        dubins_params=get_dubins_parameters(),
        obstacle_params=get_obstacle_parameters(),
        x_min=jnp.array(x_min),
        x_max=jnp.array(x_max),
        y_min=jnp.array(y_min),
        y_max=jnp.array(y_max),
    )


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
        obstacle_state=from_many(obstacle_states),
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


def rollout_state_with_mppi(state: State, params: Parameters, dt: float, max_rollout_length: int, cost_fn, terminal_cost_fn, done_fn, h_fn, horizon: int, num_rollouts: int, temp: float, variance: list): 
    mppi_state, mppi_params, mppi_dynamics_params = get_mppi_controller(horizon, num_rollouts, temp, variance)
    
    def _step(carry, _):
        state, mppi_state = carry
        done = done_fn(state).any()

        optimized_actions, mppi_state, _ = mppi_compute_action(
            state,
            params,
            cost_fn,
            terminal_cost_fn,
            mppi_state,
            mppi_params,
            mppi_dynamics_params,
            dt
        )
        action = optimized_actions[0]
        new_state = step_state(state, action, params, dt)

        return (new_state, mppi_state), nstate,

    _, states = jax.lax.scan(_step, init=(state, mppi_state), xs=None, length=max_rollout_length)

    hs = jax.vmap(h_fn)(states)

    return (states, hs)
    

NUM_ROLLOUTS = 10000
NUM_ROLLOUTS_PER_BATCH = 1024


def main():
    
    triples = []
    key = jax.random.key(seed=0)

    parameters = get_parameters()

    keys = jax.random.split(key, NUM_ROLLOUTS)


    key, sample_key = jax.random.split(key)
    start_state = sample_start_state(sample_key)


if __name__ == "__main__":
    pass

