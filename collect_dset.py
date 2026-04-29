from environment.obstacle_dynamics import ObstacleParameters, ObstacleState, from_many
from environment.environment_dynamics import State, Parameters
from environment.dubins_dynamics import DubinsParameters, DubinsState
from environment.dubins_dynamics import DubinsState
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
        acceleration_max=jnp.array(2.0)
    )

def get_obstacle_parameters() -> ObstacleParameters:
    return from_many(
        ObstacleParameters(radius=jnp.array(1.5), speed=jnp.array(0.0), start_point=jnp.array([4.0, 4.0]), end_point=jnp.array([0.0, 0.0])), 
        ObstacleParameters(radius=jnp.array(0.5), speed=jnp.array(0.6), start_point=jnp.array([5.7, 4.0]), end_point=jnp.array([5.7, 2.25])),
        ObstacleParameters(radius=jnp.array(0.5), speed=jnp.array(0.0), start_point=jnp.array([5.7, 4.0]), end_point=jnp.array([5.7, 0.0])))

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
    keys = jax.random.split(key, 20)

    return State(
        dubins_state=DubinsState(
            theta=jax.random.uniform(keys[3], minval=0, maxval=2 * jnp.pi),
        ),
        obstacle_state=from_many(
            ObstacleState( 
                alpha=jax.random.uniform(keys[1], minval=0, maxval=1.0),
                forward=jax.random.uniform(keys[2]) >= 0.5
            )
        )
    )


NUM_SAMPLES = 1024

def main():
    triples = []
    key = jax.random.key(seed=0)

    parameters = Parameters

    sample_count = 0
    while sample_count < NUM_SAMPLES:
        key, sample_key = jax.random.split(key)
        start_state = sample_start_state(sample_key)

        



if __name__ == '__main__':
    pass