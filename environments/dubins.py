import jax
import jax.numpy as jnp
from dynamics.dubins_dynamics import DubinsParameters
from dynamics.environment_dynamics import Parameters
from dynamics.obstacle_dynamics import ObstacleParameters, from_many
from typing import Callable


def make_randomized_environment(key: jnp.ndarray) -> Parameters:
    dubins_params = DubinsParameters(
        turn_rate_min=jnp.array(-1.0),
        turn_rate_max=jnp.array(1.0),
        velocity_min=jnp.array(-1.0),
        velocity_max=jnp.array(1.0),
        acceleration_min=jnp.array(-2.0),
        acceleration_max=jnp.array(2.0),
    )

    min_key, size_key, key = jax.random.split(key, 3)
    x_min, y_min = jax.random.uniform(min_key, shape=(2,), minval=-2.0, maxval=2.0)
    x_size, y_size = jax.random.uniform(size_key, shape=(2,), minval=0.5, maxval=4.0)
    x_max, y_max = x_min + x_size, y_min + y_size

    return Parameters(
        dubins_params=dubins_params,
        obstacle_params=from_many(
            ObstacleParameters(
                radius=jnp.array(0.75),
                speed=jnp.array(0.7),
                start_point=jnp.array([x_min, y_min]),
                end_point=jnp.array([x_max, y_max]),
            ),
        ),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


ENVIRONMENTS: dict[str, Callable[[jnp.ndarray], Parameters]] = {
    "basic": lambda _: Parameters(
        dubins_params=DubinsParameters(
            turn_rate_min=jnp.array(-1.0),
            turn_rate_max=jnp.array(1.0),
            velocity_min=jnp.array(-1.0),
            velocity_max=jnp.array(1.0),
            acceleration_min=jnp.array(-2.0),
            acceleration_max=jnp.array(2.0),
        ),
        obstacle_params=from_many(
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
        ),
        x_min=jnp.array(0.0),
        x_max=jnp.array(8.0),
        y_min=jnp.array(2.25),
        y_max=jnp.array(4.0),
    ),
    "single_obstacle_narrow": lambda _: Parameters(
        dubins_params=DubinsParameters(
            turn_rate_min=jnp.array(-1.0),
            turn_rate_max=jnp.array(1.0),
            velocity_min=jnp.array(-1.0),
            velocity_max=jnp.array(1.0),
            acceleration_min=jnp.array(-2.0),
            acceleration_max=jnp.array(2.0),
        ),
        obstacle_params=from_many(
            ObstacleParameters(
                radius=jnp.array(0.75),
                speed=jnp.array(0.0),
                start_point=jnp.array([4.0, 1.0]),
                end_point=jnp.array([0.0, 0.0]),
            ),
        ),
        x_min=jnp.array(0.0),
        x_max=jnp.array(8.0),
        y_min=jnp.array(0.0),
        y_max=jnp.array(2.0),
    ),
    "single_obstacle_narrow_moving": lambda _: Parameters(
        dubins_params=DubinsParameters(
            turn_rate_min=jnp.array(-1.0),
            turn_rate_max=jnp.array(1.0),
            velocity_min=jnp.array(-1.0),
            velocity_max=jnp.array(1.0),
            acceleration_min=jnp.array(-2.0),
            acceleration_max=jnp.array(2.0),
        ),
        obstacle_params=from_many(
            ObstacleParameters(
                radius=jnp.array(0.75),
                speed=jnp.array(0.7),
                start_point=jnp.array([7.0, 1.0]),
                end_point=jnp.array([1.0, 1.0]),
            ),
        ),
        x_min=jnp.array(0.0),
        x_max=jnp.array(8.0),
        y_min=jnp.array(0.0),
        y_max=jnp.array(2.0),
    ),
    "single_obstacle_narrow_moving_vertical": lambda _: Parameters(
        dubins_params=DubinsParameters(
            turn_rate_min=jnp.array(-1.0),
            turn_rate_max=jnp.array(1.0),
            velocity_min=jnp.array(-1.0),
            velocity_max=jnp.array(1.0),
            acceleration_min=jnp.array(-2.0),
            acceleration_max=jnp.array(2.0),
        ),
        obstacle_params=from_many(
            ObstacleParameters(
                radius=jnp.array(0.75),
                speed=jnp.array(0.7),
                start_point=jnp.array([1.0, 7.0]),
                end_point=jnp.array([1.0, 1.0]),
            ),
        ),
        x_min=jnp.array(0.0),
        x_max=jnp.array(2.0),
        y_min=jnp.array(0.0),
        y_max=jnp.array(8.0),
    ),
    "random": make_randomized_environment,
}


def make_environment(name: str, key: jnp.ndarray) -> Parameters:
    return ENVIRONMENTS[name](key)
