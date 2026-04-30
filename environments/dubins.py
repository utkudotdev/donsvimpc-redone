import jax.numpy as jnp
from dynamics.dubins_dynamics import DubinsParameters
from dynamics.environment_dynamics import Parameters
from dynamics.obstacle_dynamics import ObstacleParameters, from_many


ENVIRONMENTS: dict[str, Parameters] = {
    "basic": Parameters(
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
    )
}


def get_environment_parameters(name: str) -> Parameters:
    return ENVIRONMENTS[name]
