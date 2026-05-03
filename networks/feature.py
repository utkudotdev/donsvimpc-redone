from dynamics.environment_dynamics import State, Parameters
from dynamics.obstacle_dynamics import ObstacleState
import jax.numpy as jnp
import jax
from jaxlie import SE2


def make_dubins_features(s: State, p: Parameters) -> jnp.ndarray:
    """
    Input to neural-CBF contains the state. To make the neural-CBF perform better
    and be invariant to global obstacle positions, we create features from this state.
    """

    obstacle_pos = jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params)
    obstacle_vel = jax.vmap(ObstacleState.velocity)(s.obstacle_state, p.obstacle_params)

    car_tf = SE2.from_xy_theta(s.dubins_state.x, s.dubins_state.y, s.dubins_state.theta)
    world_to_car_tf = car_tf.inverse()

    obstacle_pos_car = world_to_car_tf @ obstacle_pos
    obstacle_vel_car = world_to_car_tf.rotation() @ obstacle_vel

    boundary_max_relative_pos = (
        jnp.array([p.x_max, p.y_max]) - s.dubins_state.position()
    )
    boundary_min_relative_pos = s.dubins_state.position() - jnp.array(
        [p.x_min, p.y_min]
    )

    theta_obs = jnp.array(
        [jnp.cos(s.dubins_state.theta), jnp.sin(s.dubins_state.theta)]
    )

    return jnp.concatenate(
        [
            jnp.atleast_1d(s.dubins_state.v),
            theta_obs.flatten(),
            obstacle_pos_car.flatten(),
            obstacle_vel_car.flatten(),
            p.obstacle_params.radius,
            boundary_max_relative_pos.flatten(),
            boundary_min_relative_pos.flatten(),
        ]
    )

    # return jnp.concatenate(
    #     [
    #         s.dubins_state.position(),
    #         jnp.atleast_1d(s.dubins_state.v),
    #         jnp.atleast_1d(s.dubins_state.theta),
    #         jax.vmap(ObstacleState.position)(
    #             s.obstacle_state, p.obstacle_params
    #         ).flatten(),
    #         jax.vmap(ObstacleState.velocity)(
    #             s.obstacle_state, p.obstacle_params
    #         ).flatten(),
    #         p.obstacle_params.radius,
    #     ]
    # )

    # obstacle_relative_pos = (
    #     jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params)
    #     - s.dubins_state.position()
    # )
    # boundary_max_relative_pos = (
    #     jnp.array([p.x_max, p.y_max]) - s.dubins_state.position()
    # )
    # boundary_min_relative_pos = s.dubins_state.position() - jnp.array(
    #     [p.x_min, p.y_min]
    # )
    #
    # obstacle_abs_vel = jax.vmap(ObstacleState.velocity)(
    #     s.obstacle_state, p.obstacle_params
    # )
    #
    # theta_obs = jnp.array(
    #     [jnp.cos(s.dubins_state.theta), jnp.sin(s.dubins_state.theta)]
    # )
    #
    # # TODO: if we randomize car dynamics we would have to include that here
    # # Right now, dynamics and velocity constraints are baked into NCBF
    # return jnp.concatenate(
    #     [
    #         obstacle_relative_pos.flatten(),
    #         boundary_max_relative_pos,
    #         boundary_min_relative_pos,
    #         jnp.atleast_1d(s.dubins_state.v),
    #         jnp.atleast_1d(s.dubins_state.theta),
    #         obstacle_abs_vel.flatten(),
    #     ]
    # )
