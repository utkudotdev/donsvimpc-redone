from environment.environment_dynamics import State, Parameters
from environment.obstacle_dynamics import ObstacleState

import jax.numpy as jnp
import jax


def make_goal_reaching_task(goal: jnp.ndarray):

    def task_cost_fn(s: State, a: jnp.ndarray, p: Parameters) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        ctrl = jnp.sum(a**2)
        return pos_err + 0.01 * ctrl

    def task_terminal_cost_fn(s: State, p: Parameters) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        return 100 * pos_err


    return task_cost_fn, task_terminal_cost_fn


def compute_h_vector(s: State, p: Parameters):
    h_boundary = jnp.max(jnp.array([ 
        s.dubins_state.x - p.x_max, 
        p.x_min - s.dubins_state.x,
        s.dubins_state.y - p.y_max, 
        p.y_min - s.dubins_state.y]))

    dubins_position = jnp.array([ s.dubins_state.x, s.dubins_state.y ])
    obstacle_positions = jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params)

    signed_distances = jnp.linalg.norm(dubins_position - obstacle_positions, axis=1) - p.obstacle_params.radius
    signed_distance = jnp.min(signed_distances)

    h_obstacles = -signed_distance

    return jnp.array([
        h_obstacles, 
        h_boundary
    ])