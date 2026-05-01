from dynamics.environment_dynamics import State, Parameters
from dynamics.obstacle_dynamics import ObstacleState

import jax.numpy as jnp
import jax


def _collided_with_boundary_or_obstacles(s: State, p: Parameters):
    return jnp.max(compute_h_vector(s, p)) >= 0


def make_goal_reaching_task(goal: jnp.ndarray, goal_eps=0.1):

    def task_cost_fn(s: State, a: jnp.ndarray, p: Parameters) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        ctrl = jnp.sum(a**2)
        return pos_err + 0.01 * ctrl

    def task_terminal_cost_fn(s: State, p: Parameters) -> jnp.ndarray:
        d = s.dubins_state
        pos_err = (d.x - goal[0]) ** 2 + (d.y - goal[1]) ** 2
        return 100 * pos_err


    def task_done_fn(s: State, p: Parameters) -> jnp.ndarray:
        dubins_position = jnp.array([ s.dubins_state.x, s.dubins_state.y, s.dubins_state.v ])
        done = jnp.linalg.norm(dubins_position - goal) <= goal_eps

        terminated = _collided_with_boundary_or_obstacles(s, p)

        return jnp.array([ done, terminated ])


    return task_cost_fn, task_terminal_cost_fn, task_done_fn


def compute_h_vector(s: State, p: Parameters):
    h_boundary = jnp.max(jnp.array([ 
        s.dubins_state.x - p.x_max, 
        p.x_min - s.dubins_state.x,
        s.dubins_state.y - p.y_max, 
        p.y_min - s.dubins_state.y]))

    dubins_position = s.dubins_state.position()
    obstacle_positions = jax.vmap(ObstacleState.position)(s.obstacle_state, p.obstacle_params)

    signed_distances = jnp.linalg.norm(dubins_position - obstacle_positions, axis=1) - p.obstacle_params.radius
    signed_distance = jnp.min(signed_distances)

    h_obstacles = -signed_distance

    return jnp.array([
        h_obstacles, 
        h_boundary
    ])