import jax
import jax.numpy as jnp

from dynamics.environment_dynamics import State, Parameters, step_state


def cbf_violation(h_fn, dt):
    def compute_cbf_violation(s: State, a: jnp.ndarray, p: Parameters, alpha: jnp.ndarray):
        h = jnp.max(h_fn(s, p))
        s_prime = step_state(s, a, p, dt)
        h_prime = jnp.max(h_fn(s_prime, p))
        return h_prime + (alpha - 1) * h
    return compute_cbf_violation


def embed_cbf_violation(h_vio_fn, cost_fn, terminal_cost_fn):
    def cost_fn_cbf(s: State, a: jnp.ndarray, p: Parameters):
        h_violation = h_vio_fn(s, a, p, alpha=0.95)
        cbf_cost = jnp.where(h_violation > 0, 100 * h_violation, 0.0)
        task_cost = cost_fn(s, a, p)
        return task_cost + 10 * cbf_cost    
        
    def terminal_cost_fn_cbf(s: State, p: Parameters):
        return terminal_cost_fn(s, p)
    
    return cost_fn_cbf, terminal_cost_fn_cbf

