import equinox as eqx
import jax
import jax.numpy as jnp

class NCBF(eqx.Module):
    """ 
    NCBF that works on relative states.
    """

    layers: list

    def __init__(self, key: jnp.ndarray, relative_state_dim: int, h_vector_dim: int, hidden_size: int = 256):
        super().__init__()
        # self.relative_state_dim = relative_state_dim
        # self.h_vector_dim = h_vector_dim
        # self.hidden_size = hidden_size
        
        keys = jax.random.split(key, num=3)
        self.layers = [
            eqx.nn.Linear(relative_state_dim, hidden_size, use_bias=True, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, h_vector_dim, use_bias=True, key=keys[2])
        ]


    def __call__(self, relative_state: jnp.ndarray):
        """ NCBF takes a 'relative' representation of the state . """
        x = relative_state
        for layer in self.layers:
            x = layer(x)
        return x
    
def compute_ncbf_target(ncbf: NCBF, gamma: float, h_t: jnp.ndarray, x_t1: jnp.ndarray):
    # NOTE: there can be a separate gamma per h
    # jax.debug.print("{x}, {y}", x=((1 - gamma) * h_t ).shape, y=(gamma * ncbf(x_t1)).shape)
    V_target = (1 - gamma) * h_t + gamma * ncbf(x_t1)
    return jnp.max(jnp.array([ h_t, V_target ]), axis=0)


def compute_ncbf_loss(ncbf: NCBF, gamma: float, x_t: jnp.ndarray, h_t: jnp.ndarray, x_t1: jnp.ndarray): 
    V = ncbf(x_t)
    V_target = compute_ncbf_target(ncbf, gamma, h_t, x_t1)
    return jnp.sum((V - V_target)**2)
