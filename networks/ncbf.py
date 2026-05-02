import equinox as eqx
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Callable, Any
import json

from dynamics.environment_dynamics import State, Parameters, make_relative_dubins_state
import optax


class NCBFNetwork(eqx.Module):
    """
    NCBF network that works on relative states.
    """

    layers: list
    relative_state_dim: int
    h_vector_dim: int
    hidden_size: int

    def __init__(
        self,
        key: jnp.ndarray,
        relative_state_dim: int,
        h_vector_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.relative_state_dim = relative_state_dim
        self.h_vector_dim = h_vector_dim
        self.hidden_size = hidden_size

        keys = jax.random.split(key, num=3)
        self.layers = [
            eqx.nn.Linear(relative_state_dim, hidden_size, use_bias=True, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(hidden_size, h_vector_dim, use_bias=True, key=keys[2]),
        ]

    def __call__(self, relative_state: jnp.ndarray):
        """NCBF takes a 'relative' representation of the state ."""
        x = relative_state
        for layer in self.layers:
            x = layer(x)
        return x


class NCBF(eqx.Module):

    h_fn: Callable # = eqx.field(static=True)
    ncbf_network: NCBFNetwork

    def __call__(self, state: State, params: Parameters):
        relative_state = make_relative_dubins_state(state, params)
        V = self.ncbf_network(relative_state)
        h = self.h_fn(state, params)
        return jnp.max(jnp.array([ h, V ]), axis=0)


def compute_ncbf_target(ncbf: NCBFNetwork, gamma: float, h_t: jnp.ndarray, x_t1: jnp.ndarray):
    # NOTE: there can be a separate gamma per h
    # jax.debug.print("{x}, {y}", x=((1 - gamma) * h_t ).shape, y=(gamma * ncbf(x_t1)).shape)
    V_target = (1 - gamma) * h_t + gamma * ncbf(x_t1)
    return jnp.max(jnp.array([h_t, V_target]), axis=0)


def compute_ncbf_loss(
    ncbf: NCBFNetwork, gamma: float, x_t: jnp.ndarray, h_t: jnp.ndarray, x_t1: jnp.ndarray
):
    V = ncbf(x_t)
    V_target = compute_ncbf_target(ncbf, gamma, h_t, x_t1)
    return jnp.sum((V - V_target) ** 2)


def save_checkpoint(checkpoint_dir: Path, model: NCBFNetwork, epoch: int, opt_state, opt_name: str, opt_params: dict[str, Any], args: dict = {}):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / 'ncbf.eqx'
    info_path = checkpoint_dir / 'metadata.json'

    with open(state_path, "wb") as f:
        eqx.tree_serialise_leaves(f, (model, opt_state, epoch))

    with open(info_path, "w") as f:
        print('dumping json')
        json.dump({
            'relative_state_dim': model.relative_state_dim,
            'h_vector_dim': model.h_vector_dim,
            'hidden_size': model.hidden_size,
            'optimizer': opt_name,
            'opt_params': opt_params,
            'args': args
        }, f, default=str)


def load_checkpoint(checkpoint_dir: Path) -> tuple[NCBFNetwork, optax.GradientTransformationExtraArgs, optax.OptState, int]:
    state_path = checkpoint_dir / 'ncbf.eqx'
    info_path = checkpoint_dir / 'metadata.json'

    with open(info_path, "r") as f:
        metadata = json.load(f)
        relative_state_dim = metadata['relative_state_dim']
        h_vector_dim = metadata['h_vector_dim']
        hidden_size = metadata['hidden_size']
        optimizer_name = metadata['optimizer']
        optimizer_params = metadata['opt_params']

    model_template = NCBFNetwork(
        key=jax.random.key(0),
        relative_state_dim=relative_state_dim,
        h_vector_dim=h_vector_dim,
        hidden_size=hidden_size
    )

    optimizer = getattr(optax, optimizer_name)(**optimizer_params)
    opt_state_template = optimizer.init(eqx.filter(model_template, eqx.is_array))

    with open(state_path, "rb") as f:
        model, opt_state, epoch = eqx.tree_deserialise_leaves(f, (model_template, opt_state_template, 0))
    
    return model, optimizer, opt_state, epoch
