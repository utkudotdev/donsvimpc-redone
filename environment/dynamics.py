from dataclasses import dataclass
from typing import Self
from jax.tree_util import register_dataclass
from jax.numpy import jnp


@register_dataclass
@dataclass
class QuadrotorParameters:
    mass: float
    rotor_dist: float
    moi: float
    rho: float  # ground effect coefficient
    rotor_size: float


@register_dataclass
@dataclass
class QuadrotorState:
    x: float
    z: float
    theta: float
    vx: float
    vz: float
    w: float

    def step(self) -> Self:
        pass
