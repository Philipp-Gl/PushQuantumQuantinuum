'''
Quantinuum challenge thermal state preparation 
PushQuantum Hackathon 2026
Variational Quantum Eigensolver approach
'''

from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import (qubit, angle, h, z, ry, rx, rz, cx, measure)
from guppylang.std.qsystem.helios import (zz_phase)
import jax
import jax.numpy as jnp
import optax
from sympy import Symbol
from pytket import Circuit
from pytket.extensions.qujax import tk_to_qujax



def get_periodic_4x4_edges() -> list[tuple[int, int]]:
        edges = []
        for y in range(4):
            for x_coord in range(4):
                q = x_coord + 4 * y
                q_right = ((x_coord + 1) % 4) + 4 * y
                q_down = x_coord + 4 * ((y + 1) % 4)
                edges.append((q, q_right))
                edges.append((q, q_down))
        return edges

class VariationalAnsatz():
    """this class defines a circuit """


    def __init__(self, N: int, edges: list[tuple[int, int]], layers: int, seed: int = 0):
            self.N = N
            self.edges = edges
            self.layers = layers

            # 1. Build symbolic circuit template once
            self.circ, self.symbols = self._build_ansatz()
            self.symbol_map = {s: idx for idx, s in enumerate(self.symbols)}

            # 2. Compile to a differentiable JAX function: (theta, init_state) -> state_tensor
            self.param_to_state = tk_to_qujax(self.circ, symbol_map=self.symbol_map)

            # 3. Initialize the numerical parameter vector theta
            key = jax.random.PRNGKey(seed)
            self.theta = 0.01 * jax.random.normal(key, shape=(len(self.symbols),))

    def _build_ansatz(self) -> tuple[Circuit, list[Symbol]]:
            circ = Circuit(self.N)
            symbols = []