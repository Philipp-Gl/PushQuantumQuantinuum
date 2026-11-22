# models/ansatz.py


from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import (
    qubit, angle, h, z, ry, rx, rz, cx, measure
)
from guppylang.std.qsystem.helios import (
    zz_phase
)
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
            for l in range(self.layers):
                for i in range(self.N):
                    sy = Symbol(f"ry_{l}_{i}")
                    sx = Symbol(f"rx_{l}_{i}")
                    circ.Ry(sy, i)
                    circ.Rx(sx, i)
                    symbols.extend([sy, sx])
                for i, j in self.edges:
                    szz = Symbol(f"zz_{l}_{i}_{j}")
                    circ.ZZPhase(szz, i, j)  # Native R_ZZ on Quantinuum H2
                    symbols.append(szz)
            return circ, symbols

    
    def to_guppy(self, fn_name: str = "u_theta"):
            # Bind trained numbers onto a copy so self.circ stays reusable
            bound_circ = self.circ.copy()
            bound_circ.symbol_substitution(
                {s: float(self.theta[idx]) for idx, s in enumerate(self.symbols)}
            )
            return guppy.load_pytket(fn_name, bound_circ)
    

        