'''
Quantinuum challenge thermal state preparation 
PushQuantum Hackathon 2026
Variational Quantum Eigensolver approach
'''

import os
import numpy as np
from sympy import Symbol
from pytket import Circuit
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit, h, x, ry, discard
from guppylang.std.qsystem import measure_and_reset
from VQE import VariationalAnsatz, get_periodic_4x4_edges

EDGES = get_periodic_4x4_edges()
N_LAYERS = 2  # Match the layer count used in offline training


def build_bound_pytket_circuit(theta: np.ndarray) -> Circuit:
    circ = Circuit(16)
    idx = 0
    for _ in range(N_LAYERS):
        for i in range(16):
            circ.Ry(float(theta[idx]), i)
            circ.Rx(float(theta[idx + 1]), i)
            idx += 2
        for i, j in EDGES:
            circ.ZZPhase(float(theta[idx]), i, j)
            idx += 1
    return circ


# 2. Load trained parameters at compile/import time based on TARGET_T
TARGET_T = os.environ.get("TARGET_T", "4.0")
TARGET_T = os.environ.get("TARGET_T", "0.01")
PARAM_FILE = f"params_T{TARGET_T}.npz"

if os.path.exists(PARAM_FILE):
    data = np.load(PARAM_FILE)
    trained_theta = data["theta"]
    # For high-T beta-VQE: 16 Ry angles (in radians) to sample initial bitstrings
    # For T=0.01 pure ground state: all zeros (prepares |+...+)
    init_ry_angles = data["init_ry_angles"].tolist()
    use_hadamard_basis = bool(data["use_hadamard_basis"])
    use_hadamard_basis = False
else:
    print(f"Could not find params_T{TARGET_T}.npz using initial |+++> state")
    trained_theta = np.zeros(N_LAYERS * (32 + 32), dtype=float)
    init_ry_angles = [0.0] * 16
    use_hadamard_basis = True

# 3. Compile unitary U_theta into a Guppy function at module level
u_theta = guppy.load_pytket("u_theta", build_bound_pytket_circuit(trained_theta))


@guppy
def main() -> array[qubit, 16]:
    qs = array(qubit() for _ in range(16))
    angles = comptime(init_ry_angles)

    # Step A: Sample initial classical bitstring |x> per shot via 1 reusable ancilla
    # (Consuming `anc` with `measure` frees it immediately, keeping peak qubits = 17)
    anc = qubit()
    for i in range(16):
        ry(anc, angle(angles[i]))
        if measure_and_reset(anc):
            x(qs[i])
    discard(anc)


    # Step B: Rotate |0>/|1> into |+>/|-> basis if trained with X-basis warm start
    if comptime(use_hadamard_basis):
        for i in range(16):
            h(qs[i])

    # Step C: Apply trained variational unitary U_theta
    u_theta(qs)

    # Return unmeasured system qubits in lattice order q = x + 4*y