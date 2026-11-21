"""Participant tools for the PushQuantum challenge."""
import zixy.qubit.pauli as zqp
from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import qubit, rx, rz, cx


N = 20  # Number of Qubits
hx = 1.0

# ZZ
zz_terms = [f"(-1.0, Z{i} Z{i+1})" for i in range(N - 1)]

# Transversal magnetic field: hx * sum X_j
x_terms = [f"({hx}, X{i})" for i in range(N)]

hamiltonian_string = ", ".join(zz_terms + x_terms)

hamiltonian = zqp.RealTermSum.from_str(hamiltonian_string, N)
print(hamiltonian)

M = 16
a = 1.186
b = 0.077
c = 2.181
d = 0.469

f_values = []
J_values = []

for n in range(M):
    f_n = 1.0 / ((a + b * n) ** c)
    J_n = -(f_n / hx) * ((n + 0.5) / M) ** d

    f_values.append(f_n)
    J_values.append(J_n)

@guppy
def V(qreg: array[qubit, N], J: float, f: float) -> None:

    # 1. ZZ interaction:
    # exp(i J sum_<j,k> Z_j Z_k)

    for y in range(4):
        for x in range(5):

            i = y * 5 + x

            # Right neighbour (periodic)
            right = y * 5 + (x + 1) % 5

            cx(qreg[i], qreg[right])
            rz(qreg[right], -2.0 * J)
            cx(qreg[i], qreg[right])

            # Down neighbour (periodic)
            down = ((y + 1) % 4) * 5 + x

            cx(qreg[i], qreg[down])
            rz(qreg[down], -2.0 * J)
            cx(qreg[i], qreg[down])

    # 2. Transverse field:
    # exp(i f sum_j X_j)
    for j in range(N):
        rx(qreg[j], -2.0 * f)

@guppy
def evolution(qreg: array[qubit, 20], Js: array[float, 16], fs: array[float, 16]) -> None:
    for n in range(16):
        V(qreg, Js[n], fs[n])