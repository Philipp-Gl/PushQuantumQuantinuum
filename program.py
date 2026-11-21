import math

from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import (
    qubit, angle, h, z, ry, rx, rz, cx, measure
)

# Starting temperature
T_INIT = 2.375

# Excitation probability for H0 = -sum(X)
p = 1.0 / (1.0 + math.exp(2.0 / T_INIT))
theta = 2.0 * math.asin(math.sqrt(p))


@guppy
def main() -> array[qubit, 16]:

    q = array(qubit() for _ in range(16))

    # STEP 1: Prepare thermal mixture of H0
    for i in range(16):
        h(q[i])  # Prepare |+>

        # Randomly switch to |-> with probability p
        anc = qubit()
        ry(anc, angle(comptime(theta)))

        if measure(anc).read():
            z(q[i])

    # STEP 2: Adiabatic evolution (Trotterized)
    dt = 0.25

    for step in range(3):
        s = (step + 0.5) / 3.0

        # Transverse field: -sum(X)
        for i in range(16):
            rx(q[i], angle(-2.0 * dt))

        # Ising coupling: -s * sum(ZZ)
        for y in range(4):
            for x in range(4):
                i = x + 4 * y

                # Periodic right and down neighbors
                right = ((x + 1) % 4) + 4 * y
                down = x + 4 * ((y + 1) % 4)

                # ZZ interaction with right neighbor
                cx(q[i], q[right])
                rz(q[right], angle(-2.0 * dt * s))
                cx(q[i], q[right])

                # ZZ interaction with down neighbor
                cx(q[i], q[down])
                rz(q[down], angle(-2.0 * dt * s))
                cx(q[i], q[down])

    return q