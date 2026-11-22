# models/tfim.py


class TransverseFieldIsingModel:

    def __init__(
        self,
        n_qubits: int,
        J: float,
        h: float,
    ):
        self.n_qubits = n_qubits
        self.J = J
        self.h = h

    def build_hamiltonian(self):
        """
        Construct

        H =
            -J Σ Zi Zi+1
            -h Σ Xi

        with periodic boundary conditions.
        """

        hamiltonian = []

        for i in range(self.n_qubits):

            j = (i + 1) % self.n_qubits

            hamiltonian.append(
                ("ZZ", i, j, -self.J)
            )

        for i in range(self.n_qubits):

            hamiltonian.append(
                ("X", i, -self.h)
            )

        return hamiltonian