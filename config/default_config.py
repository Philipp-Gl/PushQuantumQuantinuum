# configs/default_config.py

from dataclasses import dataclass

@dataclass
class TFIMConfig:

    n_qubits: int = 24

    # Hamiltonian parameters
    J: float = 1.0
    h: float = 0.5

    # Thermal state parameters
    beta: float = 5.0

    # Circuit
    ansatz_layers: int = 3

    # Measurements
    shots: int = 10_000

    # Backend
    use_noise: bool = False
    seed: int | None = None