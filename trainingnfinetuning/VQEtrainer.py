# main.py

import numpy as np

from configs.default_config import TFIMConfig
from execution.backend import (
    BackendType,
    GuppyBackendFactory,
)
from execution.measurement import (
    MeasurementResult,
)
from models.tfim import (
    TransverseFieldIsingModel,
)
from models.ansatz import (
    HardwareEfficientAnsatz,
)


class ThermalTFIMSolver:

    """
    Skeleton for a variational thermal
    transverse field Ising solver.

    Future algorithms:

    - VQA thermal states
    - Gibbs-state preparation
    - Purification methods
    - QITE
    - Neural quantum states
    """

    def __init__(
        self,
        config: TFIMConfig,
        backend_type: BackendType,
    ):

        self.config = config

        self.backend = (
            GuppyBackendFactory.create_backend(
                backend_type,
                config.n_qubits,
            )
        )

        self.model = (
            TransverseFieldIsingModel(
                config.n_qubits,
                config.J,
                config.h,
            )
        )

        self.ansatz = (
            HardwareEfficientAnsatz(
                config.n_qubits,
                config.ansatz_layers,
            )
        )

    def build_circuit(
        self,
        parameters,
    ):
        """
        Construct variational circuit.
        """

        circuit = ...

        self.ansatz.build(
            circuit,
            parameters,
        )

        return circuit

    def execute(
        self,
        parameters,
    ) -> MeasurementResult:
        """
        Execute a parameterized circuit.
        """

        circuit = self.build_circuit(
            parameters
        )

        counts = self.backend.run(
            circuit,
            shots=self.config.shots,
        )

        result = MeasurementResult(
            counts=counts
        )

        return result