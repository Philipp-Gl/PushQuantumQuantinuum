# execution/backend.py

from enum import Enum


class BackendType(Enum):

    LOCAL_NOISELESS = "local_noiseless"
    LOCAL_NOISY = "local_noisy"


class GuppyBackendFactory:

    @staticmethod
    def create_backend(
        backend_type: BackendType,
        n_qubits: int,
    ):
        """
        Returns a configured Guppy device.

        Replace placeholder code with actual Guppy API.
        """

        if backend_type == BackendType.LOCAL_NOISELESS:

            device = ...
            return device

        if backend_type == BackendType.LOCAL_NOISY:

            device = ...
            return device

        raise ValueError(
            f"Unknown backend: {backend_type}"
        )