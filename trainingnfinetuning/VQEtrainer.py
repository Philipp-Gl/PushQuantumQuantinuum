# trainingnfinetuning/VQEtrainer.py

import optax 
import jax 
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path
from typing import Callable, Optional

# Add project root to sys.path so imports work regardless of execution directory
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from physical_models.circuits.VQE.VQE import VariationalAnsatz, get_periodic_4x4_edges



class VariationalTrainer:
    """Trainer for Variational Quantum Eigensolver (VQE) and variational thermal state preparation.

    Optimizes the variational parameters of a VariationalAnsatz using JAX and Optax.
    """

    def __init__(
        self,
        ansatz: VariationalAnsatz,
        T: float = 0.0,
        loss_fn: Optional[Callable[[jnp.ndarray], float]] = None,
    ):
        self.ansatz = ansatz
        self.T = float(T)
        self.loss_fn = loss_fn

    def train_step(
        self,
        loss_fn: Optional[Callable[[jnp.ndarray], float]] = None,
        opt_state: Optional[optax.OptState] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
    ) -> tuple[jnp.ndarray, optax.OptState]:
        """Perform a single optimization step using JAX and Optax.

        Args:
            loss_fn: Differentiable loss function mapping theta -> scalar loss.
                If None, uses self.loss_fn.
            opt_state: Current optimizer state from Optax.
            optimizer: Optax optimizer instance.

        Returns:
            Tuple of (loss_value, new_opt_state).
        """
        fn = loss_fn if loss_fn is not None else self.loss_fn
        if fn is None:
            raise ValueError(
                "No loss_fn provided to train_step and self.loss_fn is not set."
            )
        if optimizer is None:
            raise ValueError("An Optax optimizer must be provided to train_step.")
        if opt_state is None:
            opt_state = optimizer.init(self.ansatz.theta)

        loss_val, grads = jax.value_and_grad(fn)(self.ansatz.theta)
        updates, opt_state = optimizer.update(grads, opt_state, self.ansatz.theta)
        self.ansatz.theta = optax.apply_updates(self.ansatz.theta, updates)
        return loss_val, opt_state

    def train(
        self,
        steps: int = 300,
        learning_rate: float = 0.02,
        optimizer: Optional[optax.GradientTransformation] = None,
        print_every: int = 50,
        loss_fn: Optional[Callable[[jnp.ndarray], float]] = None,
    ) -> list[float]:
        """Run the optimization loop for a given number of steps.

        Args:
            steps: Total number of optimization iterations.
            learning_rate: Learning rate 
            optimizer: Optional custom Optax optimizer. If None, defaults to Adam.
            print_every: log training progress every print_every
            loss_fn: Optional override for the loss function.

        Returns:
            List of float loss values recorded at each step.
        """
        target_loss_fn = loss_fn if loss_fn is not None else self.loss_fn
        if target_loss_fn is None:
            raise ValueError(
                "No loss_fn provided. Please provide loss_fn to __init__ or train()."
            )

        if optimizer is None:
            optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(self.ansatz.theta)

        # Wrap loss_fn with @jax.jit so XLA compiles the statevector simulation
        jit_loss_fn = jax.jit(target_loss_fn)

        loss_history: list[float] = []

        for step in range(steps):
            loss_val, opt_state = self.train_step(jit_loss_fn, opt_state, optimizer)
            val = float(loss_val)
            loss_history.append(val)
            if print_every > 0 and (step % print_every == 0 or step == steps - 1):
                print(f"Step {step:4d}: loss = {val:.6f}")

        return loss_history

    def build_tfim_energy_loss(
        self,
        J: float = 1.0,
        h: float = 1.0,
    ) -> Callable[[jnp.ndarray], float]:
        """Construct an energy expectation value loss function for the TFIM.

        H = -J sum_<i,j> Z_i Z_j - h sum_i X_i

        Args:
            J: Coupling constant for ZZ interactions.
            h: Transverse field strength for X terms.

        Returns:
            Differentiable function theta -> <H>
        """
        import qujax

        paulis = []
        qubits = []
        coeffs = []

        # ZZ interactions along edges
        for i, j in self.ansatz.edges:
            paulis.append(["Z", "Z"])
            qubits.append([i, j])
            coeffs.append(-float(J))

        # Transverse field X on each qubit
        for i in range(self.ansatz.N):
            paulis.append(["X"])
            qubits.append([i])
            coeffs.append(-float(h))

        expectation_fn = qujax.get_statetensor_to_expectation_func(
            paulis, qubits, coeffs
        )

        def energy_loss(theta: jnp.ndarray) -> float:
            state = self.ansatz.param_to_state(theta)
            return expectation_fn(state)

        return energy_loss

    def save_parameters(
        self,
        filepath: str | Path,
        init_ry_angles: Optional[list[float]] = None,
        use_hadamard_basis: bool = True,
        **extra_data,
    ) -> None:
        """Save optimized parameters to an .npz file (compatible with VQEguppy.py).

        Args:
            filepath: Destination file path (e.g. 'params_T4.0.npz').
            init_ry_angles: 16 angles for sampling classical bitstrings in beta-VQE.
            use_hadamard_basis: Whether to prepare in Hadamard basis for X-field warm start.
            **extra_data: Additional fields to store in the .npz archive.
        """
        if init_ry_angles is None:
            init_ry_angles = [0.0] * self.ansatz.N

        np.savez(
            filepath,
            theta=np.array(self.ansatz.theta),
            init_ry_angles=np.array(init_ry_angles, dtype=float),
            use_hadamard_basis=bool(use_hadamard_basis),
            T=float(self.T),
            **extra_data,
        )
        print(f"Saved parameters to {filepath}")


if __name__ == "__main__":
    # Build edges from lattice
    temperature = 0.01
    n_qubits = 16
    # Demonstration on a 4x4-qubit lattice
    edges = get_periodic_4x4_edges()
    ansatz = VariationalAnsatz(N=n_qubits, edges=edges, layers=2, seed=42)

    
    trainer = VariationalTrainer(ansatz=ansatz, T=temperature)
    loss_fn = trainer.build_tfim_energy_loss(J=1.0, h=1.0)
    trainer.loss_fn = loss_fn

    print("Initial energy loss:", float(loss_fn(ansatz.theta)))
    print("Training 4x4-qubit lattice VQE for 50 steps...")
    history = trainer.train(steps=50, learning_rate=0.05, print_every=10)
    print("Final energy loss:", float(loss_fn(ansatz.theta)))

    #Write the theta parameters to a file
    param_file = f"params_T{temperature:g}.npz"
    trainer.save_parameters(
        filepath=param_file,
        init_ry_angles=[0.0] * n_qubits,
        use_hadamard_basis=True,
    )
    print(f"Parameters saved to '{param_file}', ready for Guppy compilation.")
    