from selene_sim.backends.bundled_error_models import DepolarizingErrorModel

# Name of the @guppy function 
from program import main

noise = DepolarizingErrorModel(
    random_seed=42,
    p_1q=0.0001,   # Single-qubit gate error
    p_2q=0.001,    # Two-qubit gate error
    p_meas=0.002,  # Measurement error
    p_init=0.0001  # Initialization error
)

result = (
    main.emulator(n_qubits=17)
    .with_error_model(noise)
    .with_shots(128)
    .run()
)

print(result.collated_counts())