##
from collections import Counter

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.debug import state_output
from guppylang.std.quantum import (
    qubit, h, x, y, z, cx, cz,
    measure_array, collect_measurements
)

# ============================================================
# CONFIGURATION
# ============================================================

N = 3
SHOTS = 128
DECIMALS = 4
SHOW_ZERO_STATES = True
SIMULATE = True

# ============================================================
# YOUR QUANTUM CIRCUIT
# ============================================================

@guppy
def experiment() -> None:

    q = array(qubit() for _ in range(3))

    state_output("initial", q)

    # -------- YOUR CIRCUIT STARTS HERE --------

    h(q[0])
    cx(q[0], q[1])

    # Optional state checkpoint
    state_output("checkpoint", q)

    cx(q[1], q[2])

    # -------- YOUR CIRCUIT ENDS HERE ----------

    state_output("final", q)

    # Measure all qubits
    bits = collect_measurements(measure_array(q))
    output("bits", bits)


# ============================================================
# STATE DISPLAY
# ============================================================

def show_state(state, title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    terms = []

    for i, amp in enumerate(state):
        if abs(amp) > 1e-10:
            terms.append(
                f"({amp.real:.4f}{amp.imag:+.4f}j)"
                f"|{i:0{N}b}>"
            )

    print(" |psi> = " + " + ".join(terms))

    print("\nBasis     Amplitude             Probability")

    for i, amp in enumerate(state):
        if not SHOW_ZERO_STATES and abs(amp) < 1e-10:
            continue

        print(
            f"|{i:0{N}b}>"
            f"    {amp.real:+.{DECIMALS}f}"
            f"{amp.imag:+.{DECIMALS}f}j"
            f"         {abs(amp)**2:.4f}"
        )


# ============================================================
# RUN AND DISPLAY
# ============================================================

if __name__ == "__main__":

    experiment.check()

    results = (
        experiment.emulator(n_qubits=N)
        .with_seed(42)
        .with_shots(SHOTS if SIMULATE else 1)
        .run()
    )

    states = results.partial_state_dicts()[0]

    show_state(states["initial"].as_single_state(),
               "INITIAL STATE")

    show_state(states["checkpoint"].as_single_state(),
               "CHECKPOINT")

    show_state(states["final"].as_single_state(),
               "FINAL STATE")

    if SIMULATE:
        counts = Counter(
            "".join(str(int(b)) for b in shot.as_dict()["bits"])
            for shot in results
        )

        print(f"\nMEASUREMENTS ({SHOTS} SHOTS)")
        print("State     Count     Probability")

        for state, count in sorted(counts.items()):
            print(f"|{state}>    {count:>5}      {count/SHOTS:.4f}")