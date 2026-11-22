"""PushQuantum 2026: Granet-Dreyer adiabatic preparation, submission interface.

Use from the OFFICIAL hackathon repository/environment, e.g.:

    uv run --locked compile adiabatic_guppy_submission.py --basis z
    uv run --locked local-sim adiabatic_guppy_submission.py --basis x --seed 42
    uv run --locked local-sim adiabatic_guppy_submission.py --basis y --seed 43
    uv run --locked local-sim adiabatic_guppy_submission.py --basis z --seed 44

Then create a scorer-compatible JSON (TEST ONLY: the temperature is not calibrated):

    uv run --locked python adiabatic_guppy_submission.py pack --temperature 2.375 \
        --x adiabatic_guppy_submission.x.local.json \
        --y adiabatic_guppy_submission.y.local.json \
        --z adiabatic_guppy_submission.z.local.json \
        --output submission_test_2.375.json
    uv run --locked score.py submission_test_2.375.json

Physics retained from adiabatic_guppy.py:
- 4x4 periodic transverse-field Ising model, H=-sum(ZZ)-sum(X)
- same 16-step Granet-Dreyer gate coefficients and gate ordering
- per-shot independent |+> / |-> input states giving the Gibbs mixture for H0

For compatibility, shot labels are pre-sampled in *Python before compilation*
and packed into 64-bit lookup masks, exactly as in the official tilted-product
example. Runtime get_current_shot() selects the configuration. This is why the
module can expose the required zero-argument main() without extra qubits.

IMPORTANT: BETA0=1.0 is ONLY a test parameter. It does not correspond to
any verified challenge target temperature. Calibration is a separate step.
"""

import argparse
import json
import math
import random
from pathlib import Path

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, output
from guppylang.std.qsystem import zz_phase
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import qubit, h, z, rx
from pushquantum.program import current_run


# ---------------------------------------------------------------------------
# 1. Fixed physical model and unchanged paper-inspired gate schedule
# ---------------------------------------------------------------------------
LX, LY, HX = 4, 4, -1.0
N = LX * LY
M = 16
A, B, C, D = 1.186, 0.077, 2.181, 0.469 

# This is the only physics parameter we are testing here.
BETA0 = 1.0                      # initial inverse temperature; NOT final 1/T
SHOTS = 128                     # required per basis
BASIS_SEEDS = {"x": 1001, "y": 1002, "z": 1003}
TARGET_TEMPERATURES = (0.01, 1.1875, 2.375, 4.0)

EDGES = []
for y in range(LY):
    for x in range(LX):
        i = y * LX + x
        EDGES.append((i, y * LX + (x + 1) % LX))
        EDGES.append((i, ((y + 1) % LY) * LX + x))

F = [1.0 / (A + B * k) ** C for k in range(M)]
J = [-F[k] / HX * ((k + 0.5) / M) ** D for k in range(M)]
# Guppy angles are half-turns, 1.0 = pi radians.
X_ANGLES = [-2.0 * f / math.pi for f in F]
ZZ_ANGLES = [-2.0 * j / math.pi for j in J]


# ---------------------------------------------------------------------------
# 2. Same thermal product-state sampling as in the original Python host code
#    (now pre-sampled and packed to work with zero-argument main()).
# ---------------------------------------------------------------------------
def random_thermal_input(beta0: float, rng: random.Random) -> list[bool]:
    """True selects |->, False selects |+> for one lattice site."""
    # H0 = -sum X, so P(|->) = (1 - tanh(beta0)) / 2.
    p_minus = (1.0 - math.tanh(beta0)) / 2.0
    return [rng.random() < p_minus for _ in range(N)]


def labels_for_basis(basis: str) -> list[int]:
    """128 independent sampled input bitstrings, deterministic per basis."""
    rng = random.Random(BASIS_SEEDS[basis])
    labels = []
    for _ in range(SHOTS):
        minus = random_thermal_input(BETA0, rng)
        labels.append(sum((1 << i) for i, bit in enumerate(minus) if bit))
    return labels


def pack_masks(labels: list[int]) -> list[tuple[int, int]]:
    """For each of 16 sites, store its 128 shot labels in two signed int64s."""
    if len(labels) != SHOTS:
        raise ValueError("Expected exactly 128 pre-sampled preparation labels")
    masks = []
    for i in range(N):
        halves = []
        for half in range(2):
            unsigned = sum(((labels[half * 64 + s] >> i) & 1) << s
                           for s in range(64))
            halves.append(unsigned if unsigned < (1 << 63)
                          else unsigned - (1 << 64))
        masks.append((halves[0], halves[1]))
    return masks


# The official compile/local-sim/nexus-emulator tools set this run configuration
# before importing the submission module.
_run = current_run.get()
BASIS = _run.basis if _run is not None else "z"
SHOT_OFFSET = _run.shot_offset if _run is not None else 0
if BASIS not in BASIS_SEEDS:
    raise ValueError("Unsupported measurement basis: " + BASIS)
if _run is not None and _run.shots != SHOTS:
    raise ValueError("This pre-sampled program requires exactly 128 shots")
MASKS = pack_masks(labels_for_basis(BASIS))

@guppy
def prepare_qubit(q: qubit, shot: int, low: int, high: int) -> int:
    if shot < 64:
        bit = (low >> shot) & 1
    else:
        bit = (high >> (shot - 64)) & 1
    h(q)
    if bit == 1:
        z(q)
    return bit

@guppy.comptime
def prepare_for_current_shot(qs: array[qubit, 16]) -> int:
    """Prepare a new pure X-product branch chosen by the device shot number."""
    shot = get_current_shot() - comptime(SHOT_OFFSET)
    label = 0
    for i in range(16):
        # Python indexing is resolved at compile time (H2-compatible).
        low, high = MASKS[i]
        bit = prepare_qubit(qs[i], shot, low, high)
        label = label | (bit << i)
    return label


# ---------------------------------------------------------------------------
# 3. Original adiabatic evolution: UNCHANGED gate ordering and coefficients
# ---------------------------------------------------------------------------
@guppy.comptime
def step(qs: array[qubit, 16], zz_angle: float, x_angle: float) -> None:
    # V(J_k, f_k) = exp(i f_k sum X) exp(i J_k sum ZZ)
    # Apply ZZ first and X second.
    for i, j in EDGES:
        zz_phase(qs[i], qs[j], angle(zz_angle))
    for i in range(N):
        rx(qs[i], angle(x_angle))


@guppy.comptime
def inverse_step(qs: array[qubit, 16], zz_angle: float, x_angle: float) -> None:
    # Kept for future mirror-circuit / noise diagnostics; not used for scoring.
    for i in range(N):
        rx(qs[i], angle(-x_angle))
    for i, j in reversed(EDGES):
        zz_phase(qs[i], qs[j], angle(-zz_angle))


@guppy.comptime
def full_evolution(qs: array[qubit, 16]) -> None:
    for zz_angle, x_angle in zip(ZZ_ANGLES, X_ANGLES):
        step(qs, zz_angle, x_angle)


@guppy.comptime
def mirror_evolution(qs: array[qubit, 16]) -> None:
    first_half = list(zip(ZZ_ANGLES[: M // 2], X_ANGLES[: M // 2]))
    for zz_angle, x_angle in first_half:
        step(qs, zz_angle, x_angle)
    for zz_angle, x_angle in reversed(first_half):
        inverse_step(qs, zz_angle, x_angle)


# ---------------------------------------------------------------------------
# 4. REQUIRED HACKATHON ENTRYPOINT: zero arguments, return 16 live qubits
# ---------------------------------------------------------------------------
@guppy
def main() -> array[qubit, 16]:
    qs = array(qubit() for _ in range(16))
    label = prepare_for_current_shot(qs)
    full_evolution(qs)
    # Diagnostics under distinct tags; scorer will ignore them.
    output("shot", get_current_shot() - comptime(SHOT_OFFSET))
    output("label", label)
    return qs  # Official CLI applies X/Y/Z readout and emits "bits".


# ---------------------------------------------------------------------------
# 5. Optional local JSON converter, called only with: python THIS_FILE.py pack
#    Does NOT generate or invent measurements; it repackages 3 emulator outputs.
# ---------------------------------------------------------------------------
def read_basis_results(path: Path, basis: str) -> list[list[int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("basis") != basis.lower():
        raise ValueError(f"{path}: expected basis {basis.lower()!r}")
    if document.get("shots") != SHOTS:
        raise ValueError(f"{path}: expected exactly {SHOTS} shots")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != SHOTS:
        raise ValueError(f"{path}: expected 128 result records")

    expected_labels = labels_for_basis(basis.lower())
    ordered = [None] * SHOTS
    for shot_record in results:
        values = dict(shot_record)
        number = int(values["shot"])
        label = int(values["label"])
        bits = values["bits"]
        if not 0 <= number < SHOTS or ordered[number] is not None:
            raise ValueError(f"{path}: invalid or duplicate shot {number}")
        if label != expected_labels[number]:
            raise ValueError(f"{path}: preparation label mismatch in shot {number}")
        if not isinstance(bits, list) or len(bits) != N or any(bit not in (0, 1, False, True) for bit in bits):
            raise ValueError(f"{path}: expected 16 binary measurement values")
        ordered[number] = [int(bit) for bit in bits]
    if any(row is None for row in ordered):
        raise ValueError(f"{path}: missing measurement shots")
    return ordered


def pack_submission() -> None:
    parser = argparse.ArgumentParser(description="Pack three actual emulator results for score.py")
    parser.add_argument("command", choices=("pack",))
    parser.add_argument("--temperature", type=float, required=True, choices=TARGET_TEMPERATURES)
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--y", type=Path, required=True)
    parser.add_argument("--z", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    submission = {"temperature": args.temperature,
                  "X": read_basis_results(args.x, "X"),
                  "Y": read_basis_results(args.y, "Y"),
                  "Z": read_basis_results(args.z, "Z")}
    args.output.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: 128 x 16 bits for each of X, Y, Z")
    print(f"WARNING: beta0={BETA0} is NOT calibrated to T={args.temperature}; format test only.")


if __name__ == "__main__":
    pack_submission()
