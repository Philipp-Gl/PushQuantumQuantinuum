"""Guppy implementation of Granet-Dreyer (arXiv:2509.05206v2), Eq. 39-44.

Select MODE='paper' to reproduce the published 5x4 20-qubit gate schedule,
or MODE='challenge' to transfer the schedule to the 4x4 16-qubit
transverse-field Ising target used in the hackathon.

Requires guppylang >=1.0 (tested against documentation; not locally executed).
"""
import math
import random

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, output
from guppylang.std.qsystem import zz_phase
from guppylang.std.quantum import (
    qubit, h, z, rx, measure_array, collect_measurements,
)

# ---- Select the model ----
MODE = "challenge"  # "paper" = original 20-qubit hardware example
if MODE == "paper":
    LX, LY, HX = 5, 4, 2.0
elif MODE == "challenge":
    LX, LY, HX = 4, 4, -1.0
else:
    raise ValueError("MODE must be 'paper' or 'challenge'")

N = LX * LY
M = 16
A, B, C, D = 1.186, 0.077, 2.181, 0.469

# Each periodic undirected bond exactly once (right and down).
EDGES = []
for y in range(LY):
    for x in range(LX):
        i = y * LX + x
        EDGES.append((i, y * LX + (x + 1) % LX))
        EDGES.append((i, ((y + 1) % LY) * LX + x))

# Eqs. (42)-(43), computed by ordinary Python *before* Guppy compilation.
F = [1.0 / (A + B * k) ** C for k in range(M)]
J = [-F[k] / HX * ((k + 0.5) / M) ** D for k in range(M)]

# Guppy angle(float) counts half-turns (1.0 = pi radians).
# Rx(theta) = exp(-i theta X/2); ZZPhase(theta) = exp(-i theta ZZ/2).
# Thus e^(i f X) requires theta=-2 f, e^(i J ZZ) theta=-2 J.
X_ANGLES = [-2.0 * f / math.pi for f in F]
ZZ_ANGLES = [-2.0 * j / math.pi for j in J]


@guppy
# A classical input chooses the product eigenstate for THIS shot.
# minus[i] True means |- > on qubit i, False means |+>.
def prepare(minus: array[bool, comptime(N)]) -> array[qubit, comptime(N)]:
    qs = array(qubit() for _ in range(comptime(N)))
    for i in range(comptime(N)):
        h(qs[i])
        if minus[i]:
            z(qs[i])
    return qs


@guppy.comptime
def step(qs: array[qubit, comptime(N)], zz_angle: float, x_angle: float) -> None:
    # V(J_k, f_k) = exp(i f_k sum X) exp(i J_k sum ZZ)
    # Therefore apply ZZ first, X second.
    for i, j in EDGES:
        zz_phase(qs[i], qs[j], angle(zz_angle))
    for i in range(N):
        rx(qs[i], angle(x_angle))


@guppy.comptime
def inverse_step(qs: array[qubit, comptime(N)], zz_angle: float, x_angle: float) -> None:
    # V^dagger: inverse X first, inverse ZZ second.
    for i in range(N):
        rx(qs[i], angle(-x_angle))
    for i, j in reversed(EDGES):
        zz_phase(qs[i], qs[j], angle(-zz_angle))


@guppy.comptime
def full_evolution(qs: array[qubit, comptime(N)]) -> None:
    # Pass Python float values; do not index a Python list using a GuppyObject.
    for zz_angle, x_angle in zip(ZZ_ANGLES, X_ANGLES):
        step(qs, zz_angle, x_angle)


@guppy.comptime
def mirror_evolution(qs: array[qubit, comptime(N)]) -> None:
    first_half = list(zip(ZZ_ANGLES[: M // 2], X_ANGLES[: M // 2]))
    for zz_angle, x_angle in first_half:
        step(qs, zz_angle, x_angle)
    for zz_angle, x_angle in reversed(first_half):
        inverse_step(qs, zz_angle, x_angle)


@guppy
def forward_z(minus: array[bool, comptime(N)]) -> None:
    qs = prepare(minus)
    full_evolution(qs)
    output("bits", collect_measurements(measure_array(qs)))


@guppy
def forward_x(minus: array[bool, comptime(N)]) -> None:
    qs = prepare(minus)
    full_evolution(qs)
    for i in range(comptime(N)):
        h(qs[i])  # X-basis measurement
    output("bits", collect_measurements(measure_array(qs)))


@guppy
def mirror_x(minus: array[bool, comptime(N)]) -> None:
    qs = prepare(minus)
    mirror_evolution(qs)
    for i in range(comptime(N)):
        h(qs[i])  # X-basis measurement
    output("bits", collect_measurements(measure_array(qs)))


##############################################################################

def random_thermal_input(beta0: float) -> list[bool]:
    # Initial Hamiltonian H0 = sign(HX) * sum(X).
    # For the paper HX>0, the |- > state is favored.
    # For the challenge HX<0, the |+> state is favored.
    p_minus = (1.0 + math.copysign(1.0, HX) * math.tanh(beta0)) / 2.0
    return [random.random() < p_minus for _ in range(N)]


def read_bits(results) -> list[list[int]]:
    return [[int(c) for c in shot.as_dict()["bits"]] for shot in results]


def average_x(samples: list[list[int]]) -> float:
    return sum(sum(1 - 2 * bit for bit in bits) for bits in samples) / (N * len(samples))


def average_zz(samples: list[list[int]]) -> float:
    return sum(sum((1 - 2 * bits[i]) * (1 - 2 * bits[j])
                   for i, j in EDGES) for bits in samples) / (N * len(samples))


def binary_entropy(m: float) -> float:
    p = max(0.0, min(1.0, (1.0 + m) / 2.0))
    return -sum(q * math.log(q) for q in (p, 1 - p) if q > 0.0)


def estimate_final_beta(curve: list[tuple[float, float, float]]):
    """Return central-difference estimates (beta0, beta_f=dS/dE).

    curve must contain (beta0, energy_density, entropy_density) from
    separate experiments at increasing beta0. At least 3 points are needed.
    """
    estimates = []
    for k in range(1, len(curve) - 1):
        b0, _, _ = curve[k]
        delta_s = curve[k + 1][2] - curve[k - 1][2]
        delta_e = curve[k + 1][1] - curve[k - 1][1]
        if abs(delta_e) > 1e-12:
            estimates.append((b0, delta_s / delta_e))
    return estimates


if __name__ == "__main__":
    # Sweep initial inverse temperature beta0. This is what allows us to
    # calculate beta_final = (dS/dbeta0) / (dE/dbeta0) = dS/dE.
    BETA0_VALUES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    Z_SHOTS, X_SHOTS, MIRROR_SHOTS = 48, 48, 32
    # IMPORTANT: 128 shots PER beta0, NOT 128 shots for the entire sweep.
    # For this *noiseless* emulator, the exact initial entropy is known.
    # Use it for the final-temperature estimate to avoid unnecessary
    # statistical noise in the mirror measurement; show both estimates.
    random.seed(2)

    forward_z.check()
    forward_x.check()
    mirror_x.check()
    print(f"Compiled Guppy programs: {MODE}, {N} qubits, {M} steps, {len(EDGES)} ZZ bonds")
    print(f"Sweeping {len(BETA0_VALUES)} initial beta values, "
          f"{Z_SHOTS + X_SHOTS + MIRROR_SHOTS} shots each.\n")

    def take_samples(fn, shots: int, beta0: float, seed: int):
        emulator = fn.emulator(n_qubits=N).with_seed(seed)
        inputs = [{"minus": random_thermal_input(beta0)} for _ in range(shots)]
        return read_bits(emulator.run_per_shot(inputs))

    ideal_curve = []     # (beta0, final energy/site, initial entropy/site)
    mirror_curve = []    # (beta0, final energy/site, estimated mirror entropy/site)

    print("beta0    E/N       S(mirror)   S(ideal)")
    print("-" * 43)
    for idx, beta0 in enumerate(BETA0_VALUES):
        z_samples = take_samples(forward_z, Z_SHOTS, beta0, 1000 + 3 * idx)
        x_samples = take_samples(forward_x, X_SHOTS, beta0, 1001 + 3 * idx)
        mirror_samples = take_samples(mirror_x, MIRROR_SHOTS, beta0, 1002 + 3 * idx)

        energy = -average_zz(z_samples) + HX * average_x(x_samples)
        entropy_mirror = binary_entropy(average_x(mirror_samples))
        entropy_ideal = binary_entropy(math.tanh(beta0))

        ideal_curve.append((beta0, energy, entropy_ideal))
        mirror_curve.append((beta0, energy, entropy_mirror))
        print(f"{beta0:4.2f}   {energy:8.4f}   {entropy_mirror:8.4f}   {entropy_ideal:8.4f}")

    # Central differences at interior beta0 points. The endpoint values
    # cannot be evaluated using this 3-point central-difference formula.
    ideal_betas = dict(estimate_final_beta(ideal_curve))
    mirror_betas = dict(estimate_final_beta(mirror_curve))

    def temp_label(beta_f: float | None) -> str:
        if beta_f is None or beta_f <= 0 or not math.isfinite(beta_f):
            return "n/a"
        return f"{1.0 / beta_f:.4f}"

    print("\nFinal temperature estimates (k_B = 1):")
    print("beta0    beta_f(ideal S)  T_f(ideal S)  T_f(mirror S)")
    print("-" * 60)
    for beta0 in BETA0_VALUES[1:-1]:
        beta_ideal = ideal_betas.get(beta0)
        beta_mirror = mirror_betas.get(beta0)
        beta_str = f"{beta_ideal:.4f}" if beta_ideal is not None else "n/a"
        print(f"{beta0:4.2f}       {beta_str:>10}      "
              f"{temp_label(beta_ideal):>9}       {temp_label(beta_mirror):>9}")

    print("\nInterpretation: T_f=1/beta_f is the estimated temperature of the")
    print("locally thermal *final* state, NOT the initial temperature 1/beta0.")
    print("For this noiseless emulator, prefer T_f(ideal S); the mirror result")
    print("is a shot-noisy diagnostic. Neither proves local thermalization.")
    print("Negative/undefined beta_f values are shown as n/a because this")
    print("positive-temperature sweep is too noisy or ill-conditioned there.")
