"""A calibrated tilted-product-mixture baseline for the 4x4 g=1 Ising model.

NumPy runs on the host, through @guppy.comptime, not on H2. We independently sample
128 classical labels before compilation, then select label[shot] on the device.
Running the same compiled program again reuses that table; change seed to resample.
This fits exact target <mx> and <mz^2>; it is not a general Gibbs-state algorithm.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, output, owned
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import (
    collect_measurements, h, measure_array, qubit, ry, s,
)

SHOTS = 128
TEMPERATURES = (0.01, 1.1875, 2.375, 4.0)
SPECTRUM = Path(__file__).resolve().parent / "data" / "ising_spectra.npz"


def load_spectra(path=SPECTRUM):
    """Load computational-basis classical energies and symmetry-weighted g=1 ED.

    Classical row s is bitstring integer s, with bit q at site q=x+4*y.
    Quantum energies are symmetry-sector eigenvalues, NOT classical bitstrings.
    """
    with np.load(path, allow_pickle=False) as data:
        spectra = {name: data[name] for name in data.files}
    if spectra["classical_energies"].shape != (1 << 16,):
        raise ValueError("Expected 65,536 classical energies in bitstring order")
    return spectra


def classical_probabilities(energies, k):
    """p(s) proportional to exp[-k E_classical(s)] = exp[k sum_<ij> s_i s_j]."""
    weights = np.exp(-k * (energies - energies.min()))
    return weights / weights.sum()


def calibrate(temperature, spectra):
    """Fit theta and k to exact g=1 mx and mz2, including at T=0.01."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    energies = spectra["quantum_energies"]
    weights = spectra["quantum_multiplicities"] * np.exp(
        -(energies - energies.min()) / temperature
    )
    weights /= weights.sum()
    mx = float(spectra["quantum_mx"] @ weights)
    mz2 = float(spectra["quantum_mz2"] @ weights)
    theta = float(np.arcsin(np.clip(mx, -1, 1)))
    # Distinct-site ZZ is multiplied by cos(theta)^2; self-correlations stay 1.
    classical_target = (mz2 - 1 / 16) / math.cos(theta) ** 2 + 1 / 16
    labels = np.arange(1 << 16, dtype=np.uint32)
    populations = np.zeros(1 << 16, dtype=np.int16)
    for q in range(16):
        populations += ((labels >> q) & 1).astype(np.int16)
    classical_mz2 = (1 - 2 * populations / 16) ** 2
    classical_energies = spectra["classical_energies"]

    def moment(k):
        return float(classical_probabilities(classical_energies, k) @ classical_mz2)

    lo, hi = 0.0, 16.0
    if not moment(lo) <= classical_target <= moment(hi):
        raise ValueError("Target is outside this ferromagnetic tilted-product family")
    for _ in range(64):
        mid = (lo + hi) / 2
        if moment(mid) < classical_target:
            lo = mid
        else:
            hi = mid
    return {"k": (lo + hi) / 2, "theta_radians": theta,
            "target_mx": mx, "target_mz2": mz2}


def sample_labels(energies, k, seed, shots=SHOTS):
    """Ordinary Python sampler: sample whole correlated configurations, not bits."""
    probabilities = classical_probabilities(energies, k)
    return np.random.default_rng(seed).choice(
        len(probabilities), size=shots, p=probabilities
    ).tolist()


@guppy
def measure_z(qs: array[qubit, 16] @ owned) -> None:
    output("bits", collect_measurements(measure_array(qs)))


@guppy.comptime
def measure_x(qs: array[qubit, 16] @ owned) -> None:
    for q in range(16):
        h(qs[q])
    measure_z(qs)


@guppy.comptime
def measure_y(qs: array[qubit, 16] @ owned) -> None:
    for q in range(16):
        # S dagger, then H, maps a Y measurement to Z. S^3 = S dagger.
        s(qs[q])
        s(qs[q])
        s(qs[q])
        h(qs[q])
    measure_z(qs)


@guppy
def tilt_qubit(q: qubit, shot: int, low: int, high: int,
               zero_angle: angle, one_angle: angle) -> int:
    bit = 0
    if shot < 64:
        bit = (low >> shot) & 1
    else:
        bit = (high >> (shot - 64)) & 1
    if bit == 1:
        ry(q, one_angle)
    else:
        ry(q, zero_angle)
    return bit


def build_program(temperature=2.375, basis="Z", seed=1821, spectrum_path=SPECTRUM,
                  shot_offset=1):
    """Return a zero-argument Guppy main and metadata; execute exactly 128 shots.

    Each shot uses an independently sampled label; 128 shots are the total for
    this temperature/basis, not 128 per configuration. Qubits are never entangled.
    H2 shot indices start at 1; pass shot_offset=0 for local Selene execution.
    """
    if basis not in ("X", "Y", "Z"):
        raise ValueError("basis must be X, Y, or Z")
    if shot_offset not in (0, 1):
        raise ValueError("shot_offset must be 1 for H2 or 0 for Selene")
    spectra = load_spectra(spectrum_path)
    parameters = calibrate(temperature, spectra)
    energies = spectra["classical_energies"]
    k = parameters["k"]
    theta = parameters["theta_radians"]
    measurement = {"X": measure_x, "Y": measure_y, "Z": measure_z}[basis]

    @guppy.comptime
    def sample_and_tilt(qs: array[qubit, 16]) -> int:
        # NumPy executes on the host. Pack each qubit's 128 sampled bits into
        # two signed integers, avoiding H2's unsupported dynamic array indexing.
        labels = sample_labels(energies, k, seed)
        shot = get_current_shot() - shot_offset
        label = 0
        for q in range(16):
            low = sum(((labels[i] >> q) & 1) << i for i in range(64))
            high = sum(((labels[i + 64] >> q) & 1) << i for i in range(64))
            low = low if low < (1 << 63) else low - (1 << 64)
            high = high if high < (1 << 63) else high - (1 << 64)
            bit = tilt_qubit(qs[q], shot, low, high, angle(theta / math.pi),
                             angle((math.pi - theta) / math.pi))
            label = label | (bit << q)
        return label

    @guppy
    def body() -> array[qubit, 16]:
        # Sampling is at compile time; label selection is per shot on the device.
        qs = array(qubit() for _ in range(16))
        label = sample_and_tilt(qs)
        shot = get_current_shot() - comptime(shot_offset)
        output("shot", shot)
        output("label", label)
        return qs

    @guppy
    def main() -> None:
        qs = body()
        measurement(qs)

    metadata = {"temperature": float(temperature), "basis": basis, "seed": seed,
                "shots": SHOTS, "shot_index_offset": shot_offset,
                "parameters": parameters,
                "sampled_labels": sample_labels(energies, k, seed)}
    return main, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=2.375)
    parser.add_argument("--basis", choices=("X", "Y", "Z"), default="Z")
    parser.add_argument("--seed", type=int, default=1821)
    parser.add_argument("--output", type=Path, default=Path("tilted_product.hugr"))
    args = parser.parse_args()
    main, metadata = build_program(args.temperature, args.basis, args.seed)
    args.output.write_bytes(main.compile().to_bytes())
    print(f"Compiled {args.output}; parameters={metadata['parameters']}")
