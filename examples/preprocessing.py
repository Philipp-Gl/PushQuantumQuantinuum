"""Classical calibration and sampling, performed before building Guppy programs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
SPECTRUM = DATA_DIR / "ising_spectra.npz"
BASES = ("X", "Y", "Z")
SHOTS = 128
TEMPERATURES = (0.01, 1.1875, 2.375, 4.0)


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


def pack_samples(labels):
    """Encode each qubit's 128 label bits as two signed 64-bit masks for H2."""
    if len(labels) != SHOTS or any(not 0 <= int(label) < (1 << 16) for label in labels):
        raise ValueError("Expected 128 configuration indices in 0..65535")
    masks = np.empty((16, 2), dtype=np.int64)
    for q in range(16):
        for half in range(2):
            mask = sum(((int(labels[64 * half + i]) >> q) & 1) << i for i in range(64))
            masks[q, half] = mask if mask < (1 << 63) else mask - (1 << 64)
    return masks


def preprocess(temperature=2.375, seed=1821, output_dir=DATA_DIR, spectrum_path=SPECTRUM):
    """Save parameters and independent 128-shot sample tables for X, Y, Z."""
    if temperature not in TEMPERATURES:
        raise ValueError(f"Choose a challenge temperature from {TEMPERATURES}")
    spectra = load_spectra(spectrum_path)
    parameters = calibrate(temperature, spectra)
    metadata = {"temperature": temperature, "shots": SHOTS, "parameters": parameters,
                "seeds": {}}
    samples = {}
    for basis_index, basis in enumerate(BASES):
        # Preserve the seeds used by the original, tested Nexus workflow.
        basis_seed = seed + TEMPERATURES.index(temperature) * 3 + basis_index
        labels = sample_labels(spectra["classical_energies"], parameters["k"], basis_seed)
        samples[f"{basis}_labels"] = np.array(labels, dtype=np.int64)
        samples[f"{basis}_masks"] = pack_samples(labels)
        metadata["seeds"][basis] = basis_seed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = output_dir / f"parameters_{temperature:g}.json"
    sample_path = output_dir / f"samples_{temperature:g}.npz"
    parameter_path.write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(sample_path, **samples)
    return parameter_path, sample_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, nargs="+", choices=TEMPERATURES,
                        default=[2.375])
    parser.add_argument("--seed", type=int, default=1821, help="Base seed for the three sample tables")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--spectrum", type=Path, default=SPECTRUM)
    args = parser.parse_args()
    for temperature in args.temperature:
        paths = preprocess(temperature, args.seed, args.output_dir, args.spectrum)
        print("Saved", *paths)


