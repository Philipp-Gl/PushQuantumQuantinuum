"""Calibrate the original 16-step Guppy thermal-preparation circuit against score.py.

Put this script beside Guppy1.py and score.py in the official hackathon repo.
Requires the repo's data/reference_g1.npz and `uv` installation.

Example (first, deliberately coarse sweep):
    uv run --locked python calibrate_guppy.py --source Guppy1.py --betas 0.25 1.0 3.0

The only circuit change is the module-level BETA0 constant. This script does
NOT edit Guppy1.py, change the adiabatic schedule, or claim any candidate is
already calibrated to the specified target temperatures. It uses the official
score.py functions and its exact thermal reference data.
"""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from score import (
    BASES,
    DISPLACEMENTS,
    REFERENCE_PATH,
    estimate_observables,
    exact_observables,
    score_observables,
)

TARGET_TEMPERATURES = (0.01, 1.1875, 2.375, 4.0)
EXPECTED_SHOTS = 128
EXPECTED_QUBITS = 16


def energy_per_site(means, correlations):
    """H/N = -<ZZ>_right - <ZZ>_down - <X> for the 4x4 periodic TFI."""
    right = DISPLACEMENTS.index((1, 0))
    down = DISPLACEMENTS.index((0, 1))
    return float(-correlations[2, right] - correlations[2, down] - means[0])


def candidate_source(original: str, beta0: float) -> str:
    """Replace exactly one top-level BETA0 assignment; preserve everything else."""
    changed, n = re.subn(
        r"(?m)^BETA0\s*=\s*[^\n]*$",
        f"BETA0 = {beta0:.12g}   # initial inverse temperature (calibration candidate)",
        original,
    )
    if n != 1:
        raise ValueError(f"Expected one module-level BETA0 assignment, found {n}")
    return changed


def load_shots(path: Path, expected_basis: str):
    """Extract the 128x16 'bits' records from the official local-sim JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("basis") != expected_basis:
        raise ValueError(f"{path}: expected basis {expected_basis}")
    if data.get("shots") != EXPECTED_SHOTS:
        raise ValueError(f"{path}: expected 128 shots")
    records = data.get("results")
    if not isinstance(records, list) or len(records) != EXPECTED_SHOTS:
        raise ValueError(f"{path}: expected 128 result records")

    rows = [None] * EXPECTED_SHOTS
    for record in records:
        tags = dict(record)
        shot = tags.get("shot")
        bits = tags.get("bits")
        if type(shot) is not int or not 0 <= shot < EXPECTED_SHOTS:
            raise ValueError(f"{path}: invalid shot index {shot!r}")
        if rows[shot] is not None:
            raise ValueError(f"{path}: duplicate shot index {shot}")
        if not isinstance(bits, list) or len(bits) != EXPECTED_QUBITS:
            raise ValueError(f"{path}: shot {shot} does not have 16 bits")
        if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
            raise ValueError(f"{path}: shot {shot} contains nonbinary values")
        rows[shot] = bits
    if any(row is None for row in rows):
        raise ValueError(f"{path}: missing shots")
    return rows


def run_simulator(source: Path, basis: str, dest: Path, seed: int):
    command = [
        "uv", "run", "--locked", "local-sim", str(source),
        "--basis", basis, "--seed", str(seed), "-o", str(dest),
    ]
    print("  ", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not dest.exists():
        raise RuntimeError(f"Simulator did not produce {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Guppy1.py"),
                        help="Your existing, successfully compiled submission circuit")
    parser.add_argument("--betas", nargs="+", type=float,
                        default=[0.25, 1.0, 3.0],
                        help="Trial initial inverse temperatures beta0")
    parser.add_argument("--out", type=Path, default=Path("calibration"),
                        help="Directory for temporary circuits, shot data and summary")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"Cannot find source circuit {args.source}")
    if not REFERENCE_PATH.is_file():
        parser.error(f"Missing exact Gibbs reference: {REFERENCE_PATH}; run in the official repo")
    if any(not math.isfinite(b) or b < 0 for b in args.betas):
        parser.error("All beta0 candidates must be finite and nonnegative")
    if len(set(args.betas)) != len(args.betas):
        parser.error("Duplicate beta0 candidates")

    # Load reference observables BEFORE expensive simulator work.
    targets = {}
    for temperature in TARGET_TEMPERATURES:
        exact_means, exact_corrs = exact_observables(temperature)
        targets[temperature] = (exact_means, exact_corrs,
                                energy_per_site(exact_means, exact_corrs))

    original = args.source.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    used_names = set()

    for beta0 in args.betas:
        slug = f"beta_{beta0:.8g}".replace(".", "p").replace("+", "")
        if slug in used_names:
            parser.error("Two beta values map to the same filename; use less similar values")
        used_names.add(slug)
        folder = args.out / slug
        folder.mkdir(parents=True, exist_ok=True)
        candidate = folder / f"Guppy_{slug}.py"
        candidate.write_text(candidate_source(original, beta0), encoding="utf-8")

        print(f"\n=== beta0 = {beta0:g} ===", flush=True)
        arrays = {}
        for n, basis in enumerate(("x", "y", "z")):
            shot_json = folder / f"{basis}.local.json"
            run_simulator(candidate, basis, shot_json, 42 + n)
            arrays[basis.upper()] = load_shots(shot_json, basis)

        # Full shape/type validation is performed by the actual scorer.
        example_submission = {"temperature": 2.375, **arrays}
        _, means, corrs = estimate_observables(example_submission)
        measured_energy = energy_per_site(means, corrs)
        print(f"  Measured E/N = {measured_energy:.5f}", flush=True)

        for temperature in TARGET_TEMPERATURES:
            ref_means, ref_corrs, ref_energy = targets[temperature]
            components = score_observables(means, corrs, ref_means, ref_corrs)
            result = {
                "beta0": beta0,
                "target_temperature": temperature,
                "measured_energy_per_site": measured_energy,
                "target_energy_per_site": ref_energy,
                "energy_difference": measured_energy - ref_energy,
                **components,
            }
            results.append(result)
            print(f"  T={temperature:7g}: exact E/N={ref_energy:8.4f}, "
                  f"dE={result['energy_difference']:+8.4f}, "
                  f"score={result['score']:8.4f}", flush=True)

    summary_path = args.out / "calibration_summary.json"
    summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print("\nLowest scores among the tested beta0 candidates:")
    for t in TARGET_TEMPERATURES:
        candidates = [r for r in results if r["target_temperature"] == t]
        best_score = min(candidates, key=lambda r: r["score"])
        best_energy = min(candidates, key=lambda r: abs(r["energy_difference"]))
        print(f"  T={t:g}: beta0={best_score['beta0']:g} by score "
              f"({best_score['score']:.4f}); "
              f"beta0={best_energy['beta0']:g} by energy")
    print(f"\nSaved {summary_path}")
    print("These are coarse, shot-noisy calibration results, NOT verified final temperatures.")


if __name__ == "__main__":
    main()
