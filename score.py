"""Score 128-shot X/Y/Z submissions against exact 4x4 g=1 Gibbs observables.

Score = 9 * sum of 3 site-averaged magnetization squared errors
      + sum of 27 translation-averaged correlation squared errors.
Average before squaring; opposite displacements are identified, but no C4 pooling.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

BASES = ("X", "Y", "Z")
N_QUBITS = 16
N_SHOTS = 128
PAIR_I, PAIR_J = np.triu_indices(N_QUBITS, k=1)
REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "reference_g1.npz"
# Representatives of nonzero (dx,dy) modulo d ~ -d on the 4x4 torus.
DISPLACEMENTS = ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                 (1, 3), (2, 0), (2, 1), (2, 2))
SINGLE_SITE_WEIGHT = len(DISPLACEMENTS)
TRANSLATED_SITES = np.array([
    [((q // 4 + dy) % 4) * 4 + (q % 4 + dx) % 4 for q in range(N_QUBITS)]
    for dx, dy in DISPLACEMENTS
])
PAIR_INDEX = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(PAIR_I, PAIR_J))}
TRANSLATED_PAIR_INDICES = np.array([
    [PAIR_INDEX[tuple(sorted((i, int(j))))] for i, j in enumerate(sites)]
    for sites in TRANSLATED_SITES
])


def estimate_observables(submission):
    """Return T, three magnetizations, and a (3,9) displacement-correlation array.

    Displacement order is DISPLACEMENTS; correlations average all 16 origins.
    Shots in different bases are independent measurement sets.
    """
    if not isinstance(submission, dict) or set(submission) != {"temperature", *BASES}:
        raise ValueError("Submission must contain exactly temperature, X, Y, and Z")
    temperature = submission["temperature"]
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be a finite positive number")

    magnetizations, correlations = [], []
    for basis in BASES:
        try:
            bits = np.asarray(submission[basis])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{basis}: expected a rectangular 128-by-16 array") from error
        if bits.shape != (N_SHOTS, N_QUBITS):
            raise ValueError(f"{basis}: expected shape (128, 16), got {bits.shape}")
        has_boolean = any(isinstance(value, (bool, np.bool_))
                          for row in submission[basis] for value in row)
        if (has_boolean or bits.dtype.kind not in "iu"
                or not np.all((bits == 0) | (bits == 1))):
            raise ValueError(f"{basis}: outcomes must be integer bits 0 or 1")
        # Convert before subtraction to avoid unsigned-integer underflow.
        spins = 1.0 - 2.0 * bits.astype(np.float64)
        magnetizations.append(float(spins.mean()))
        correlations.append(
            (spins[:, None, :] * spins[:, TRANSLATED_SITES]).mean(axis=(0, 2))
        )
    return float(temperature), np.array(magnetizations), np.array(correlations)


def exact_observables(temperature, reference_path=REFERENCE_PATH):
    """Read the precomputed Gibbs references, including the finite-T doublet."""
    with np.load(reference_path, allow_pickle=False) as reference:
        matches = np.flatnonzero(np.isclose(reference["temperatures"], temperature,
                                            rtol=0, atol=1e-12))
        if len(matches) != 1:
            allowed = reference["temperatures"].tolist()
            raise ValueError(f"Unsupported temperature {temperature}; choose one of {allowed}")
        expected_pairs = np.column_stack((PAIR_I, PAIR_J))
        if not np.array_equal(reference["pairs"], expected_pairs):
            raise ValueError("Reference pair ordering does not match the scorer")
        index = int(matches[0])
        correlations = reference["correlations"][index][:, TRANSLATED_PAIR_INDICES].mean(axis=2)
        return reference["magnetizations"][index], correlations


def score_observables(magnetizations, correlations, exact_magnetizations,
                      exact_correlations):
    """Score the 30 averaged observables with single-site weight 9."""
    arrays = [np.asarray(a, dtype=float) for a in (
        magnetizations, correlations, exact_magnetizations, exact_correlations)]
    for values, shape in zip(arrays, [(3,), (3, 9), (3,), (3, 9)], strict=True):
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected finite observables of shape {shape}")
    means, pairs, target_means, target_pairs = arrays
    single_site_score = float(SINGLE_SITE_WEIGHT * np.sum((means - target_means) ** 2))
    pair_correlation_score = float(np.sum((pairs - target_pairs) ** 2))
    return {
        "score": single_site_score + pair_correlation_score,
        "single_site_score": single_site_score,
        "pair_correlation_score": pair_correlation_score,
    }


def score_submission(submission, reference_path=REFERENCE_PATH):
    temperature, magnetizations, correlations = estimate_observables(submission)
    target_magnetizations, target_correlations = exact_observables(temperature, reference_path)
    return {
        "temperature": temperature,
        **score_observables(magnetizations, correlations,
                            target_magnetizations, target_correlations),
        "estimated_magnetizations": dict(zip(BASES, magnetizations.tolist(), strict=True)),
        "exact_magnetizations": dict(zip(BASES, target_magnetizations.tolist(), strict=True)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submissions", nargs="+", type=Path, help="One simple JSON file per temperature")
    parser.add_argument("--output", type=Path, help="Also save the score report as JSON")
    args = parser.parse_args()
    results = []
    for path in args.submissions:
        try:
            submission = json.loads(path.read_text())
            results.append({"file": str(path), **score_submission(submission)})
        except (OSError, ValueError, TypeError) as error:
            parser.exit(2, f"{path}: {error}\n")
    report = json.dumps({"results": results}, indent=2, allow_nan=False) + "\n"
    if args.output:
        # A report must never overwrite a participant's input file.
        if args.output.resolve() in {p.resolve() for p in args.submissions}:
            parser.error("--output must differ from every submission path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
