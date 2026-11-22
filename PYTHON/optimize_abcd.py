"""Fast screening of the four adiabatic schedule parameters A,B,C,D.

Put beside Guppy1.py and score.py in the official PushQuantum repository.
Run from repo root:
    uv run --locked python optimize_abcd_quick.py --source Guppy1.py

Uses 5 candidate schedules x 2 calibrated temperatures x 3 bases = 30
128-shot local-sim calls. The 16-step circuit, gate order, and thermal
preparation remain unchanged. This is a SCREEN, not a converged optimizer.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from score import REFERENCE_PATH, score_submission

# Existing paper schedule; test four nearby mixed perturbations (10%).
BASE = (1.186, 0.077, 2.181, 0.469)
CANDIDATES = {
    "paper": (1.00, 1.00, 1.00, 1.00),
    "try1":  (1.10, 0.90, 1.10, 0.90),
    "try2":  (0.90, 1.10, 0.90, 1.10),
    "try3":  (1.10, 1.10, 0.90, 0.90),
    "try4":  (0.90, 0.90, 1.10, 1.10),
}
# Retain the beta0 calibration found earlier.
TARGETS = ((4.0, 0.60), (2.375, 1.30))


def variant_source(text, params, beta):
    a, b, c, d = params
    text, n1 = re.subn(
        r"(?m)^A\s*,\s*B\s*,\s*C\s*,\s*D\s*=.*$",
        f"A, B, C, D = {a:.12g}, {b:.12g}, {c:.12g}, {d:.12g}",
        text,
    )
    text, n2 = re.subn(
        r"(?m)^BETA0\s*=.*$",
        f"BETA0 = {beta:.12g}  # initial inverse temperature",
        text,
    )
    if (n1, n2) != (1, 1):
        raise ValueError(f"Expected exactly one A/B/C/D and one BETA0 assignment; got {n1}, {n2}")
    return text


def read_shots(path, basis):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("basis") != basis or data.get("shots") != 128:
        raise ValueError(f"Invalid basis/shot count: {path}")
    records = data.get("results", [])
    if len(records) != 128:
        raise ValueError(f"Expected 128 records: {path}")
    result = [None] * 128
    for record in records:
        tags = dict(record)
        idx = tags.get("shot")
        bits = tags.get("bits")
        if type(idx) is not int or not 0 <= idx < 128 or result[idx] is not None:
            raise ValueError(f"Invalid shot index in {path}: {idx!r}")
        if not isinstance(bits, list) or len(bits) != 16 or any(type(v) is not int or v not in (0, 1) for v in bits):
            raise ValueError(f"Invalid bits in {path}: shot {idx}")
        result[idx] = bits
    if any(row is None for row in result):
        raise ValueError(f"Missing shot(s) in {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Guppy1.py", type=Path)
    parser.add_argument("--out", default="abcd_quick", type=Path)
    parser.add_argument("--noise", action="store_true", help="Add the local depolarizing noise model")
    args = parser.parse_args()

    if not args.source.is_file() or not REFERENCE_PATH.is_file():
        parser.error("Run from the official repo with Guppy1.py, score.py, and data/reference_g1.npz")
    original = args.source.read_text(encoding="utf-8")
    # Validate both source substitutions before costly simulation.
    variant_source(original, BASE, 0.60)
    args.out.mkdir(parents=True, exist_ok=True)
    all_results = []

    for name, factors in CANDIDATES.items():
        params = tuple(round(x * scale, 10) for x, scale in zip(BASE, factors))
        for temperature, beta in TARGETS:
            tlabel = "4" if temperature == 4.0 else "2375"
            folder = args.out / name / f"T_{tlabel}"
            folder.mkdir(parents=True, exist_ok=True)
            program = folder / "candidate.py"
            program.write_text(variant_source(original, params, beta), encoding="utf-8")
            submission = {"temperature": temperature}
            print(f"\n{name}: T={temperature:g}, beta0={beta:g}, A/B/C/D={params}", flush=True)
            for i, basis in enumerate("xyz"):
                destination = folder / f"{basis}.local.json"
                # Identical simulator seeds for a given target/basis across schedules.
                # The source's BASIS_SEEDS also remain unchanged across schedules.
                seed = (420 if temperature == 4.0 else 720) + i
                cmd = ["uv", "run", "--locked", "local-sim", str(program),
                       "--basis", basis, "--seed", str(seed), "-o", str(destination)]
                if args.noise:
                    cmd.extend(["--noise-model", "depolarizing"])
                print("  running", basis.upper(), flush=True)
                subprocess.run(cmd, check=True)
                submission[basis.upper()] = read_shots(destination, basis)

            submission_path = folder / "submission.json"
            submission_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
            score = score_submission(submission)  # official score.py, including shape validation
            item = {"schedule": name, "parameters": dict(zip("ABCD", params)),
                    "beta0": beta, "temperature": temperature,
                    "submission": str(submission_path), **score}
            all_results.append(item)
            print(f"  SCORE = {score['score']:.5f}"
                  f" (mag={score['single_site_score']:.5f}, corr={score['pair_correlation_score']:.5f})",
                  flush=True)
            summary = args.out / "summary.json"
            summary.write_text(json.dumps({"results": all_results}, indent=2) + "\n", encoding="utf-8")

    print("\n=== FIRST SCREEN: lowest single-run scores ===")
    for temperature, _ in TARGETS:
        contenders = [r for r in all_results if r["temperature"] == temperature]
        contenders.sort(key=lambda r: r["score"])
        for r in contenders:
            print(f"T={temperature:g}: {r['schedule']:5}  score={r['score']:.5f}  A/B/C/D={r['parameters']}")
    print(f"\nSaved {args.out / 'summary.json'}")
    print("Caution: 128-shot noise can change rankings; recalibrate beta0 and repeat winners before final selection.")


if __name__ == "__main__":
    main()
