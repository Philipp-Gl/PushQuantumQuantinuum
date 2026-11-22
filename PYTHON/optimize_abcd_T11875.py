"""Quick A/B/C/D schedule screen at T=1.1875 for the verified Guppy circuit.

Run in the official hackathon project root (with score.py and data/):
    uv run --locked python optimize_abcd_T11875.py

Defaults: beta0=2.0, 3 schedules, 128 shots per X/Y/Z basis.
Each schedule changes all four paper coefficients by +/-10%; this is a
coarse screen, not a converged optimization or a proof of thermalization.
Results are stored under abcd_T11875_quick/beta_<value>/.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from score import score_submission

BASE = (1.186, 0.077, 2.181, 0.469)
FACTORS = {
    "try1": (1.10, 0.90, 1.10, 0.90),
    "try2": (0.90, 1.10, 0.90, 1.10),
    "try3": (1.10, 1.10, 0.90, 0.90),
}
T = 1.1875
BASELINE_SCORE = 0.18627439478917546  # existing beta0=2.0, paper schedule


def run(command, env):
    print("  $ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Initial inverse temperature, default 2.0")
    parser.add_argument("--source", type=Path, default=Path("adiabatic_T11875.py"))
    parser.add_argument("--out", type=Path, default=Path("abcd_T11875_quick"))
    args = parser.parse_args()
    if not args.source.is_file() or not Path("score.py").is_file():
        parser.error("Run from the hackathon root; need adiabatic_T11875.py and score.py")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be a finite nonnegative number")

    original = args.source.read_text(encoding="utf-8")
    match = re.findall(r"(?m)^A\s*,\s*B\s*,\s*C\s*,\s*D\s*=.*$", original)
    if len(match) != 1 or "ADIABATIC_BETA0" not in original:
        parser.error("Incompatible source: expected working adiabatic_T11875.py")
    tag = f"beta_{args.beta:g}".replace(".", "p")
    target_folder = args.out / tag
    target_folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ADIABATIC_BETA0"] = str(args.beta)
    results = []

    for label, factors in FACTORS.items():
        parameters = tuple(round(v * f, 12) for v, f in zip(BASE, factors))
        row = "A, B, C, D = " + ", ".join(map(str, parameters))
        variant, n = re.subn(r"(?m)^A\s*,\s*B\s*,\s*C\s*,\s*D\s*=.*$", row, original)
        if n != 1:
            raise RuntimeError("Source has changed unexpectedly")
        folder = target_folder / label
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "candidate.py"
        source.write_text(variant, encoding="utf-8")
        print(f"\n=== {label}: beta0={args.beta:g}, {row} ===", flush=True)
        started = time.monotonic()
        raw = {}
        for i, basis in enumerate("xyz"):
            out = folder / f"{basis}.local.json"
            run(["uv", "run", "--locked", "local-sim", source,
                 "--basis", basis, "--seed", 42+i, "-o", out], env)
            raw[basis] = out
        submission_path = folder / "submission.json"
        run(["uv", "run", "--locked", "python", source, "pack", "--temperature", T,
             "--x", raw["x"], "--y", raw["y"], "--z", raw["z"],
             "--output", submission_path], env)
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        score = score_submission(submission)  # exact official score.py function
        row_result = {"label": label, "initial_beta0": args.beta,
                      "parameters": dict(zip("ABCD", parameters)),
                      "submission": str(submission_path), **score}
        results.append(row_result)
        print(f"RESULT {label}: score={score['score']:.6f} "
              f"(single={score['single_site_score']:.6f}, "
              f"pairs={score['pair_correlation_score']:.6f}) "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        (target_folder / "summary.json").write_text(
            json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")

    print(f"\nPrevious baseline at beta0=2: {BASELINE_SCORE:.6f}")
    for r in sorted(results, key=lambda r: r["score"]):
        print(f"{r['label']}: {r['score']:.6f} | {r['submission']}")
    winner = min(results, key=lambda r: r["score"])
    if winner["score"] < BASELINE_SCORE:
        print(f"BEST NEW CANDIDATE: {winner['submission']} (single-seed screen only)")
    else:
        print("No improvement over the previous baseline in this screen.")
    print(f"Saved detailed results to {target_folder / 'summary.json'}")


if __name__ == "__main__":
    main()
