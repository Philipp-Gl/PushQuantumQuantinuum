"""Independent 128-shot validation of the selected adiabatic candidate at T=1.1875.

From the official challenge repo root:
    uv run --locked python validate_adiabatic_T11875.py

One new X/Y/Z local-sim set. Both the thermal-preparation RNG seeds and emulator
seeds differ from the tuning runs. Existing measurements/files remain unchanged.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import time

from score import score_submission

ROOT = Path.cwd()
SOURCE = ROOT / "adiabatic_T11875.py"
PREVIOUS = ROOT / "submission_adiabatic_T11875_b2p2.json"
FOLDER = ROOT / "validation_T11875_independent"
BETA0 = 2.2
TARGET_T = 1.1875
PREPARATION_SEEDS = {"x": 91001, "y": 91002, "z": 91003}
EMULATOR_SEEDS = {"x": 742, "y": 743, "z": 744}


def run(*args, env):
    command = [str(a) for a in args]
    print("  $", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    if not SOURCE.is_file() or not (ROOT / "score.py").is_file():
        raise SystemExit("Run from hackathon project root with adiabatic_T11875.py and score.py")
    if not PREVIOUS.is_file():
        raise SystemExit("Missing original beta0=2.2 submission JSON: " + str(PREVIOUS))
    baseline = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if baseline.get("temperature") != TARGET_T:
        raise SystemExit("Existing baseline does not target T=1.1875")
    baseline_result = score_submission(baseline)

    original = SOURCE.read_text(encoding="utf-8")
    assert "ADIABATIC_BETA0" in original, "Unexpected source: BETA0 environment setting missing"
    replacement = f"BASIS_SEEDS = {PREPARATION_SEEDS!r}  # independent validation labels"
    changed, n = re.subn(r"(?m)^BASIS_SEEDS\s*=\s*\{[^\n]*\}[^\n]*$", replacement, original)
    if n != 1:
        raise SystemExit(f"Cannot find unique BASIS_SEEDS assignment (found {n})")

    FOLDER.mkdir(parents=True, exist_ok=True)
    candidate = FOLDER / "candidate.py"
    candidate.write_text(changed, encoding="utf-8")
    env = os.environ.copy()
    env["ADIABATIC_BETA0"] = str(BETA0)
    start = time.monotonic()
    outputs = {}
    print(f"Independent validation: beta0={BETA0}, T={TARGET_T}, original schedule", flush=True)
    for basis in "xyz":
        destination = FOLDER / f"{basis}.local.json"
        run("uv", "run", "--locked", "local-sim", candidate,
            "--basis", basis, "--seed", EMULATOR_SEEDS[basis], "-o", destination, env=env)
        outputs[basis] = destination
    submission_path = FOLDER / "submission.json"
    run("uv", "run", "--locked", "python", candidate, "pack", "--temperature", TARGET_T,
        "--x", outputs["x"], "--y", outputs["y"], "--z", outputs["z"],
        "--output", submission_path, env=env)
    candidate_submission = json.loads(submission_path.read_text(encoding="utf-8"))
    confirmed = score_submission(candidate_submission)
    report = {"beta0": BETA0, "A_B_C_D": [1.186, 0.077, 2.181, 0.469],
              "original_path": str(PREVIOUS), "original": baseline_result,
              "independent_path": str(submission_path), "independent": confirmed,
              "preparation_seeds": PREPARATION_SEEDS, "emulator_seeds": EMULATOR_SEEDS}
    (FOLDER / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nRESULTS (two independent 128-shot samples, not an accuracy guarantee)")
    print(f"Original tuning score:  {baseline_result['score']:.6f}")
    print(f"Independent score:      {confirmed['score']:.6f}")
    print(f"Two-run mean:           {(baseline_result['score']+confirmed['score'])/2:.6f}")
    print(f"Independent mag/pairs:  {confirmed['single_site_score']:.6f} / {confirmed['pair_correlation_score']:.6f}")
    print(f"Elapsed: {time.monotonic()-start:.1f}s")
    print(f"Saved independent submission to {submission_path}")
    print("Caution: two runs are insufficient for rigorous statistical error bars;")
    print("the official score is not a bound on Gibbs-state trace distance.")


if __name__ == "__main__":
    main()
