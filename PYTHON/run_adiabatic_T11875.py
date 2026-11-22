"""One-command adiabatic thermal-state scorer for PushQuantum, T=1.1875.

Place this file and adiabatic_T11875.py next to score.py and run:
  uv run --locked python run_adiabatic_T11875.py
Optional sweep, only after first successful result:
  uv run --locked python run_adiabatic_T11875.py --betas 1.5 2.5 3.5

This executes genuine local-sim measurements, not synthetic samples.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROGRAM = "adiabatic_T11875.py"
TARGET_T = 1.1875


def execute(args, env):
    print("  $", " ".join(map(str, args)), flush=True)
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed (exit {result.returncode}). Outputs already written have been kept.")
    return result.stdout


def run_one(beta0):
    tag = f"b{beta0:g}".replace(".", "p")
    env = os.environ.copy()
    env["ADIABATIC_BETA0"] = str(beta0)
    outputs = {}
    start = time.monotonic()
    for basis, seed in (("x", 42), ("y", 43), ("z", 44)):
        out = f"adiabatic_T11875_{tag}.{basis}.local.json"
        execute(["uv", "run", "--locked", "local-sim", PROGRAM,
                 "--basis", basis, "--seed", str(seed), "-o", out], env)
        if not Path(out).is_file():
            raise SystemExit(f"No simulator output found: {out}")
        outputs[basis] = out
    submission = f"submission_adiabatic_T11875_{tag}.json"
    execute(["uv", "run", "--locked", "python", PROGRAM, "pack",
             "--temperature", str(TARGET_T), "--x", outputs["x"],
             "--y", outputs["y"], "--z", outputs["z"],
             "--output", submission], env)
    result = execute(["uv", "run", "--locked", "python", "score.py", submission], env)
    print(result, flush=True)
    match = re.search(r'"score"\s*:\s*([0-9.eE+-]+)', result)
    score = float(match.group(1)) if match else None
    print(f"Completed beta0={beta0:g}, score={score}, elapsed={time.monotonic()-start:.1f}s", flush=True)
    return score, submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", nargs="+", type=float, default=[2.0],
                   help="Initial inverse temperatures to try (NOT final 1/T)")
    args = p.parse_args()
    if not Path(PROGRAM).is_file():
        raise SystemExit(f"Missing {PROGRAM}; put both downloaded files in the challenge folder.")
    if not Path("score.py").is_file():
        raise SystemExit("score.py not found. Run in the official challenge folder that contains score.py.")
    results = []
    for beta in args.betas:
        print(f"\n=== beta0={beta:g}, target T={TARGET_T} ===", flush=True)
        score, path = run_one(beta)
        results.append((score, path, beta))
    valid = [r for r in results if r[0] is not None]
    if valid:
        best = min(valid, key=lambda r: r[0])
        print(f"\nBEST: {best[1]}  score={best[0]:.6f}  initial beta0={best[2]}")
    else:
        print("\nOutputs created, but could not extract score. Inspect scorer outputs above.")

if __name__ == "__main__":
    main()
