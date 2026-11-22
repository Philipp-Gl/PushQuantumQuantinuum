"""Rescore existing Guppy1/Guppy2/adiabatic measurement data at T=1.1875.

Run in the official challenge repo beside score.py:
    uv run --locked python reuse_adiabatic_sweeps.py
    uv run --locked python reuse_adiabatic_sweeps.py --save-best

No local-sim/compile calls are made. The original experiment files are unchanged.
"""
import argparse
import json
import re
from pathlib import Path

from score import score_submission

TARGET_T = 1.1875
ROOT = Path.cwd()


def unpack_raw(paths):
    """Three raw official local-sim basis files -> a scorer-compatible record."""
    submission = {"temperature": TARGET_T}
    for basis, path in paths.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("basis") != basis.lower() or doc.get("shots") != 128:
            raise ValueError(f"Bad basis or shot count in {path}")
        rows = [None] * 128
        for record in doc["results"]:
            item = dict(record)
            index = int(item["shot"])
            bits = item["bits"]
            if not 0 <= index < 128 or rows[index] is not None:
                raise ValueError(f"Invalid shot in {path}")
            rows[index] = bits
        if any(row is None for row in rows):
            raise ValueError(f"Missing shots in {path}")
        submission[basis] = rows
    return submission


def beta_label(folder):
    for p in folder.glob("*.py"):
        try:
            source = p.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        match = re.search(r"(?m)^BETA0\s*=\s*([0-9.eE+-]+)", source)
        if match:
            return match.group(1)
    return "?"


def collect_candidates():
    # Earlier parameter and beta sweeps: all original basis outputs are preserved.
    for base in ("calibration", "calibration_repeated", "abcd_quick"):
        directory = ROOT / base
        if directory.exists():
            for x in directory.rglob("x.local.json"):
                y, z = x.with_name("y.local.json"), x.with_name("z.local.json")
                if y.is_file() and z.is_file():
                    yield str(x.parent.relative_to(ROOT)), beta_label(x.parent), lambda x=x,y=y,z=z: unpack_raw({"X": x, "Y": y, "Z": z})
    # Current and previous single-candidate root measurements (Guppy2 etc.).
    for x in ROOT.glob("*.x.local.json"):
        stem = x.name[:-len(".x.local.json")]
        y, z = ROOT / f"{stem}.y.local.json", ROOT / f"{stem}.z.local.json"
        if y.is_file() and z.is_file():
            yield stem, "from name", lambda x=x,y=y,z=z: unpack_raw({"X": x, "Y": y, "Z": z})
    # Already packed submissions, including Guppy2 and the new beta sweep.
    for file in [*ROOT.glob("submission*.json"), *(ROOT / "abcd_quick").glob("**/submission.json")]:
        yield str(file.relative_to(ROOT)), "packed", lambda file=file: {**json.loads(file.read_text(encoding="utf-8")), "temperature": TARGET_T}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-best", action="store_true", help="Write scorer-compatible winner to submission_reused_T11875.json")
    args = p.parse_args()
    ranked = []
    bad = 0
    for name, beta, load in collect_candidates():
        try:
            data = load()
            result = score_submission(data)
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
            bad += 1
            continue
        ranked.append((result["score"], name, beta, result, data))
    ranked.sort(key=lambda x: x[0])
    print(f"Candidates rescored at T={TARGET_T}: {len(ranked)} (skipped invalid/incomplete: {bad})")
    print("All are REUSED measurements. No quantum simulation was run.\n")
    print(f"{'score':>10} {'mag':>10} {'pairs':>10} {'beta0':>8}  source")
    for score, name, beta, result, _ in ranked[:20]:
        print(f"{score:10.6f} {result['single_site_score']:10.6f} {result['pair_correlation_score']:10.6f} {beta:>8}  {name}")
    if not ranked:
        raise SystemExit("No usable prior measurements were found. Run from the hackathon repository root.")
    if args.save_best:
        target = ROOT / "submission_reused_T11875.json"
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing {target}")
        target.write_text(json.dumps(ranked[0][4], indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved {target.name} from {ranked[0][1]} (score={ranked[0][0]:.6f})")
    print("\nNote: selecting the minimum of many noisy trials can overfit measurement noise.")


if __name__ == "__main__":
    main()
