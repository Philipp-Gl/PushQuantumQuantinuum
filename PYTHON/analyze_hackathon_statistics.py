"""Analyze REAL PushQuantum thermal-state measurement files (no simulations).

Run from project root, beside official score.py and data/reference_g1.npz:
    uv run --locked python analyze_hackathon_statistics.py
    uv run --locked python analyze_hackathon_statistics.py --bootstrap 1000
    uv run --locked python analyze_hackathon_statistics.py --focus my_submission.json

Outputs: hackathon_statistics/runs.csv, groups.csv, focus.csv, report.json.
All scores use the OFFICIAL score.py and reference_g1.npz.

Interpretation:
* Between-run SD/SE requires independent shot sets of the SAME configuration.
* Within-run bootstrap percentiles estimate 128-shot score variability, not a
  rigorous confidence interval for the underlying Gibbs-state error.
* A noise-floor-corrected squared observable error is unbiased under iid shot
  sampling; it can be negative, and is NOT a quantum-state fidelity bound.
* Reused/combined files are excluded by default to prevent double counting.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import score  # Official challenge scorer; do not replace with custom references.

TARGETS = (0.01, 1.1875, 2.375, 4.0)
BASES = ("X", "Y", "Z")
FINAL_HINTS = (
    "submission_adiabatic_T11875_b2p2.json",
    "validation_T11875_independent/submission.json",
    "submission_noisy2_2375.json",
    "submission_Guppy2_T4.json",
    "submission_VQE_001.json",
)
# These files recycle earlier measurements. Useful diagnostics, not new runs.
EXCLUDE_MARKERS = ("submission_reused", "submission_adiabatic_T11875_combined",
                   "submission_best_reused", "submission_best_reused_T11875")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_arrays(arrays):
    out = {}
    for basis in BASES:
        a = np.asarray(arrays[basis])
        if a.shape != (128, 16) or a.dtype.kind not in "iu" or not np.all((a == 0) | (a == 1)):
            raise ValueError(f"{basis}: need exactly 128 x 16 integer bits")
        out[basis] = a.astype(np.uint8)
    return out


def raw_triplet(xfile):
    prefix = xfile.name[:-len("x.local.json")]
    # Handles both 'x.local.json' and 'myrun.x.local.json'.
    files = {b: xfile.with_name(prefix + b.lower() + ".local.json") for b in BASES}
    if not all(p.is_file() for p in files.values()):
        return None
    arrays = {}
    for basis, path in files.items():
        data = read_json(path)
        if data.get("basis") != basis.lower() or data.get("shots") != 128:
            raise ValueError(f"{path}: wrong basis or shot count")
        ordered = [None] * 128
        for record in data.get("results", []):
            d = dict(record)
            idx = d.get("shot")
            if type(idx) is not int or not 0 <= idx < 128 or ordered[idx] is not None:
                raise ValueError(f"{path}: missing/duplicate/invalid shot")
            ordered[idx] = d.get("bits")
        if any(v is None for v in ordered):
            raise ValueError(f"{path}: incomplete measurements")
        arrays[basis] = ordered
    return validate_arrays(arrays)


def digest(arrays):
    h = hashlib.sha256()
    for basis in BASES:
        h.update(arrays[basis].tobytes())
    return h.hexdigest()


def fmt_num(n):
    return str(float(n)).rstrip("0").rstrip(".") if "." in str(float(n)) else str(n)


def source_metadata(path, root):
    """Best-effort metadata; NEVER guess a beta or ABCD for an unknown file."""
    p = path.as_posix().lower()
    family = path.parts[0] if len(path.parts) > 1 else "root"
    if "validation_t11875_independent" in p or "adiabatic_t11875" in p:
        family = "T11875_original_schedule"
    elif "calibration_repeated" in p:
        family = "calibration_repeated"
    elif "calibration" in p:
        family = "calibration_single_seed"
    elif "abcd_t11875_quick" in p:
        family = "abcd_T11875_quick"
    elif "abcd_quick" in p:
        family = "abcd_quick"
    beta = None
    m = re.search(r"(?:beta_|_b)(\d+(?:p\d+)?)(?:/|\\|\.|_|$)", p)
    if m:
        beta = float(m.group(1).replace("p", "."))
    params = None
    # Inspect source in experiment folder, not comments/names alone.
    folder = root / path.parent
    sources = sorted(folder.glob("*.py")) if folder.is_dir() and folder != root else []
    # Do not infer Guppy2/VQE parameters from an unrelated root-level script.
    if not sources and family == "T11875_original_schedule":
        candidate = root / "adiabatic_T11875.py"
        if candidate.is_file(): sources = [candidate]
    if not sources and "guppy2" in p and (root / "Guppy2.py").is_file():
        sources = [root / "Guppy2.py"]
    if not sources and "guppy1" in p and (root / "Guppy1.py").is_file():
        sources = [root / "Guppy1.py"]
    for py in sources:
        s = py.read_text(encoding="utf-8", errors="replace")
        ab = re.search(r"(?m)^A\s*,\s*B\s*,\s*C\s*,\s*D\s*=\s*([^\n#]+)", s)
        if ab:
            try:
                v = tuple(float(x.strip()) for x in ab.group(1).split(","))
                if len(v) == 4:
                    params = v
            except ValueError:
                pass
        bt = re.search(r"(?m)^BETA0\s*=\s*([+-]?\d+(?:\.\d+)?)\b", s)
        if beta is None and bt:
            beta = float(bt.group(1))
        if params is not None and beta is not None:
            break
    if "validation_t11875_independent" in p and (root / "validation_T11875_independent" / "results.json").exists():
        record = read_json(root / "validation_T11875_independent" / "results.json")
        beta = float(record["beta0"])
        if "A_B_C_D" in record:
            params = tuple(float(x) for x in record["A_B_C_D"])
    if beta is None and "t11875" in p and re.search(r"_b\d", p):
        q = re.search(r"_b(\d+(?:p\d+)?)", p)
        if q: beta = float(q.group(1).replace("p", "."))
    if "guppy2" in p:
        family = "Guppy2_T4"
    if "noisy" in p:
        family += "_noisy"
    return family, beta, params


def temperatures_for_raw(path):
    p = path.as_posix().lower()
    if "calibration_repeated/" in p:
        return (2.375, 4.0)  # Repeated calibration was run for these two targets.
    if "calibration/" in p and "calibration_repeated" not in p:
        return TARGETS  # Original script rescored each calibration sample at all T.
    if "t11875" in p:
        return (1.1875,)
    if "t_2375" in p or "_2375" in p:
        return (2.375,)
    if "t_4" in p or "_t4" in p or "guppy2" in p.lower():
        return (4.0,)
    return ()  # Avoid assigning an invented temperature to an unnamed raw run.


def collect(root):
    """Deduplicate identical 3-basis measurements even if stored many times."""
    samples, skipped = {}, []
    def add(arrays, relpath, temperatures, origin):
        if not temperatures:
            return
        key = digest(arrays)
        family, beta, params = source_metadata(relpath, root)
        if key not in samples:
            samples[key] = dict(id=key[:12], arrays=arrays, aliases=[], targets=set(),
                                family=family, beta0=beta, ABCD=params, origin=origin,
                                source=str(relpath))
        sample = samples[key]
        sample["aliases"].append(str(relpath))
        sample["targets"].update(float(t) for t in temperatures)
        # Prefer generated source metadata from raw experimental records.
        if origin == "raw" and sample["origin"] != "raw":
            sample.update(family=family, origin="raw", source=str(relpath))
        if sample["beta0"] is None and beta is not None:
            sample["beta0"] = beta
        if sample["ABCD"] is None and params is not None:
            sample["ABCD"] = params
    for f in sorted(root.rglob("*x.local.json")):
        if "hackathon_statistics" in f.parts:
            continue
        try:
            arrays = raw_triplet(f)
            if arrays is None: continue
            rel = f.parent.relative_to(root) / f.name
            # Use folder as record path for source lookup, filename for root name.
            add(arrays, rel, temperatures_for_raw(rel), "raw")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            skipped.append((str(f.relative_to(root)), str(e)))
    # Include packed submissions; exact temperature field defines target.
    for f in sorted(root.rglob("submission*.json")):
        if "hackathon_statistics" in f.parts or any(m in f.name for m in EXCLUDE_MARKERS):
            continue
        try:
            document = read_json(f)
            if set(document) != {"temperature", *BASES}:
                continue
            t = float(document["temperature"])
            if t not in TARGETS: continue
            arrays = validate_arrays(document)
            add(arrays, f.relative_to(root), (t,), "packed")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            skipped.append((str(f.relative_to(root)), str(e)))
    return samples, skipped


def features(arrays):
    """Per-SHOT observables (not 2048 falsely independent site measurements)."""
    out = {}
    for b in BASES:
        spins = 1.0 - 2.0 * arrays[b].astype(np.float64)
        mag = spins.mean(axis=1)  # 128 shot-averaged magnetizations
        corrs = (spins[:, None, :] * spins[:, score.TRANSLATED_SITES]).mean(axis=2)
        out[b] = np.column_stack((mag, corrs))  # (128,10)
    return out


def measure(sample, t, references, bootstrap=0, seed=2026):
    feats = features(sample["arrays"])
    measured = np.stack([feats[b].mean(axis=0) for b in BASES])
    ref_mag, ref_corr = references[t]
    diff_mag = measured[:, 0] - ref_mag
    diff_cor = measured[:, 1:] - ref_corr
    mag_score = float(9 * np.sum(diff_mag ** 2))
    pair_score = float(np.sum(diff_cor ** 2))
    # Independent SHOTS, not independent sites; spread within each shot preserved.
    se = np.stack([feats[b].std(axis=0, ddof=1) / np.sqrt(128) for b in BASES])
    noise_floor = float(9 * np.sum(se[:, 0] ** 2) + np.sum(se[:, 1:] ** 2))
    score_total = mag_score + pair_score
    dx = score.DISPLACEMENTS.index((1, 0))
    dy = score.DISPLACEMENTS.index((0, 1))
    energy = float(-measured[0, 0] - measured[2, 1 + dx] - measured[2, 1 + dy])
    ref_energy = float(-ref_mag[0] - ref_corr[2, dx] - ref_corr[2, dy])
    d = dict(id=sample["id"], temperature=t, source=sample["source"],
             family=sample["family"], beta0=sample["beta0"], ABCD=sample["ABCD"],
             score=score_total, mag_score=mag_score, pair_score=pair_score,
             corrected_score=score_total-noise_floor, estimated_shot_noise_floor=noise_floor,
             weighted_rms=math.sqrt(score_total/54), pair_rms=math.sqrt(pair_score/27),
             energy_per_site=energy, target_energy_per_site=ref_energy,
             energy_error=energy-ref_energy,
             X=float(measured[0, 0]), Y=float(measured[1, 0]), Z=float(measured[2, 0]),
             X_error=float(diff_mag[0]), Y_error=float(diff_mag[1]), Z_error=float(diff_mag[2]),
             X_shot_SE=float(se[0, 0]), Y_shot_SE=float(se[1, 0]), Z_shot_SE=float(se[2, 0]),
             worst_pair_error=float(np.max(np.abs(diff_cor))),
             aliases=sample["aliases"])
    if bootstrap:
        rng = np.random.default_rng(seed)
        sam = np.stack([feats[b][rng.integers(0, 128, size=(bootstrap, 128))].mean(axis=1)
                        for b in BASES], axis=1)  # (B,3,10)
        errors_m = sam[:, :, 0] - ref_mag
        errors_c = sam[:, :, 1:] - ref_corr
        boot = 9 * (errors_m ** 2).sum(axis=1) + (errors_c ** 2).sum(axis=(1, 2))
        d.update(bootstrap_n=bootstrap, bootstrap_median=float(np.median(boot)),
                 bootstrap_p025=float(np.percentile(boot, 2.5)),
                 bootstrap_p975=float(np.percentile(boot, 97.5)),
                 bootstrap_SD=float(np.std(boot, ddof=1)))
    return d


def stats_for_group(items):
    values = np.array([r["score"] for r in items], dtype=float)
    n = len(values)
    q05,q25,q50,q75,q95 = np.percentile(values, [5,25,50,75,95])
    iqr = q75-q25
    flags = ((values < q25-1.5*iqr) | (values > q75+1.5*iqr)) if n >= 5 else np.zeros(n, dtype=bool)
    std = float(np.std(values, ddof=1)) if n > 1 else None
    means = {f"mean_{k}": float(np.mean([r[k] for r in items])) for k in
             ("mag_score", "pair_score", "energy_error", "X", "Y", "Z", "corrected_score")}
    out = dict(temperature=items[0]["temperature"], family=items[0]["family"],
               beta0=items[0]["beta0"], ABCD=items[0]["ABCD"], n_independent=n,
               score_mean=float(np.mean(values)), score_SD=std,
               score_SE=std/math.sqrt(n) if std is not None else None,
               score_min=float(np.min(values)), score_max=float(np.max(values)),
               score_p05=float(q05), score_p25=float(q25), score_median=float(q50),
               score_p75=float(q75), score_p95=float(q95), outliers_1p5IQR=int(flags.sum()),
               outlier_sources=[r["source"] for r, flag in zip(items, flags) if flag],
               sources=[r["source"] for r in items], **means)
    for k in ("X", "Y", "Z"):
        out[f"{k}_across_run_SD"] = (float(np.std([r[k] for r in items], ddof=1)) if n > 1 else None)
    return out


def save_csv(path, items):
    if not items: return
    keys = list(dict.fromkeys(k for item in items for k in item.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for item in items:
            writer.writerow({k:(json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
                             for k,v in item.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Hackathon repository directory")
    parser.add_argument("--bootstrap", type=int, default=500, help="Resamples per selected submission; 0 disables")
    parser.add_argument("--focus", nargs="*", default=list(FINAL_HINTS),
                        help="Submitted/validation filenames to analyze in detail")
    parser.add_argument("--out", type=Path, default=Path("hackathon_statistics"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/"score.py").is_file() or not (root/"data/reference_g1.npz").is_file():
        parser.error("Run from the official hackathon folder containing score.py and data/reference_g1.npz")
    if args.bootstrap < 0: parser.error("--bootstrap must be nonnegative")
    references = {t: score.exact_observables(t) for t in TARGETS}
    samples, skipped = collect(root)
    if not samples:
        parser.error("No valid 128x16 X/Y/Z shot sets found under --root")
    rows = []
    for s in samples.values():
        for t in sorted(s["targets"]):
            r = measure(s, t, references)
            # Sanity check: official scorer agrees exactly with our per-shot aggregation.
            official = score.score_submission({"temperature": t,
                                               **{b:s["arrays"][b].tolist() for b in BASES}})
            if not np.isclose(r["score"], official["score"], atol=1e-10):
                raise RuntimeError("Internal statistics disagree with official score.py!")
            rows.append(r)
    # A 'group' is same experimental family AND same beta AND same ABCD AND same target.
    # This separates intentionally different trials from genuine independent repeats.
    grouped = defaultdict(list)
    for r in rows:
        if r["beta0"] is None or r["ABCD"] is None:
            continue  # Unknown configuration must not be silently grouped.
        group_key = (r["temperature"], r["family"], r["beta0"], r["ABCD"])
        grouped[group_key].append(r)
    summary = [stats_for_group(v) for v in grouped.values()]
    summary.sort(key=lambda x: (x["temperature"], -x["n_independent"], x["score_mean"]))
    focused = []
    for wanted in args.focus:
        matching = [s for s in samples.values() if any(Path(a).as_posix().lower().endswith(
            wanted.replace("\\", "/").lower()) for a in s["aliases"])]
        if not matching:
            print(f"  [not found] {wanted}")
            continue
        for s in matching:
            for t in sorted(s["targets"]):
                if t not in TARGETS: continue
                focused.append(measure(s, t, references, args.bootstrap,
                                       seed=2026 + int(s["id"][:6], 16) % 100000))
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out/"runs.csv", sorted(rows,key=lambda x:(x["temperature"], x["score"])))
    save_csv(args.out/"groups.csv", summary)
    save_csv(args.out/"focus.csv", focused)
    (args.out/"report.json").write_text(json.dumps({"settings":{
        "bootstrap":args.bootstrap,"targets":TARGETS,"root":str(root)},
        "unique_measurement_sets":len(samples),"per_target_results":len(rows),
        "excluded_reused_or_combined":list(EXCLUDE_MARKERS),
        "skipped_invalid_raw_or_packed":skipped,
        "runs":rows,"repeat_groups":summary,"focus":focused}, indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    print(f"\nReal measurement sets: {len(samples)} unique | scored rows: {len(rows)} | invalid: {len(skipped)}")
    print("Exact references and score formula: official score.py / data/reference_g1.npz")
    print("\nINDEPENDENT REPEATS: SD = sample std (ddof=1), SE = SD/sqrt(n)")
    print(f"{'T':>7} {'family':<25} {'beta0':>6} {'n':>3} {'mean':>9} {'SD':>9} {'SE':>9} {'p05':>9} {'p95':>9} {'out':>3}")
    repeated = [x for x in summary if x["n_independent"]>=2]
    for g in repeated:
        print(f"{g['temperature']:7g} {g['family'][:25]:<25} {g['beta0']:6g} {g['n_independent']:3d} "
              f"{g['score_mean']:9.5f} {g['score_SD']:9.5f} {g['score_SE']:9.5f} "
              f"{g['score_p05']:9.5f} {g['score_p95']:9.5f} {g['outliers_1p5IQR']:3d}")
    if not repeated:
        print("No groups with >=2 independent distinct 128-shot datasets and identifiable beta/ABCD.")
    print("\nFOCUS FILES: score = official 30-observable squared-error score")
    for r in focused:
        print(f"\n  {r['source']}  T={r['temperature']:g}  beta={r['beta0']}")
        print(f"    score={r['score']:.6f}  [mag={r['mag_score']:.6f}, pairs={r['pair_score']:.6f}]"
              f"  energy error/site={r['energy_error']:+.5f}")
        print(f"    X={r['X']:+.5f} +/- {r['X_shot_SE']:.5f} (shot SE);"
              f" Y={r['Y']:+.5f} +/- {r['Y_shot_SE']:.5f};"
              f" Z={r['Z']:+.5f} +/- {r['Z_shot_SE']:.5f}")
        print(f"    shot-noise expected contribution={r['estimated_shot_noise_floor']:.5f};"
              f" noise-corrected squared-error estimate={r['corrected_score']:.5f}")
        if args.bootstrap:
            print(f"    bootstrap 128-shot score: median={r['bootstrap_median']:.5f}, "
                  f"2.5%={r['bootstrap_p025']:.5f}, 97.5%={r['bootstrap_p975']:.5f}"
                  "  [NOT a Gibbs-error confidence interval]")
    print("\nTOP OBSERVED SCORES (may be affected by trial selection and shot noise):")
    for t in TARGETS:
        candidates = sorted((r for r in rows if r["temperature"] == t),key=lambda x:x["score"])
        if candidates:
            print(f"  T={t:g}: " + "; ".join(f"{r['score']:.5f} ({r['source']})" for r in candidates[:3]))
    print(f"\nWritten: {args.out/'report.json'}, runs.csv, groups.csv, focus.csv")
    print("No quantum simulator was called. No files used as measurements were altered.")
    print("Note: percentile/outlier estimates are unstable for n=5; n=2 gives no robust SD/CI.")

if __name__ == "__main__":
    main()
