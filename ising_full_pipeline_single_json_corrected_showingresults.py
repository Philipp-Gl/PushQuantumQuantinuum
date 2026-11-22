# -*- coding: utf-8 -*-
"""4x4 transverse-field Ising: analyze ONE existing submission JSON and plot scores.

Spyder: edit CONFIG below and click Run. Dependencies: numpy scipy matplotlib.

Input: one submission JSON with temperature, X, Y, Z; each basis 128x16 bits.
Change only JSON_FILE in CONFIGURATION. Filename need not start with T_.

IMPORTANT: Gibbs reference is stochastic typicality, not exact competition
reference. This script never generates or deletes submission JSONs.
If HX or J changes, set RECOMPUTE_REFERENCE=True to avoid stale cache.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import expm_multiply, eigsh

# ========================== CONFIGURATION ================================
BASE = Path(__file__).resolve().parent
# ONLY CHANGE THIS: filename in the script folder, or a complete Windows path.
JSON_FILE = "submission_VQE_001.json"
# Example: JSON_FILE = r"C:\Users\YourName\Downloads\friend_measurements.json"
HX = -1.0
J = -1.0                         # H = J sum_<ij> ZiZj + HX sum_i Xi
N = 16
L = 4
SHOTS = 128
DATA_LABEL = 'SUBMITTED MEASUREMENTS'
OUTPUT_DIR = BASE / 'score_analysis'
REFERENCE_CSV = OUTPUT_DIR / 'gibbs_30_observables.csv'
ENERGY_CSV = OUTPUT_DIR / 'exact_energy.csv'
RECOMPUTE_REFERENCE = True
TYPICALITY_VECTORS = 12
BOOTSTRAP_REPEATS = 1000
SEED = 2026
SHOW_PLOTS = True
# Temperatures are detected from the JSON, NOT entered manually.
TEMPERATURES = []
# ==========================================================================

# Nine displacements, no C4 averaging, as specified in the score definition.
DISPLACEMENTS = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                 (1, 3), (2, 0), (2, 1), (2, 2)]
AXES = 'XYZ'
OBSERVABLES = [f'm_{a}' for a in AXES] + [f'C_{a}_{dx}_{dy}'
                 for a in AXES for dx, dy in DISPLACEMENTS]


def neighbor_pairs():
    """Unique horizontal and vertical bonds with periodic boundaries."""
    return [(x + L*y, ((x+1) % L) + L*y) for y in range(L) for x in range(L)] + [
        (x + L*y, x + L*((y+1) % L)) for y in range(L) for x in range(L)]


def displacement_indices(dx, dy):
    i = np.arange(N, dtype=int)
    x, y = i % L, i // L
    j = (x + dx) % L + L*((y + dy) % L)
    return i, j


def read_json(path):
    with path.open(encoding='utf-8') as f:
        obj = json.load(f)
    if not all(k in obj for k in ('temperature', 'X', 'Y', 'Z')):
        raise ValueError(f'{path}: missing temperature/X/Y/Z')
    T = float(obj['temperature'])
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f'{path}: invalid temperature')
    arrays = {}
    for a in AXES:
        bits = np.asarray(obj[a])
        if bits.shape != (SHOTS, N) or not np.all((bits == 0) | (bits == 1)):
            raise ValueError(f'{path}: {a} must have 128 rows x 16 binary bits')
        arrays[a] = 1 - 2*bits.astype(np.int8)
    return T, arrays


def observable_shots(arrays):
    """Return (128,30) shot-wise spatially averaged observables."""
    cols = []
    for a in AXES:
        cols.append(arrays[a].mean(axis=1))
    for a in AXES:
        spins = arrays[a]
        for dx, dy in DISPLACEMENTS:
            i, j = displacement_indices(dx, dy)
            cols.append((spins[:, i] * spins[:, j]).mean(axis=1))
    return np.stack(cols, axis=1)


def make_hamiltonian():
    """Sparse 2^16 x 2^16 Ising Hamiltonian, bit q=x+4*y."""
    dim = 1 << N
    states = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=np.float64)
    for i, j in neighbor_pairs():
        zi = 1 - 2*((states >> i) & 1)
        zj = 1 - 2*((states >> j) & 1)
        diag += J*zi*zj
    rows = np.tile(states, N)
    cols = np.concatenate([states ^ (1 << i) for i in range(N)])
    off = csr_matrix((np.full(N*dim, HX), (rows, cols)), shape=(dim, dim))
    return diags(diag, format='csr') + off


def pauli_expectation(w, axis, q1, q2=None):
    """<w|P|w> for P = X/Y/Z at one or two sites; w unnormalized.

    The phases for Y follow Y|0>=i|1>, Y|1>=-i|0>.
    """
    idx = np.arange(len(w), dtype=np.int64)
    sites = [q1] if q2 is None else [q1, q2]
    if axis == 'Z':
        eigen = np.ones(len(w), dtype=np.float64)
        for q in sites:
            eigen *= 1 - 2*((idx >> q) & 1)
        return float(np.vdot(w, eigen*w).real)
    flipped = idx.copy()
    for q in sites:
        flipped ^= 1 << q
    if axis == 'X':
        return float(np.vdot(w, w[flipped]).real)
    # Y: <w|Y|w> = sum conj(w[b]) * (-i)(-1)^b w[b xor 1]
    phase = np.ones(len(w), dtype=np.complex128)
    for q in sites:
        phase *= -1j * (1 - 2*((idx >> q) & 1))
    return float(np.vdot(w, phase*w[flipped]).real)


def thermal_observables(w):
    """Spatial averages of the 30 Pauli strings in one thermal vector."""
    vals = []
    for a in AXES:
        vals.append(sum(pauli_expectation(w, a, i) for i in range(N))/N)
    for a in AXES:
        for dx, dy in DISPLACEMENTS:
            i_arr, j_arr = displacement_indices(dx, dy)
            vals.append(sum(pauli_expectation(w, a, int(i), int(j))
                            for i, j in zip(i_arr, j_arr))/N)
    return np.array(vals)


def compute_gibbs_reference():
    """Stochastic trace via Rademacher vectors: <O> = Tr(O exp(-H/T))/Z.

    For each vector v, w = exp(-H/(2T)) v. Summing <w|O|w> / <w|w>
    across vectors gives a ratio-of-traces estimate. Reference uncertainty
    is approximated by leave-one-vector-out jackknife.
    """
    H = make_hamiltonian()
    dim = 1 << N
    E0 = float(eigsh(H, k=1, which='SA', return_eigenvectors=False, tol=1e-9)[0])
    H_shifted = H - E0 * identity(dim, format='csr')
    print(f'Ground-state energy: {E0:.8f}')
    rng = np.random.default_rng(SEED)
    # Reuse vectors across T to reduce reference-to-reference jitter.
    vs = [rng.choice([-1.0, 1.0], size=dim) for _ in range(TYPICALITY_VECTORS)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for T in TEMPERATURES:
        print(f'Gibbs reference at T={T:g} ({TYPICALITY_VECTORS} vectors)...', flush=True)
        num, den = [], []
        for k, v in enumerate(vs):
            w = expm_multiply((-0.5/T)*H_shifted, v)
            norm = float(np.vdot(w, w).real)
            num.append(thermal_observables(w))
            den.append(norm)
        # Important: avoid exponential-weight overflow via per-vector scaling.
        den = np.asarray(den, dtype=float)
        num = np.asarray(num, dtype=float)
        if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)) or den.sum() <= 0:
            raise RuntimeError(f'Nonfinite Gibbs trace estimates at T={T:g}')
        ref = num.sum(axis=0)/den.sum()
        if not np.all(np.isfinite(ref)):
            raise RuntimeError(f'Nonfinite Gibbs reference at T={T:g}')
        if len(den) >= 3:
            leave = (num.sum(axis=0)[None, :] - num)/(den.sum() - den)[:, None]
            se = np.sqrt((len(den)-1)/len(den) * np.sum((leave-leave.mean(axis=0))**2, axis=0))
        else:
            se = np.full(len(OBSERVABLES), np.nan)
        rows.append([T, *ref, *se])
    with REFERENCE_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['temperature', *OBSERVABLES, *[s+'_se' for s in OBSERVABLES]])
        writer.writerows(rows)
    print('Saved Gibbs reference:', REFERENCE_CSV)
    save_energy_csv(rows)


def save_energy_csv(rows):
    """Export energy/site from the SAME Gibbs 30-observable reference.

    E/N = J [C_Z(1,0)+C_Z(0,1)] + HX*m_X.
    This CSV is an estimated reference, not exact diagonalization.
    """
    with ENERGY_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['temperature', 'energy_per_site'])
        for row in rows:
            observables = np.asarray(row[1:1+len(OBSERVABLES)], dtype=float)
            writer.writerow([row[0], energy_from_observables(observables)])
    print('Saved Gibbs energy CSV:', ENERGY_CSV)


def ensure_energy_csv(reference):
    """Re-create energy CSV if using an existing 30-observable reference."""
    with ENERGY_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['temperature', 'energy_per_site'])
        for T in sorted(reference):
            writer.writerow([T, energy_from_observables(reference[T])])
    print('Saved Gibbs energy CSV:', ENERGY_CSV)


def load_reference():
    """Reuse existing Gibbs values when possible; compute missing T only."""
    existing_rows = []
    if REFERENCE_CSV.exists() and not RECOMPUTE_REFERENCE:
        with REFERENCE_CSV.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required = {'temperature', *OBSERVABLES}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Reference CSV lacks one or more of the 30 observables')
            existing_rows = list(reader)

    existing_t = [float(row['temperature']) for row in existing_rows]
    missing = [t for t in TEMPERATURES if not any(
        np.isclose(t, old, rtol=0, atol=1e-6) for old in existing_t)]
    if RECOMPUTE_REFERENCE or missing:
        # Compute only requested temperatures, then merge old cached rows.
        if missing:
            print(f'Gibbs reference missing T={missing}; computing it now.')
        compute_gibbs_reference()
        with REFERENCE_CSV.open(newline='', encoding='utf-8-sig') as f:
            new_rows = list(csv.DictReader(f))
        if not RECOMPUTE_REFERENCE and existing_rows:
            new_t = [float(row['temperature']) for row in new_rows]
            merged = [r for r in existing_rows if not any(
                np.isclose(float(r['temperature']), t, rtol=0, atol=1e-6)
                for t in new_t)] + new_rows
            fieldnames = ['temperature', *OBSERVABLES,
                          *[o+'_se' for o in OBSERVABLES]]
            with REFERENCE_CSV.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged)
            existing_rows = merged
        else:
            existing_rows = new_rows

    reference = {}
    for row in existing_rows:
        T = float(row['temperature'])
        if T in reference:
            raise ValueError(f'Duplicate Gibbs temperature {T}')
        reference[T] = np.array([float(row[k]) for k in OBSERVABLES])
    return reference


def reference_at(reference, T):
    matches = [k for k in reference if np.isclose(k, T, atol=1e-6, rtol=0)]
    if len(matches) != 1:
        raise ValueError(f'No unique Gibbs reference for T={T:g}; available {sorted(reference)}')
    return reference[matches[0]]


def score_components(measured, reference):
    """Official score weights: 9 for 3 magnetizations, 1 for 27 correlations."""
    weights = np.array([9.0]*3 + [1.0]*27)
    return weights*(measured-reference)**2


def bootstrap_scores(arrays, reference, rng):
    """Resample each basis independently, retaining within-shot correlations."""
    shot_obs = observable_shots(arrays)
    point = shot_obs.mean(axis=0)
    boots = np.zeros((BOOTSTRAP_REPEATS, len(OBSERVABLES)))
    for ai, a in enumerate(AXES):
        ids = rng.integers(0, SHOTS, size=(BOOTSTRAP_REPEATS, SHOTS))
        cols = [ai] + list(range(3+9*ai, 3+9*(ai+1)))
        boots[:, cols] = shot_obs[ids][:, :, cols].mean(axis=1)
    component = score_components(point, reference)
    score_samples = np.sum(score_components(boots, reference), axis=1)
    return {
        'observables': point,
        'component': component,
        'score': float(component.sum()),
        'score_lo': float(np.quantile(score_samples, .025)),
        'score_hi': float(np.quantile(score_samples, .975)),
        'boot_scores': score_samples,
    }


def load_selected_dataset():
    """Load exactly one file, with arbitrary filename, no deletion or generation."""
    path = Path(JSON_FILE).expanduser()
    if not path.is_absolute():
        path = BASE / path
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}\n"
            "Put it beside this script or use its full path in JSON_FILE."
        )
    T, arrays = read_json(path)
    return path, T, arrays


def energy_from_observables(o):
    """E/N = J*(C_Z(1,0)+C_Z(0,1)) + HX*m_X.

    C_Z displacement is the spatial average over 16 bonds in that direction.
    """
    return J*(o[OBSERVABLES.index('C_Z_1_0')] +
              o[OBSERVABLES.index('C_Z_0_1')]) + HX*o[0]


def save_scores_csv(all_results):
    path = OUTPUT_DIR/'scores.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'temperature', 'score', 'bootstrap_95_lo', 'bootstrap_95_hi',
                         'magnetization_score', 'correlation_score'])
        for name, data in all_results.items():
            for T, r in sorted(data.items()):
                writer.writerow([name, T, r['score'], r['score_lo'], r['score_hi'],
                                 r['component'][:3].sum(), r['component'][3:].sum()])
    print('Saved:', path)


def make_plots(all_results, reference):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = np.array(sorted(TEMPERATURES), dtype=float)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for name, records in all_results.items():
        scores = np.array([records[T]['score'] for T in ts])
        lo = np.array([records[T]['score_lo'] for T in ts])
        hi = np.array([records[T]['score_hi'] for T in ts])
        axs[0, 0].errorbar(ts, scores, yerr=[np.maximum(scores-lo,0), np.maximum(hi-scores,0)],
                           fmt='o-', capsize=3, label=name)
        axs[0, 1].plot(ts, [records[T]['component'][:3].sum() for T in ts], 'o-', label=f'{name}: magnetization')
        axs[0, 1].plot(ts, [records[T]['component'][3:].sum() for T in ts], 's--', label=f'{name}: correlations')
        e_meas = [energy_from_observables(records[T]['observables']) for T in ts]
        axs[1, 0].plot(ts, e_meas, 'o-', label=f'{name}: measured')
    e_gibbs = [energy_from_observables(reference_at(reference, T)) for T in ts]
    axs[1, 0].plot(ts, e_gibbs, 'k--', label='Gibbs reference')
    axs[0, 0].set(title='Official 30-observable score (lower is better)', xlabel='Target temperature T', ylabel='Weighted squared error')
    axs[0, 1].set(title='Where does the score come from?', xlabel='Target temperature T', ylabel='Score contribution')
    axs[1, 0].set(title='Energy density (secondary diagnostic)', xlabel='Target temperature T', ylabel='E/N')
    # Heatmap: weighted per-observable errors for best dataset at central T.
    mid = float(ts[len(ts)//2])
    best_name = min(all_results, key=lambda n: all_results[n][mid]['score'])
    r = all_results[best_name][mid]
    matrix = np.vstack([r['component'][3+9*i:3+9*(i+1)] for i in range(3)])
    im = axs[1, 1].imshow(matrix, aspect='auto', cmap='magma')
    axs[1, 1].set(title=f'Correlation error contributions: {best_name}, T={mid:g}',
                   xlabel='Displacement (dx,dy)', ylabel='Basis')
    axs[1, 1].set_yticks(range(3), AXES)
    axs[1, 1].set_xticks(range(9), [str(d) for d in DISPLACEMENTS], rotation=45, ha='right')
    fig.colorbar(im, ax=axs[1, 1], label='Squared correlation error')
    for ax in (axs[0,0], axs[0,1], axs[1,0]):
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle(f'4x4 Ising diagnostics | {DATA_LABEL} | periodic | 128 shots/basis', fontsize=13)
    path = OUTPUT_DIR/'score_dashboard.png'
    fig.savefig(path, dpi=170)
    print('Saved:', path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    # Extra graph: all 30 observables at one temperature for best dataset.
    o = r['observables']
    ref = reference_at(reference, mid)
    fig, ax = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    xx = np.arange(30)
    ax.plot(xx, ref, 'ko', markersize=4, label='Gibbs reference')
    ax.plot(xx, o, 'x', markersize=6, label=f'{best_name}: measured')
    ax.axvline(2.5, color='gray', linestyle=':')
    ax.set_xticks(xx, OBSERVABLES, rotation=90)
    ax.set(title=f'All 30 observables at T={mid:g}', ylabel='Expectation value', xlabel='Observable')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    path = OUTPUT_DIR/'all_30_observables.png'
    fig.savefig(path, dpi=170)
    print('Saved:', path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def main():
    # Read the user's chosen file first; it determines the target temperature.
    path, T, arrays = load_selected_dataset()
    global TEMPERATURES
    TEMPERATURES = [T]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Input file: {path}')
    print(f'Target temperature from JSON: T={T:g}')
    print('No input JSON files will be created, changed or deleted.')
    ref = load_reference()
    ensure_energy_csv(ref)
    rng = np.random.default_rng(SEED + 1)
    record = bootstrap_scores(arrays, reference_at(ref, T), rng)
    all_results = {DATA_LABEL: {T: record}}
    save_scores_csv(all_results)
    make_plots(all_results, ref)
    print(f"\nT={T:g}: score={record['score']:.6f}, "
          f"bootstrap 95%=[{record['score_lo']:.6f}, {record['score_hi']:.6f}]")
    print(f'Outputs: {OUTPUT_DIR}')
    print('Gibbs reference is stochastic; bootstrap intervals include shot noise only.')
    print('A low observable score is not a proof of full Gibbs-state fidelity.')


if __name__ == '__main__':
    main()
