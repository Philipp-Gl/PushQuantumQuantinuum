# Tilted-product example

This baseline fits exact g=1 transverse magnetization and squared Z magnetization.
It prepares a separable approximation, not an exact quantum Gibbs state. The
classical enumeration and exact-data calibration do not provide a scalable
temperature-only thermal-state preparation method.

Run these commands from this `examples/` directory. uv uses the shared
environment defined in the repository root.

```bash
# Validate the three T=2.375 programs through the H2 checker, without Nexus access.
uv run --locked submit.py --compile-only --output results/compile_check.json

# Noiseless local simulation: T=2.375, three bases, 128 shots each.
uv run --locked submit.py --local

# Nexus H2-1E execution (uses your account's emulator credits).
# Login opens a browser only if your cached Nexus login is absent or expired.
uv run --locked submit.py
```

The Nexus command waits for completion and writes
`results/tilted_product_h2_1e.json`. If interrupted after submission, resume the
same job with `uv run --locked submit.py --resume results/tilted_product_h2_1e.json`.
Use `--temperature 2.375 --basis Z --output results/smoke.json` for one experiment.
H2-1E uses its default noise model; the local Selene run is noiseless and clearly
identified as such in its separate JSON output.

`create_guppy.py` provides `build_program(T, basis, seed)`, a Guppy `body`, and
separate `measure_x`, `measure_y`, `measure_z` functions. A zero-argument `main`
combines body with the chosen measurement. Run `uv run --locked create_guppy.py`
to export one HUGR program without submission.

NumPy cannot execute on the device. `body` calls a `@guppy.comptime` helper that
calls `sample_labels`: 128 whole classical configurations are independently drawn on
the host, then embedded into the program. `get_current_shot()` selects
the appropriate configuration. H2 reports indices 1..128, so the H2 program
subtracts 1; Selene reports 0..127 and uses offset 0. This difference was checked
on H2-1E. Submit exactly 128 shots with the backend's default shot numbering;
recompiling with the same seed deliberately reproduces the same sample table.
Re-executing an existing compiled program reuses those labels. The emitted
`shot` and `label` outputs are checked against the table when collecting results.
The sample table is packed into two 64-bit masks per qubit so H2 can select the
bit for the current shot without unsupported dynamic array indexing. Measurement
basis changes are expanded at compile time for the same compatibility reason.

For sampled spin s_q=+1 (bit 0), apply Ry(theta); for s_q=-1 (bit 1), apply
Ry(pi-theta). The label distribution is exp[-k E_classical(s)]/Z, where k is a
fitted parameter, not necessarily 1/T. A shot produces a pure product branch;
averaging the independently sampled branches produces the mixture.

The supplied `data/ising_spectra.npz` is a compact subset of the parent project's
exact-diagonalization data: the g=0 computational-basis energies in integer-label
order, and g=1 eigenenergies, symmetry multiplicities, and eigenstate mx/mz2.
Never interpret the g=1 eigenstate index as a computational-basis bitstring.
At T=0.01 the calibration is recomputed from the finite-temperature spectrum;
the earlier T=1e-8 coherent-cat benchmark is not used.

Each JSON experiment records T, basis, calibration parameters, sampled labels,
and 128 measurements. `bits[q]` corresponds to site q=x+4*y; bit b means measured
Pauli eigenvalue 1-2*b. X readout applies H; Y readout applies S-dagger then H.
All basis settings measure all 16 qubits. The default example uses T=2.375 only:
three bases times 128 shots gives 384 total shots. Other challenge temperatures
can be selected explicitly with `--temperature`.

H2 conversion and submission follow the current
[hugr-qir workflow](https://quantinuum.github.io/hugr-qir/getting-started.html).
The supported subset includes shot indices, fixed loops, and measurement arrays;
see the [H2 support matrix](https://quantinuum.github.io/hugr-qir/guppy-for-h2/support-matrix.html).
