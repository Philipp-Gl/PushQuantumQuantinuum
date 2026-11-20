# Exact score references

`reference_g1.npz` contains exact Gibbs expectation values for the 4x4 periodic
Pauli Ising Hamiltonian H=-sum_<ij> ZiZj-sum_i Xi, with 32 bonds and J=kB=1.
The temperatures are 0.01, 1.1875, 2.375, and 4.0.

Arrays:

- `temperatures`: shape (4,).
- `pairs`: shape (120,2), lexicographic unordered pairs i<j, sites q=x+4*y.
- `magnetizations`: shape (4,3), axes X,Y,Z, site-averaged expectations.
- `correlations`: shape (4,3,120), expectations of Ai Aj, axes X,Y,Z.

These were calculated from the existing full-spectrum exact-diagonalization
checkpoint `results/g_1.000_spectrum.npz` in the parent research project. For
each temperature, eigenstate observables are averaged with normalized weights
`multiplicity * exp[-(E-E_min)/T]`. All 65,536 eigenstates are included through
their symmetry multiplicities. The T=0.01 reference is recomputed at that
temperature; it is not the earlier nearly pure T=1e-8 target.

Exact one-site Y and Z means vanish by symmetry. Exact pair values are expanded
from the checkpoint's translation- and square-symmetry-averaged XX/YY/ZZ classes
to every individual pair. This is valid for the symmetric Gibbs reference. The
scorer averages submitted correlations over translations for nine displacement
representatives modulo d ~ -d, without pooling rotations or reflections. It
converts the stored pair references to the same nine translation averages.
The source checkpoint's SHA-256 is stored in `source_sha256` inside the archive.
No GPU, diagonalization, or private project files are needed to run scoring.
