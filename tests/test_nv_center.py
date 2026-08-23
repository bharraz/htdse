"""NV center suite verification: `python tests/test_nv_center.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htdse.submodules.nv_center import (zero_field_splitting, zeeman, hyperfine_tensor,
                                        isc_dephasing, NVCenter)

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

D = 2.87e9  # Hz, commonly cited NV ground-state zero-field splitting

print("== zero_field_splitting: ms=0 <-> ms=+-1 gap is exactly D ==")
M = np.asarray(zero_field_splitting(D).hamiltonian(0))
check("ZFS matrix is diagonal", np.allclose(M, np.diag(np.diag(M))))
check("ZFS matrix is Hermitian", np.allclose(M, M.conj().T))
evals = np.sort(np.diag(M).real)
check("ms=+-1 are degenerate at E=0", np.isclose(evals[1], evals[2]))
check("ms=0 <-> ms=+-1 splitting == D", np.isclose(evals[2] - evals[0], D))

print("== E term (strain) splits the ms=+-1 degeneracy by exactly 2E ==")
E = 5e6
M2 = np.asarray(zero_field_splitting(D, E=E).hamiltonian(0))
evals2 = np.sort(np.linalg.eigvalsh(M2))
check("ms=0 is an exact eigenstate untouched by E (Sx^2-Sy^2 only connects "
      "states 2 apart in m, and ms=0 has no such partner in the S=1 manifold)",
      np.isclose(evals2[0], -2 * D / 3))
check("top two eigenvalues split by 2E", np.isclose(evals2[2] - evals2[1], 2 * E))

print("== zeeman: linear shift +g*Bz/0/-g*Bz for ms=+1/0/-1 ==")
g_e, Bz = 2.0028, 1e6
Mz = np.asarray(zeeman(g_e, Bz).hamiltonian(0))
check("zeeman is diagonal", np.allclose(Mz, np.diag(np.diag(Mz))))
check("zeeman diagonal == [g*Bz, 0, -g*Bz]",
      np.allclose(np.diag(Mz).real, [g_e * Bz, 0, -g_e * Bz]))

print("== hyperfine_tensor: dimension, Hermiticity, symmetric-tensor structure ==")
Mh = np.asarray(hyperfine_tensor(-2.14e6, -2.7e6, I=1.0).hamiltonian(0))
check("hyperfine_tensor(I=1) shape is (9,9)", Mh.shape == (9, 9))
check("hyperfine_tensor is Hermitian", np.allclose(Mh, Mh.conj().T))
Mh_half = np.asarray(hyperfine_tensor(1.0, 1.0, I=0.5).hamiltonian(0))
check("hyperfine_tensor(I=1/2) shape is (6,6)", Mh_half.shape == (6, 6))

print("== NVCenter: end-to-end composition ==")
nv = NVCenter(D=D, nuclear_spins={"N14": 1.0})
check("NVCenter.subsystems includes electron (dim 3) and N14 (dim 3)",
      nv.subsystems == {"e": 3, "N14": 3})
check("NVCenter.dim == 9", nv.dim == 9)
Hfull = nv.hamiltonian(Bz=1e5, hyperfine={"N14": (-2.14e6, -2.7e6)})
Mfull = np.asarray(Hfull.hamiltonian(0))
check("NVCenter.hamiltonian() is Hermitian", np.allclose(Mfull, Mfull.conj().T))
check("NVCenter.hamiltonian() shape matches dim", Mfull.shape == (9, 9))

raised = False
try:
    nv.hamiltonian(hyperfine={"C13": (1.0, 1.0)})
except KeyError:
    raised = True
check("NVCenter.hamiltonian raises KeyError for an unknown nuclear spin", raised)

print("== isc_dephasing: rate-scaled projectors, zero rate skipped ==")
L = isc_dephasing((1e6, 0.0, 1e6))
jumps = L.jump_operators(0)
check("isc_dephasing(ms0 rate=0) produces exactly 2 jump operators (ms=+1,-1 only)",
      len(jumps) == 2)
check("isc_dephasing jump ops are sqrt(rate)-scaled projectors",
      all(np.allclose(j, j.conj().T) and np.allclose(j @ j, j * 1000.0) for j in jumps))

raised = False
try:
    isc_dephasing((1.0, 1.0))
except ValueError:
    raised = True
check("isc_dephasing raises for wrong-length rates", raised)

print(f"\n{len(PASS)} checks passed.")
