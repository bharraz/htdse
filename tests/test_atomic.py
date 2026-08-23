"""Atomic structure suite verification: `python tests/test_atomic.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htdse.submodules.atomic import (g_sum, g_s, g_l, hyperfine, spin_manifold,
                                     couple_reduced_element, dipole_couple_matrix,
                                     hyperfine_matrix)
from htdse.submodules.angular_momentum import clebsch_gordan

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

print("== g-factors (AMO.jl fixtures) ==")
check("g_s ~ 2 + 1/137/pi", np.isclose(g_s, 2 + 1 / 137 / np.pi, atol=0.5e-4))
check("g_l == 1", g_l == 1.0)
check("g_sum(2, 1/2, 2, 3/2, 0) = 1/2", np.isclose(g_sum(2, 0.5, 2, 1.5, 0), 0.5))
check("g_sum(1, 1/2, 2, 3/2, 0) = -1/2", np.isclose(g_sum(1, 0.5, 2, 1.5, 0), -0.5))
check("g_sum(2, 3/2, 0, 1/2, 2) = 1/2", np.isclose(g_sum(2, 1.5, 0, 0.5, 2), 0.5))
check("g_sum(1, 3/2, 0, 1/2, 2) = -1/2", np.isclose(g_sum(1, 1.5, 0, 0.5, 2), -0.5))
check("g_sum(3/2, 1, 1, 1/2, 2) = 4/3", np.isclose(g_sum(1.5, 1, 1, 0.5, 2), 4 / 3))
check("g_sum(3/2, 1/2, 2, 1, 1) = 4/3", np.isclose(g_sum(1.5, 0.5, 2, 1, 1), 4 / 3))

print("== hyperfine: Sodium D-line fixtures (AMO.jl test/atomic.jl) ==")
AS1 = 885.81306440
check("hyperfine(F=2, S1/2)", np.isclose(hyperfine(2, 1.5, 0.5, AS1, 0), 664.35979830, atol=5e-7))
check("hyperfine(F=1, S1/2)", np.isclose(hyperfine(1, 1.5, 0.5, AS1, 0), -1107.26633050, atol=5e-7))
AP1 = 94.44
check("hyperfine(F=2, P1/2)", np.isclose(hyperfine(2, 1.5, 0.5, AP1, 0), 70.830, atol=0.01))
check("hyperfine(F=1, P1/2)", np.isclose(hyperfine(1, 1.5, 0.5, AP1, 0), -118.05, atol=0.01))
AP3, BP3 = 18.534, 2.724
check("hyperfine(F=3, P3/2)", np.isclose(hyperfine(3, 1.5, 1.5, AP3, BP3), 42.382, atol=0.01))
check("hyperfine(F=2, P3/2)", np.isclose(hyperfine(2, 1.5, 1.5, AP3, BP3), -15.944, atol=0.01))
check("hyperfine(F=1, P3/2)", np.isclose(hyperfine(1, 1.5, 1.5, AP3, BP3), -50.288, atol=0.01))
check("hyperfine(F=0, P3/2)", np.isclose(hyperfine(0, 1.5, 1.5, AP3, BP3), -66.097, atol=0.01))

print("== spin_manifold: exact AMO.jl expected sequences ==")
expected = [(1, 1, 0.5), (1, 1, 1.5), (1, 1, 2.5), (1, 2, 0.5), (1, 2, 1.5), (1, 2, 2.5),
            (1, 2, 3.5), (1, 3, 1.5), (1, 3, 2.5), (1, 3, 3.5), (1, 3, 4.5)]
got = list(spin_manifold(1, 2, 1.5))
check(f"spin_manifold(1,2,1.5) == AMO.jl's 11-tuple sequence", got == expected)
check("spin_manifold(10) == [(10,)]", list(spin_manifold(10)) == [(10,)])
check("spin_manifold() == []", list(spin_manifold()) == [])

print("== couple_reduced_element: cross-validated via direct CG-sum (AMO.jl check_coupling pattern) ==")
def coupling0(j0, j1, j2, j1p, m1p, j2p, m2p, k, q):
    total = 0.0
    for m0 in np.arange(-j0, j0 + 0.5):
        for m1 in np.arange(-j1, j1 + 0.5):
            for m2 in np.arange(-j2, j2 + 0.5):
                total += (clebsch_gordan(j1, m1, j0, m0, j1p, m1p)
                          * clebsch_gordan(j2, m2, j0, m0, j2p, m2p)
                          * clebsch_gordan(j2, m2, 1, q, j1, m1))
    return total

n_checked = 0
for j0 in (0.5, 1, 1.5):
    for j1 in (0.5, 1):
        for j2 in (0.5, 1):
            if abs(j1 - j2) > 1 or j1 + j2 < 1:
                continue
            if abs(round(j1 - j2) - (j1 - j2)) > 1e-9:
                continue  # rank-1 tensor needs integer j1-j2 (integer q)
            for j1p in np.arange(abs(j1 - j0), j1 + j0 + 0.5):
                for j2p in np.arange(abs(j2 - j0), j2 + j0 + 0.5):
                    vf = couple_reduced_element(j1p, j2p, j0, j1, j2, 1)
                    if abs(j1p - j2p) > 1 or j1p + j2p < 1:
                        check(f"couple_reduced_element=0 outside triangle "
                              f"(j1p={j1p},j2p={j2p})", vf == 0.0)
                        continue
                    for m1p in np.arange(-j1p, j1p + 0.5):
                        for m2p in np.arange(-j2p, j2p + 0.5):
                            q = m1p - m2p
                            if abs(q) > 1:
                                continue
                            v0 = coupling0(j0, j1, j2, j1p, m1p, j2p, m2p, 1, q)
                            v1 = vf * clebsch_gordan(j2p, m2p, 1, q, j1p, m1p)
                            n_checked += 1
                            if not np.isclose(v0, v1, atol=1e-8):
                                check(f"coupling0 == R*CG for j0={j0},j1={j1},j2={j2},"
                                      f"j1p={j1p},j2p={j2p},m1p={m1p},m2p={m2p}", False)
check(f"couple_reduced_element matches direct CG-sum recoupling over {n_checked} combos",
      True)

print("== dipole_couple_matrix: structural sanity checks ==")
M = dipole_couple_matrix(1, 0, (0.0, 1.0, 0.0))  # pi-polarized L=1 -> L=0
check("dipole_couple_matrix(1,0,pi) shape (3,1)", M.shape == (3, 1))
check("dipole_couple_matrix(1,0,pi): only mJ1=0 -> mJ2=0 nonzero",
      abs(M[1, 0]) > 1e-9 and abs(M[0, 0]) < 1e-12 and abs(M[2, 0]) < 1e-12)

M_spin = dipole_couple_matrix(1, 0, (0.0, 1.0, 0.0), S=(0.5,))
check("dipole_couple_matrix with spectator spin 1/2 doubles both dimensions",
      M_spin.shape == (6, 2))

raised = False
try:
    dipole_couple_matrix(2, 0, (0.0, 1.0, 0.0))  # |L1-L2|=2, not dipole-coupled
except ValueError:
    raised = True
check("dipole_couple_matrix raises for |L1-L2|>1", raised)

print("== hyperfine_matrix: Hermiticity + Bm=0 block-diagonal reduction ==")
I, J = 1.5, 0.5  # sodium S1/2-like
M0 = hyperfine_matrix(I, J, Bm=0.0, Ahf=AS1)
check("hyperfine_matrix(Bm=0) is Hermitian", np.allclose(M0, M0.conj().T))
check("hyperfine_matrix(Bm=0) is diagonal", np.allclose(M0, np.diag(np.diag(M0))))
diag_vals = sorted(set(np.round(np.diag(M0).real, 4)))
expected_vals = sorted({hyperfine(2, I, J, AS1, 0), hyperfine(1, I, J, AS1, 0)})
check("hyperfine_matrix(Bm=0) diagonal reduces to hyperfine(F,...) for each F",
      len(diag_vals) == len(expected_vals)
      and all(np.isclose(a, b) for a, b in zip(diag_vals, expected_vals)))

M_field = hyperfine_matrix(I, J, Bm=1.0, g_I=0.0002, g_J=2.0, Ahf=AS1)
check("hyperfine_matrix(Bm!=0) is Hermitian", np.allclose(M_field, M_field.conj().T))
check("hyperfine_matrix(Bm!=0) mixes F blocks (off-diagonal nonzero)",
      not np.allclose(M_field, np.diag(np.diag(M_field))))

print("== hyperfine_matrix: stretched state |F_max, mF=F_max> has no F-mixing ==")
Bm, g_I, g_J = 0.001, 0.0002, 2.0023
M_stretched = hyperfine_matrix(I, J, Bm=Bm, g_I=g_I, g_J=g_J, Ahf=AS1)
F_max = I + J
idx_last = M_stretched.shape[0] - 1  # stretched state is the very last basis index
expected_shift = hyperfine(F_max, I, J, AS1, 0) + g_J * Bm * J + g_I * Bm * I
check("stretched state shift = hyperfine(F_max) + g_J*Bm*J + g_I*Bm*I exactly",
      np.isclose(M_stretched[idx_last, idx_last].real, expected_shift))
check("stretched state has zero off-diagonal coupling (pure product state)",
      np.isclose(np.abs(M_stretched[idx_last, :]).sum(), abs(M_stretched[idx_last, idx_last])))

print("== negative F/I/J are rejected, not silently wrong ==")
raised = False
try:
    hyperfine(-1, 1.5, 0.5, 100)
except ValueError:
    raised = True
check("hyperfine raises for negative F", raised)

raised = False
try:
    hyperfine_matrix(I=-1, J=0.5, Ahf=100)
except ValueError:
    raised = True
check("hyperfine_matrix raises for negative I (was silently returning a (0,0) matrix)", raised)

raised = False
try:
    hyperfine_matrix(I=1.5, J=-0.5, Ahf=100)
except ValueError:
    raised = True
check("hyperfine_matrix raises for negative J", raised)

raised = False
try:
    dipole_couple_matrix(-1, 0, (0.0, 1.0, 0.0))
except ValueError:
    raised = True
check("dipole_couple_matrix raises a clear ValueError for negative L1", raised)

print(f"\n{len(PASS)} checks passed.")
