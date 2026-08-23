"""Rydberg (optical tweezer) interaction suite: `python tests/test_rydberg.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htdse.submodules.rydberg import (rydberg_interaction, dipole_dipole_interaction,
                                      blockade_radius)
from htdse import otimes, ket

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

print("== rydberg_interaction: two atoms, exact C6/r^6 shift on |00> only ==")
positions = np.array([[0.0, 0.0], [5.0, 0.0]])
shift = 1e6
C6 = 5.0 ** 6 * shift
M = np.asarray(rydberg_interaction(positions, C6, state="0").hamiltonian(0))
check("shape (4,4) for 2 qubits", M.shape == (4, 4))
check("Hermitian", np.allclose(M, M.conj().T))
diag = np.diag(M).real
check("only |00> carries the shift (state='0' is the Rydberg state)",
      np.allclose(diag, [shift, 0, 0, 0]))

M1 = np.asarray(rydberg_interaction(positions, C6, state="1").hamiltonian(0))
check("state='1' puts the shift on |11> instead",
      np.allclose(np.diag(M1).real, [0, 0, 0, shift]))

print("== rydberg_interaction: three atoms sums over all pairs correctly ==")
positions3 = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
C6_3 = 1000.0
M3 = np.asarray(rydberg_interaction(positions3, C6_3, state="0").hamiltonian(0))
expected_000 = C6_3 * (1 / 3 ** 6 + 1 / 3 ** 6 + 1 / 6 ** 6)  # pairs (0,1),(1,2),(0,2)
check("|000> energy is the sum of all 3 pairwise C6/r^6 terms",
      np.isclose(M3[0, 0].real, expected_000))

psi_one_excited = otimes(ket("0"), ket("1"), ket("1"))  # only atom 0 in the Rydberg state
idx_one = int(np.argmax(np.abs(psi_one_excited)))
check("exactly one atom in the Rydberg state carries zero interaction energy "
      "(needs >=2 atoms simultaneously excited)", np.isclose(M3[idx_one, idx_one].real, 0.0))

psi_two_excited = otimes(ket("0"), ket("0"), ket("1"))  # atoms 0,1 excited, distance 3
idx_two = int(np.argmax(np.abs(psi_two_excited)))
check("exactly two (adjacent) atoms excited carries just that one pair's C6/r^6",
      np.isclose(M3[idx_two, idx_two].real, C6_3 / 3 ** 6))

print("== dipole_dipole_interaction: flip-flop only couples |01><->|10> ==")
C3 = 100.0
Mdd = np.asarray(dipole_dipole_interaction(positions, C3).hamiltonian(0))
check("Hermitian", np.allclose(Mdd, Mdd.conj().T))
expected_coupling = C3 / 5.0 ** 3
check("coupling strength matches C3/r^3",
      np.isclose(Mdd[1, 2].real, expected_coupling) and np.isclose(Mdd[2, 1].real, expected_coupling))
check("no other matrix elements are populated",
      np.isclose(np.abs(Mdd).sum(), 2 * abs(expected_coupling)))

print("== blockade_radius: standard (C6/Omega)^(1/6) estimate ==")
check("blockade_radius(5^6 * 1e6, 1e6) == 5.0", np.isclose(blockade_radius(C6, shift), 5.0))
check("blockade_radius scales correctly with Omega",
      np.isclose(blockade_radius(C6, shift * 64), 5.0 / 2))  # 64x Omega -> r_b/2 (64=2^6)

print("== error handling ==")
raised = False
try:
    rydberg_interaction(np.array([[0.0, 0.0], [0.0, 0.0]]), 1.0)
except ValueError:
    raised = True
check("rydberg_interaction raises for coincident atom positions", raised)

raised = False
try:
    rydberg_interaction(positions, 1.0, state="2")
except ValueError:
    raised = True
check("rydberg_interaction raises for an invalid state label", raised)

raised = False
try:
    rydberg_interaction(positions, 1.0, spins=["a", "b", "c"])
except ValueError:
    raised = True
check("rydberg_interaction raises when spins length mismatches positions", raised)

print(f"\n{len(PASS)} checks passed.")
