"""General spin-J operators suite verification: `python tests/test_spin_j.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htdse.submodules.spin_j import spin_operators, raising_lowering, identity
from htdse.submodules.spin import sigma_x, sigma_y, sigma_z, sigma_plus, sigma_minus, I2

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

print("== spin_operators(0.5) matches spin.py's existing Pauli convention exactly ==")
Jx, Jy, Jz = spin_operators(0.5)
check("Jz == sigma_z/2", np.allclose(Jz, sigma_z / 2))
check("Jx == sigma_x/2", np.allclose(Jx, sigma_x / 2))
check("Jy == sigma_y/2", np.allclose(Jy, sigma_y / 2))
Jp, Jm = raising_lowering(0.5)
check("J+ == sigma_plus exactly (not rescaled)", np.allclose(Jp, sigma_plus))
check("J- == sigma_minus exactly", np.allclose(Jm, sigma_minus))
check("identity(0.5) == I2", np.allclose(identity(0.5), I2))

print("== angular momentum algebra holds for J = 0 .. 3 ==")
for J in (0, 0.5, 1, 1.5, 2, 2.5, 3):
    Jx, Jy, Jz = spin_operators(J)
    dim = Jz.shape[0]
    check(f"J={J}: dimension = 2J+1", dim == int(round(2 * J + 1)))
    check(f"J={J}: Jx,Jy,Jz all Hermitian",
          all(np.allclose(M, M.conj().T) for M in (Jx, Jy, Jz)))
    check(f"J={J}: [Jx,Jy] = i*Jz", np.allclose(Jx @ Jy - Jy @ Jx, 1j * Jz))
    check(f"J={J}: [Jy,Jz] = i*Jx", np.allclose(Jy @ Jz - Jz @ Jy, 1j * Jx))
    check(f"J={J}: [Jz,Jx] = i*Jy", np.allclose(Jz @ Jx - Jx @ Jz, 1j * Jy))
    J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
    check(f"J={J}: J^2 = J(J+1)*I", np.allclose(J2, J * (J + 1) * np.eye(dim)))
    Jp, Jm = raising_lowering(J)
    check(f"J={J}: J+ = (J-)^dagger", np.allclose(Jp, Jm.conj().T))
    check(f"J={J}: Jz eigenvalues descend from +J to -J",
          np.allclose(np.diag(Jz).real, [J - k for k in range(dim)]))

print("== malformed J raises ==")
for bad_J in (-1, 0.3, 1.7):
    raised = False
    try:
        spin_operators(bad_J)
    except ValueError:
        raised = True
    check(f"spin_operators({bad_J}) raises ValueError", raised)

print(f"\n{len(PASS)} checks passed.")
