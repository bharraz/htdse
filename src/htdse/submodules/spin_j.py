"""General angular-momentum (spin-J) operators for arbitrary J = 0, 1/2, 1,
3/2, 2, ... -- what `spin.py`'s hardcoded 2-level sigma_x/y/z don't cover.
Needed for anything with a higher-spin ground manifold, e.g. the NV center's
spin-1 ground state (see `nv_center.py`).

Basis convention: index k (k = 0..2J) is m = J-k -- descending m, so index 0
is the top of the ladder. This exactly matches `spin.py`'s convention: for
J=1/2, `spin_operators(0.5)` gives `Jz = sigma_z/2` and `J+ = sigma_plus`
identically (the standard S = sigma/2 relation, not a rescaled copy of it).
"""
import numpy as np


def spin_operators(J):
    """(Jx, Jy, Jz) for angular momentum J, dimension 2J+1. Jz = diag(J,
    J-1, ..., -J); J+ raises m (superdiagonal in this basis); Jx=(J++J-)/2,
    Jy=(J+-J-)/(2i). Standard matrix elements:
    J+|j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1>."""
    if J < 0 or not np.isclose(2 * J, round(2 * J)):
        raise ValueError(f"spin_operators: J={J} must be a non-negative (half-)integer")
    dim = int(round(2 * J + 1))
    ms = np.array([J - k for k in range(dim)])
    Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        m = ms[k]
        Jp[k - 1, k] = np.sqrt(J * (J + 1) - m * (m + 1))
    Jm = Jp.conj().T
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2j)
    return Jx, Jy, Jz


def raising_lowering(J):
    """(J+, J-) for angular momentum J -- see spin_operators for the basis
    convention. J+ = Jx + i*Jy raises m by 1; J- lowers it."""
    Jx, Jy, _ = spin_operators(J)
    Jp = Jx + 1j * Jy
    return Jp, Jp.conj().T


def identity(J) -> np.ndarray:
    """Identity on the 2J+1-dimensional spin-J space."""
    dim = int(round(2 * J + 1))
    return np.eye(dim, dtype=complex)
