"""Neutral-atom (optical tweezer) Rydberg interactions: pairwise van der
Waals and resonant dipole-dipole coupling between atoms' Rydberg-state
occupations, plus the standard blockade-radius estimate.

Reuses the existing 2-level qubit machinery (`spin.py`'s ket/PAULIS
convention) rather than inventing a new basis -- "the Rydberg state" is
just whichever computational basis state (`"0"` or `"1"`) you name; nothing
here knows or cares about the actual atomic structure that got you there.
Multi-level Rydberg manifolds (e.g. a separate intermediate state for
two-photon excitation) are out of scope -- build that Hamiltonian with
`term()` directly and add this module's interaction term on top of it.
"""
import itertools

import numpy as np

from ..core.terms import term, plus_hc
from .spin import sigma_plus, sigma_minus


def _projector(state):
    if state not in ("0", "1"):
        raise ValueError(f"rydberg: state must be '0' or '1', got {state!r}")
    proj = np.zeros((2, 2), dtype=complex)
    proj[0 if state == "0" else 1, 0 if state == "0" else 1] = 1.0
    return proj


def _pair_spins(positions, spins, prefix):
    positions = np.asarray(positions, dtype=float)
    n_atoms = positions.shape[0]
    if spins is None:
        spins = [f"{prefix}{i}" for i in range(n_atoms)]
    elif len(spins) != n_atoms:
        raise ValueError("spins must have exactly one name per row of positions")
    return positions, spins


def rydberg_interaction(positions, C6, spins=None, prefix="q", state="0") -> "Model":
    """Pairwise van der Waals interaction between Rydberg-state occupations:

        H = sum_{i<j} (C6 / |r_i - r_j|^6) * n_i * n_j

    n_i = |state><state| on atom i -- `state` names which computational
    basis state ("0" or "1") represents the Rydberg level. `positions` is
    an (N, d) array of atom coordinates, in whatever length unit C6 is
    already expressed in (i.e. C6 carries units of energy*length^6 for that
    unit -- there is no implicit unit conversion here)."""
    positions, spins = _pair_spins(positions, spins, prefix)
    proj = _projector(state)
    H = None
    for i, j in itertools.combinations(range(len(spins)), 2):
        r = float(((positions[i] - positions[j]) ** 2).sum() ** 0.5)
        if r == 0:
            raise ValueError(f"rydberg_interaction: atoms {i} and {j} share the same position")
        t = term({spins[i]: proj, spins[j]: proj}, coeff=C6 / r ** 6, name=f"vdw_{i}_{j}")
        H = t if H is None else H + t
    return H


def dipole_dipole_interaction(positions, C3, spins=None, prefix="q") -> "Model":
    """Resonant dipole-dipole (Foerster) exchange between two atoms' Rydberg
    states:

        H = sum_{i<j} (C3 / |r_i - r_j|^3) * (sigma_plus_i sigma_minus_j + h.c.)

    A flip-flop term exchanging excitation between atoms -- the resonant
    counterpart to the off-resonant van der Waals shift in
    `rydberg_interaction`. Same position/units convention as that function."""
    positions, spins = _pair_spins(positions, spins, prefix)
    H = None
    for i, j in itertools.combinations(range(len(spins)), 2):
        r = float(((positions[i] - positions[j]) ** 2).sum() ** 0.5)
        if r == 0:
            raise ValueError(f"dipole_dipole_interaction: atoms {i} and {j} share the same position")
        t = plus_hc(term({spins[i]: sigma_plus, spins[j]: sigma_minus},
                         coeff=C3 / r ** 3, name=f"dd_{i}_{j}"))
        H = t if H is None else H + t
    return H


def blockade_radius(C6, Omega) -> float:
    """r_b = (|C6| / Omega)^(1/6): the atom separation at which the van der
    Waals shift equals the drive Rabi frequency Omega -- the standard
    estimate for where double Rydberg excitation becomes off-resonant (the
    blockade mechanism behind neutral-atom two-qubit gates)."""
    return (abs(C6) / Omega) ** (1 / 6)
