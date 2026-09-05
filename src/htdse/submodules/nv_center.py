"""Nitrogen-vacancy (NV) center ground-state (3A2, spin-1) physics, built on
`spin_j.py`'s general angular-momentum operators.

Ground-state-manifold physics only: no optical/orbital levels (3E excited
state, the 1A1/1E metastable singlets that mediate intersystem crossing) are
modeled as actual levels -- `isc_dephasing` below is an explicit, named
simplification of the readout mechanism those levels are normally
responsible for, not a claim that they've been simulated.

    H = D (Sz^2 - S(S+1)/3) + E (Sx^2 - Sy^2) + g_e*Bz*Sz
        + sum_nuclei [ A_par Sz Iz + A_perp (Sx Ix + Sy Iy) ]

D ~ 2.87 GHz is the zero-field splitting; E (strain/electric-field
dependent, usually << D) splits ms=+-1 at zero field; the hyperfine term is
the standard axially-symmetric (uncoupled-basis) dipolar coupling to a
nearby nuclear spin (14N, 15N, 13C, ...) -- deliberately NOT the coupled
|F,mF> formalism `atomic.py` builds, since a solid-state defect's tensor
axes are fixed to the crystal, not built from a good total-F quantum number.
"""
import numpy as np

from ..core.terms import System

from ..core.terms import term
from .spin_j import spin_operators


def zero_field_splitting(D, E=0.0, spin="e") -> "System":
    """D(Sz^2 - S(S+1)/3) + E(Sx^2-Sy^2) for the S=1 electronic ground
    state named `spin`. The -S(S+1)/3 piece only shifts the overall energy
    zero (it's proportional to identity) but is kept so `D` matches the
    literature convention directly."""
    Sx, Sy, Sz = spin_operators(1.0)
    S = 1.0
    H = term(Sz @ Sz - (S * (S + 1) / 3) * np.eye(3), on=spin, coeff=D, name="zfs_D")
    if E != 0:
        H = H + term(Sx @ Sx - Sy @ Sy, on=spin, coeff=E, name="zfs_E")
    return H


def zeeman(g, Bz, spin="e", J=1.0) -> "System":
    """g*Bz*Jz -- axial Zeeman shift for a spin-J subsystem (electron: J=1,
    g~2.003; a nuclear spin: J=I, g=g_I). Assumes the field is along the
    quantization (NV) axis -- off-axis field mixing is not modeled."""
    _, _, Jz = spin_operators(J)
    return term(Jz, on=spin, coeff=g * Bz, name=f"zeeman_{spin}")


def hyperfine_tensor(A_parallel, A_perp, electron="e", nuclear="n", I=1.0) -> "System":
    """Axially-symmetric dipolar hyperfine coupling in the UNCOUPLED
    electron(x)nuclear product basis:

        A_parallel Sz Iz + A_perp (Sx Ix + Sy Iy)

    electron is always spin-1 (the NV ground state); `nuclear` is the
    nuclear spin I (14N: I=1, 15N: I=1/2, 13C: I=1/2)."""
    Sx, Sy, Sz = spin_operators(1.0)
    Ix, Iy, Iz = spin_operators(I)
    H = term({electron: Sz, nuclear: Iz}, coeff=A_parallel, name="hf_par")
    H = H + term({electron: Sx, nuclear: Ix}, coeff=A_perp, name="hf_perp_x")
    H = H + term({electron: Sy, nuclear: Iy}, coeff=A_perp, name="hf_perp_y")
    return H


def isc_dephasing(rates, spin="e") -> "System":
    """A simplified proxy for NV's spin-dependent optical readout contrast:
    one pure-dephasing-style Lindblad jump per m_s sublevel (m_s=+1,0,-1,
    matching spin_operators(1.0)'s index order), each built from that
    sublevel's own projector with its own rate.

    This is NOT the full singlet-shelving level structure (the metastable
    1A1/1E states aren't modeled as levels) -- it's the minimal jump set
    that reproduces different m_s sublevels losing coherence/population at
    different rates under illumination, which is the actual mechanism
    behind NV's spin-to-fluorescence contrast, without carrying the extra
    orbital/singlet levels through the whole simulation. `rates` is
    (rate_ms=+1, rate_ms=0, rate_ms=-1); a real NV has ms=+-1 shelving much
    faster than ms=0 (that asymmetry is what makes optical readout work)."""
    from ..core.terms import jump
    if len(rates) != 3:
        raise ValueError("isc_dephasing: rates must have exactly 3 entries (ms=+1,0,-1)")
    L = None
    for k, rate in enumerate(rates):
        if rate == 0:
            continue
        proj = np.zeros((3, 3), dtype=complex)
        proj[k, k] = 1.0
        j = jump(proj, on=spin, coeff=np.sqrt(rate), name=f"isc_{k}")
        L = j if L is None else L + j
    return L


class NVCenter:
    """The NV ground-state manifold: electron spin (always S=1, subsystem
    name `e`) plus zero or more nuclear spins.

        nv = NVCenter(D=2.87e9, nuclear_spins={"N14": 1.0})
        H = nv.hamiltonian(Bz=1e5, hyperfine={"N14": (-2.14e6, -2.7e6)})
    """

    def __init__(self, D, E=0.0, nuclear_spins=None, prefix_e="e"):
        self.D, self.E = D, E
        self.nuclear_spins = dict(nuclear_spins or {})
        self.e = prefix_e
        self.subsystems = {self.e: 3,
                           **{name: int(round(2 * I + 1)) for name, I in self.nuclear_spins.items()}}

    @property
    def dim(self) -> int:
        d = 1
        for v in self.subsystems.values():
            d *= v
        return d

    def hamiltonian(self, Bz=0.0, g_e=2.0028, hyperfine=None) -> "System":
        """hyperfine: optional {nuclear_name: (A_parallel, A_perp)} for any
        subset of self.nuclear_spins."""
        H = zero_field_splitting(self.D, self.E, spin=self.e)
        if Bz != 0:
            H = H + zeeman(g_e, Bz, spin=self.e, J=1.0)
        for name, (A_par, A_perp) in (hyperfine or {}).items():
            if name not in self.nuclear_spins:
                raise KeyError(f"NVCenter has no nuclear spin named {name!r}; "
                               f"known: {list(self.nuclear_spins)}")
            H = H + hyperfine_tensor(A_par, A_perp, electron=self.e, nuclear=name,
                                     I=self.nuclear_spins[name])
        return H

    def __repr__(self):
        return f"NVCenter(D={self.D}, E={self.E}, nuclear_spins={self.nuclear_spins})"
