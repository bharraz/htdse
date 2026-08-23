"""Trapped-ion specialization of `spin_boson.py`: a chain of `n_ions` spins
coupled to a table of shared motional `Mode`s.

`IonChain` owns exactly what the general layer doesn't know about an ion:
how many spins there are, which subsystem names they get, and the per-mode
Fock truncation -- so `subsystems` has one source instead of being inferred
from whatever arrays happen to be the right length (see the deleted
`_infer_n_ions` in the old `molmer_sorensen.py`).

Computing a real chain's normal modes (nu_m, b_{i,m}) from trap parameters --
equilibrium positions from the Coulomb + harmonic potential, then Hessian
eigenvectors (James 1998) -- is NOT implemented here. `IonChain.from_trap(...)`
is the documented future home for it.
"""
import numpy as np

from .harmonic_oscillator import thermal
from .spin_boson import Mode, driven_spins
from . import trap as _trap


class IonChain:
    """`n_ions` two-level ions sharing the motional `modes`.

        chain = IonChain(modes=[Mode(nu=nu, eta=eta*b, n_max=8)], n_ions=2)
        H = chain.drive(tones, lamb_dicke=1)

    Spins are named "q0", "q1", ... -- change `prefix=` to rename them.
    """

    def __init__(self, modes, n_ions, prefix="q"):
        self.modes = list(modes)
        self.n_ions = int(n_ions)
        self.prefix = prefix
        self.spins = [f"{prefix}{i}" for i in range(self.n_ions)]
        self.subsystems = {**{q: 2 for q in self.spins},
                           **{md.name: md.n_max + 1 for md in self.modes}}

    @property
    def dim(self) -> int:
        d = 1
        for v in self.subsystems.values():
            d *= v
        return d

    def drive(self, tones, lamb_dicke=1, rwa=False):
        """The Hamiltonian of this chain driven by `tones` -- a thin
        pass-through to `spin_boson.driven_spins` so ion-chain code never
        assembles that call by hand."""
        return driven_spins(tones, self.spins, self.modes, lamb_dicke=lamb_dicke, rwa=rwa)

    def thermal(self, nbar):
        """Motional density matrix: each mode independently thermal at `nbar`
        (scalar, or one value per mode), tensored together in mode order.
        `nbar` is an INITIAL-STATE choice, orthogonal to which rung of the
        approximation ladder `drive()` uses.

        Cost note: this returns a density matrix (dim^2 for a mixed motional
        state), so pair it with `DensityMatrixEvolution`/`LindbladEvolution`.
        For large chains, Monte-Carlo averaging over Fock states sampled from
        the thermal distribution is the cheaper alternative and stays in
        state-vector land -- not implemented here."""
        nbars = np.broadcast_to(np.asarray(nbar, dtype=float), (len(self.modes),))
        rho = thermal(float(nbars[0]), self.modes[0].n_max)
        for nb, md in zip(nbars[1:], self.modes[1:]):
            rho = np.kron(rho, thermal(float(nb), md.n_max))
        return rho

    def _mode(self, mode_name):
        for md in self.modes:
            if md.name == mode_name:
                return md
        raise KeyError(f"IonChain has no mode named {mode_name!r}; "
                       f"known modes: {[md.name for md in self.modes]}")

    def sideband_coupling(self, mode_name, n1, n2, phase=True):
        """<n1|exp(i eta (a+a^dagger))|n2> for the named mode's own eta --
        a thin pass-through to `trap.sideband` so callers don't have to pull
        `mode.eta` out by hand."""
        return _trap.sideband(n1, n2, self._mode(mode_name).eta, phase=phase)

    def thermal_sideband(self, mode_name, delta_n, t, nbar, thresh=1e-3):
        """Thermally-averaged sideband Rabi flopping probability on the
        named mode's delta_n-th sideband -- a thin pass-through to
        `trap.thermal_sideband` using the mode's own eta."""
        return _trap.thermal_sideband(nbar, delta_n, self._mode(mode_name).eta, t, thresh=thresh)

    def __repr__(self):
        modes = ", ".join(f"{md.name}(n_max={md.n_max})" for md in self.modes)
        return f"IonChain(n_ions={self.n_ions}, modes=[{modes}])"
