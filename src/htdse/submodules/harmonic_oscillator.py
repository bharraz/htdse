"""Bosonic mode (harmonic oscillator) operators, Fock-space truncated at
dimension n_max+1, plus a thermal-bath Lindblad mechanism for a damped and
dephased motional mode (e.g. trapped-ion motional heating/dephasing).
"""
import numpy as np

from ..core.mechanism import Mechanism
from ..core.operator import Operator


def annihilation(n_max: int) -> np.ndarray:
    """a, truncated to Fock states |0>...|n_max>: a|n> = sqrt(n)|n-1>.

    Note: a^dagger|n_max> is artificially truncated to 0 (no room above
    n_max) -- choose n_max well above any population expected to reach it.
    """
    # a[n-1, n] = sqrt(n): the first superdiagonal. Built in one shot -- this is
    # called from inner loops (per ion, per mode, per plot), so no Python loop.
    return np.diag(np.sqrt(np.arange(1, n_max + 1, dtype=float)), 1).astype(complex)


def creation(n_max: int) -> np.ndarray:
    """a^dagger in the truncated space (= annihilation(n_max).conj().T)."""
    return annihilation(n_max).conj().T


def number_operator(n_max: int) -> np.ndarray:
    a = annihilation(n_max)
    return a.conj().T @ a  # a^dagger a


def fock(n: int, n_max: int) -> np.ndarray:
    """Fock state |n> in the truncated (n_max+1)-dim space."""
    psi = np.zeros(n_max + 1, dtype=complex)
    psi[n] = 1.0
    return psi


class ThermalMotionalDecoherence(Mechanism):
    """Damped + dephased quantum harmonic oscillator coupled to a thermal
    reservoir (e.g. trapped-ion motional-mode heating and dephasing):

        d(rho)/dt = gamma_a*nbar     * D[a^dagger] rho    (heating)
                  + gamma_a*(nbar+1) * D[a] rho            (decay/cooling)
                  + gamma_p          * D[a^dagger a] rho   (pure dephasing)

    with D[L]rho = L rho L^dagger - 1/2{L^dagger L, rho}. No coherent
    Hamiltonian term (H=0) -- this models pure decoherence, as in a Ramsey
    interrogation with no dressing.

    gamma_a: single-quantum coupling rate (initial heating rate from the
             ground state is ndot = gamma_a*nbar).
    nbar: thermal occupation, nbar = 1/(exp(hbar*omega/kT) - 1).
    gamma_p: pure dephasing rate.

    Exact |0>,|1> coherence decay rate at leading order (before heating
    re-feeds coherence from higher Fock pairs):

        1/T2 = 2*ndot + gamma_a/2 + gamma_p/2

    The often-quoted T2 ~ 1/(2*ndot + gamma_p/2) drops the gamma_a/2
    spontaneous-decay piece and is only valid for nbar >> 1.
    """

    def __init__(self, n_max, gamma_a=0.0, nbar=0.0, gamma_p=0.0):
        self.n_max = n_max
        self.gamma_a, self.nbar, self.gamma_p = gamma_a, nbar, gamma_p
        self.dim = n_max + 1
        self._a = annihilation(n_max)
        self._n = self._a.conj().T @ self._a  # number operator
        self._jumps = None      # time-independent; built on demand, see below
        self._jumps_key = None  # the rates _jumps was built from
        self.subsystems = {"mode": self.dim}

    def hamiltonian(self, t) -> Operator:
        return Operator(np.zeros((self.dim, self.dim), dtype=complex))  # no coherent term

    def jump_operators(self, t) -> list:
        # Time-independent, so build once rather than per rhs eval -- but key the
        # cache to the rates it was built FROM. A bare "build once" cache would
        # survive `m.gamma_a = ...` and hand the old rates to a freshly bound
        # evolution, whose mutation guard (snapshot taken at binding) sees
        # nothing wrong.
        key = (self.gamma_a, self.nbar, self.gamma_p)
        if self._jumps is None or self._jumps_key != key:
            gamma_a, nbar, gamma_p = key
            ops = []
            if gamma_a > 0 and nbar > 0:
                ops.append(Operator(np.sqrt(gamma_a * nbar) * self._a.conj().T))  # heating
            if gamma_a > 0:
                ops.append(Operator(np.sqrt(gamma_a * (nbar + 1)) * self._a))  # cooling
            if gamma_p > 0:
                ops.append(Operator(np.sqrt(gamma_p) * self._n))  # pure dephasing
            self._jumps, self._jumps_key = ops, key
        return self._jumps

    def __repr__(self):
        return (f"ThermalMotionalDecoherence(n_max={self.n_max}, gamma_a={self.gamma_a}, "
                f"nbar={self.nbar}, gamma_p={self.gamma_p})")
