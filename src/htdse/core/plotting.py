import numpy as np
import matplotlib.pyplot as plt

from ..util import index_to_binary


def plot_populations(ts, states, labels=None, ax=None):
    """Population of each computational basis state vs t.

    `states` may be:
      - an evolution object (anything with `.state_at`) -- sampled at `ts`;
      - a ket trajectory, shape (n_times, dim): populations |<i|psi(t)>|^2;
      - a density-matrix trajectory, shape (n_times, dim, dim): diagonal
        Re rho_ii(t) (e.g. from trace_out or a LindbladEvolution).

    Dimension/mechanism-agnostic. `labels`: optional per-basis-state labels;
    default to bitstrings if dim is a power of 2 (qubit register), otherwise
    plain numeric indices (e.g. Fock states of a truncated oscillator).
    """
    if hasattr(states, "state_at"):
        states = states.state_at(ts)
    states = np.asarray(states)
    if states.ndim == 3:                      # density-matrix trajectory
        pops = np.real(np.einsum("nii->ni", states))
    else:                                     # ket trajectory
        pops = np.abs(states) ** 2
    n_times, dim = pops.shape
    if labels is None:
        n_bits = round(np.log2(dim))
        if 2 ** n_bits == dim:
            labels = [index_to_binary(i, n_bits) for i in range(dim)]
        else:
            labels = [str(i) for i in range(dim)]

    if ax is None:
        _, ax = plt.subplots()
    for i in range(dim):
        ax.plot(ts, pops[:, i], label=f"|{labels[i]}>")
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    return ax


def plot_eigenspectrum(evolution, ts, ax=None):
    """Instantaneous eigenvalues of H(t), one line per level, vs t.

    `evolution`: a HamiltonianEvolution (uses its instantaneous_eigenbasis).
    Levels are sorted ascending by eigh convention -- level 0 is the
    instantaneous ground state at every t (levels can swap identity at
    crossings; see the degeneracy caveat on instantaneous_eigenbasis).
    """
    ts = np.asarray(ts)
    spectra = np.array([evolution.instantaneous_eigenbasis(t)[0] for t in ts])  # eigenvalues per t

    if ax is None:
        _, ax = plt.subplots()
    for n in range(spectra.shape[1]):
        ax.plot(ts, spectra[:, n], label=f"level {n}")
    ax.set_xlabel("t")
    ax.set_ylabel("instantaneous eigenvalue")
    ax.legend()
    return ax
