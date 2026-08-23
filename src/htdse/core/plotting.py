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

    Dimension/system-agnostic. `labels`: optional per-basis-state labels;
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


def plot_phases(ts, states, labels=None, ax=None):
    """Phase arg(<i|psi(t)>) of each computational basis amplitude vs t.

    `states` may be:
      - an evolution object (anything with `.state_at`) -- sampled at `ts`;
      - a ket trajectory, shape (n_times, dim).

    Density matrices have no well-defined per-basis-state phase (a global
    phase on |psi> is unobservable, but rho's diagonal fixes populations
    only) -- pass a ket trajectory or an evolution whose `.state_at` returns
    kets. Phase is masked (not drawn) wherever the amplitude is numerically
    zero, since arg(0) is meaningless noise rather than a real value.

    Dimension/system-agnostic. `labels`: as in `plot_populations`.
    """
    if hasattr(states, "state_at"):
        states = states.state_at(ts)
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(f"plot_phases needs a ket trajectory (n_times, dim), got shape {states.shape}")
    n_times, dim = states.shape
    if labels is None:
        n_bits = round(np.log2(dim))
        if 2 ** n_bits == dim:
            labels = [index_to_binary(i, n_bits) for i in range(dim)]
        else:
            labels = [str(i) for i in range(dim)]

    thresh = 1e-10 * (np.max(np.abs(states)) or 1.0)
    phases = np.where(np.abs(states) > thresh, np.angle(states), np.nan)

    if ax is None:
        _, ax = plt.subplots()
    for i in range(dim):
        ax.plot(ts, phases[:, i], label=f"|{labels[i]}>")
    ax.set_xlabel("t")
    ax.set_ylabel("phase (rad)")
    ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax.legend()
    return ax


def plot_matrix(M, t=0.0, kind="abs", ax=None):
    """Heatmap of an operator -- the "what does this actually look like"
    sanity check for a Hamiltonian or a gate, before or instead of solving
    anything.

    M: a System/Model (`.hamiltonian(t)` is called) or a plain array.
    kind: "abs" (|M_ij|, default), "real"/"imag" (signed, diverging colormap
    centered at 0), or "phase" (arg(M_ij), shown only where |M_ij| is
    non-negligible -- phase of a numerically-zero entry is meaningless noise).
    """
    Mv = M.hamiltonian(t) if hasattr(M, "hamiltonian") else M
    Mv = np.asarray(Mv, dtype=complex)
    if kind == "abs":
        data, cmap, vmin, vmax, label = np.abs(Mv), "viridis", 0, None, "|M_ij|"
    elif kind == "real":
        lim = np.max(np.abs(Mv.real)) or 1.0
        data, cmap, vmin, vmax, label = Mv.real, "RdBu_r", -lim, lim, "Re(M_ij)"
    elif kind == "imag":
        lim = np.max(np.abs(Mv.imag)) or 1.0
        data, cmap, vmin, vmax, label = Mv.imag, "RdBu_r", -lim, lim, "Im(M_ij)"
    elif kind == "phase":
        mag = np.abs(Mv)
        thresh = 1e-10 * (mag.max() or 1.0)
        data = np.where(mag > thresh, np.angle(Mv), np.nan)
        cmap, vmin, vmax, label = "twilight", -np.pi, np.pi, "arg(M_ij)"
    else:
        raise ValueError(f"kind must be 'abs'/'real'/'imag'/'phase', got {kind!r}")

    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("column"); ax.set_ylabel("row")
    ax.set_title(f"{label}, t={t}")
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


def plot_adiabatic_populations(evolution, ts, ax=None):
    """Population in each instantaneous eigenstate of H(t) vs t:
    |<n(t)|psi(t)>|^2, one line per level, ordered by ascending E_n(t).

    `evolution`: a HamiltonianEvolution (uses its `adiabatic_populations`).
    Same caveat as `plot_eigenspectrum`: level identity can jump at a
    (near-)degeneracy crossing.
    """
    ts = np.asarray(ts)
    pops = evolution.adiabatic_populations(ts)

    if ax is None:
        _, ax = plt.subplots()
    for n in range(pops.shape[1]):
        ax.plot(ts, pops[:, n], label=f"level {n}")
    ax.set_xlabel("t")
    ax.set_ylabel("adiabatic population")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    return ax
