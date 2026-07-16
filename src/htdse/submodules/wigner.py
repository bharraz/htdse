"""Wigner function of a bosonic state given in the number (Fock) basis.

Convention: alpha = (x + i p)/sqrt(2), normalized so that
integral W(x,p) dx dp = 1 (vacuum peaks at 1/pi, |1> has W(0,0) = -1/pi).
Uses the closed-form Fock-basis expansion via associated Laguerre polynomials

    W(x,p) = (1/pi) e^{-2|a|^2} sum_{m>=n} (-1)^n sqrt(n!/m!) L_n^{m-n}(4|a|^2)
             * [ rho_nn  (m=n)  |  2 Re( rho_mn (2 a*)^{m-n} )  (m>n) ]

verified to machine precision against (2/pi) Tr[D(a)^dag rho D(a) Pi] (that
definition is d^2alpha-normalized, i.e. exactly 2x this one).

For a mode inside a larger register, reduce first:
    rho_mode = evolution.trace_out("q0", "q1", t=T)   # keep only the mode
    W = wigner(rho_mode, xs, ps)
"""
import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def wigner(state, xs, ps) -> np.ndarray:
    """W(x, p) on a grid, from a Fock-basis ket (d,) or density matrix (d, d).

    Returns array of shape (len(ps), len(xs)): W[i, j] = W(xs[j], ps[i]),
    ready for pcolormesh/contourf(xs, ps, W).
    """
    state = np.asarray(state, dtype=complex)
    rho = np.outer(state, state.conj()) if state.ndim == 1 else state
    d = rho.shape[0]

    X, P = np.meshgrid(np.asarray(xs, dtype=float), np.asarray(ps, dtype=float))
    alpha = (X + 1j * P) / np.sqrt(2)
    B = 4 * np.abs(alpha) ** 2
    pref = np.exp(-B / 2) / np.pi          # (1/pi) e^{-2|alpha|^2}

    # (2 alpha*)^k for every k that can appear, built once by repeated
    # multiplication -- recomputing the power on the whole grid inside the
    # element loop is the dominant cost otherwise.
    A = 2 * np.conj(alpha)
    Apow = [np.ones_like(A)]
    for _ in range(1, d):
        Apow.append(Apow[-1] * A)

    W = np.zeros_like(X)
    for n in range(d):
        if rho[n, n].real != 0:
            W += pref * (-1) ** n * rho[n, n].real * eval_genlaguerre(n, 0, B)
        for m in range(n + 1, d):
            if abs(rho[m, n]) < 1e-14:   # negligible coherence: skip the Laguerre
                continue
            k = m - n
            amp = np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)))  # sqrt(n!/m!), stable
            W += pref * 2 * (-1) ** n * amp \
                 * np.real(rho[m, n] * Apow[k]) \
                 * eval_genlaguerre(n, k, B)
    return W


def plot_wigner(state, extent=4.0, n_grid=201, ax=None):
    """Filled-contour Wigner plot on [-extent, extent]^2, diverging colormap
    symmetric about 0 so negativity (nonclassicality) is immediately visible."""
    import matplotlib.pyplot as plt
    xs = np.linspace(-extent, extent, n_grid)
    ps = np.linspace(-extent, extent, n_grid)
    W = wigner(state, xs, ps)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    lim = np.max(np.abs(W))
    im = ax.pcolormesh(xs, ps, W, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_aspect("equal")
    ax.figure.colorbar(im, ax=ax, label="W(x, p)")
    return ax