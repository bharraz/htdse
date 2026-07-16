"""Magnus expansion of a time-dependent H(t), read out in the Pauli basis.

A utility, not a solver: use it to SEE what effective Hamiltonian a pulse
actually generates -- which Pauli strings it turns on, how big each one is, and
which ones a schedule is supposed to cancel. For the actual dynamics, use the
evolution classes.

    U(T) = exp(Omega_1 + Omega_2 + ...)

    Omega_1 = -i int_0^T H(t) dt
    Omega_2 = -1/2 int_0^T dt1 int_0^t1 dt2 [H(t1), H(t2)]

Only orders 1 and 2 are implemented. That is not a shortcut: it is exactly the
regime the expansion is used in here -- for a spin-dependent force the series
TERMINATES at second order (all the spin operators commute, so [H(t1), H(t2)] is
a scalar spin operator that commutes with everything). Asking for order 3 raises
rather than returning a truncation you might mistake for exact.

    >>> terms = magnus_pauli(H, T, order=2)
    >>> terms[1]["ZZ"]        # the ZZ coupling generated at second order

Both functions accept a Mechanism (anything with `.hamiltonian(t)`) or a plain
callable t -> matrix.
"""
import itertools

import numpy as np

from .submodules.spin import I2, sigma_x, sigma_y, sigma_z

_PAULI_1Q = {"I": I2, "X": sigma_x, "Y": sigma_y, "Z": sigma_z}


def _H_of(H):
    """Accept a Mechanism or a bare callable t -> matrix."""
    fn = getattr(H, "hamiltonian", None)
    return fn if callable(fn) else H


def _cumtrapz(y, x):
    """Cumulative trapezoid along the FIRST axis (a stack of matrices)."""
    dx = np.diff(x).reshape(-1, *([1] * (y.ndim - 1)))
    seg = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y)
    out[1:] = np.cumsum(seg, axis=0)
    return out


def magnus(H, T, order=2, t0=0.0, n_grid=2001) -> list:
    """The Magnus terms [Omega_1, ...] of H(t) on [t0, T], as dense matrices.

    Quadrature is trapezoid on a uniform grid of `n_grid` points -- raise it if
    H oscillates fast on the scale of (T - t0)/n_grid.

    Omega_2 uses [H(t1), int_0^t1 H(t2) dt2], which is the same double integral
    with the inner one accumulated once (O(n_grid) matrix products, not O(n^2)).
    """
    if order not in (1, 2):
        raise ValueError(
            f"magnus: order must be 1 or 2, got {order}. Higher orders are not "
            "implemented -- returning a truncated Omega_3 that looked exact "
            "would be worse than refusing.")
    h = _H_of(H)
    ts = np.linspace(float(t0), float(T), int(n_grid))
    Hs = np.array([np.asarray(h(t), dtype=complex) for t in ts])   # (n, d, d)

    Om1 = -1j * np.trapezoid(Hs, ts, axis=0)
    if order == 1:
        return [Om1]

    G = _cumtrapz(Hs, ts)                       # G(t1) = int_0^t1 H dt2
    comm = Hs @ G - G @ Hs                      # [H(t1), G(t1)], batched
    Om2 = -0.5 * np.trapezoid(comm, ts, axis=0)
    return [Om1, Om2]


def pauli_decompose(M, tol=1e-12) -> dict:
    """Expand a 2^n x 2^n matrix in the Pauli basis: {"XZ": coeff, ...}.

    c_P = Tr(P M) / 2^n, since the Paulis are Hermitian and Tr(P P') = 2^n d_PP'.
    Strings are ordered qubit 0 first. Coefficients below `tol` are dropped.
    A Hermitian M gives real coefficients (up to rounding).
    """
    M = np.asarray(M, dtype=complex)
    d = M.shape[0]
    n = int(round(np.log2(d)))
    if 2 ** n != d or M.shape != (d, d):
        raise ValueError(f"pauli_decompose needs a square 2^n matrix, got {M.shape}")
    out = {}
    for letters in itertools.product("IXYZ", repeat=n):
        P = _PAULI_1Q[letters[0]]
        for L in letters[1:]:
            P = np.kron(P, _PAULI_1Q[L])
        c = np.trace(P @ M) / d
        if abs(c) > tol:
            out["".join(letters)] = complex(c)
    return out


def magnus_pauli(H, T, order=2, t0=0.0, n_grid=2001, tol=1e-9) -> list:
    """`magnus(...)` with every term decomposed in the Pauli basis.

    Returns one {pauli_string: coeff} dict per Magnus order. The generator is
    anti-Hermitian (Omega = -i H_eff), so the coefficients here are those of
    Omega itself -- multiply by 1j to read them as an effective Hamiltonian.

    Only works on a pure qubit register (dim 2^n): a mechanism carrying a
    motional mode has to be traced/projected down to the spins first.
    """
    return [pauli_decompose(Om, tol=tol)
            for Om in magnus(H, T, order=order, t0=t0, n_grid=n_grid)]
