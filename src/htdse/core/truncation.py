"""Truncation guard: warn when a state reaches the top of a truncated ladder.

A bosonic mode in this package lives in a Fock space cut off at `n_max`, where
`a^dagger |n_max>` is silently set to zero (see `submodules.harmonic_oscillator`).
That truncation is not an approximation the solver can detect: the evolution
stays perfectly norm-preserving and the ODE converges beautifully onto an
answer for the wrong Hamiltonian. Nothing looks wrong.

So the check has to be on the state, not the solver: if appreciable population
has reached |n_max>, the ceiling is participating in the dynamics and the
result is not the physics you asked for. This module measures that population
and warns once per subsystem per evolution.

Which subsystems get checked: by default, every registered factor of dimension
>= 3, on the reasoning that a dim-2 factor is a qubit (its top level is a
perfectly ordinary state, not a ceiling) while anything larger is probably a
truncated ladder. Name them explicitly with `ladders=(...)` when that guess
is wrong -- a genuine 3-level qudit, say, or a mode you know is safe.

Turn it off with `truncation=False` on one evolution, or globally:

    with htdse.no_truncation_check():
        ...
"""
import warnings

import numpy as np

from . import config

MIN_LADDER_DIM = 3   # dim-2 factors are qubits; their top level is not a ceiling


class TruncationWarning(UserWarning):
    """Population has reached the top level of a truncated subsystem.

    Promote to an error with
        warnings.simplefilter("error", htdse.TruncationWarning)
    which is the right setting inside a test suite or an overnight sweep.
    """


def resolve_threshold(setting):
    """Per-instance `truncation=` -> an active threshold, or None for off.

    None / True -> on, at config.TRUNCATION_THRESHOLD (so `no_truncation_check()`
                   and any global retuning apply, and apply at query time)
    False       -> off for this evolution
    float       -> that threshold, for this evolution only

    `True` is handled explicitly rather than falling through to `float()`:
    `float(True)` is 1.0, which as a threshold would mean "warn only once the
    ceiling holds the entire population" -- i.e. silently never firing, from an
    argument that plainly reads as "yes, check this".
    """
    if setting is None or setting is True:
        return config.TRUNCATION_THRESHOLD
    if setting is False:
        return None
    return float(setting)


def _basis_probs(state, kind):
    """Population over the full product basis, as a (..., d) real array."""
    a = np.asarray(state)
    if kind == "ket":
        return np.abs(a) ** 2
    if kind == "density":
        return np.real(np.diagonal(a, axis1=-2, axis2=-1))
    if kind == "unitary":
        # U[row, col]: column j is where basis state j has gone. Swap so the
        # last axis is the row (basis) index and the column joins the batch --
        # then a propagator is just a stack of kets and the rest is common.
        return np.swapaxes(np.abs(a) ** 2, -1, -2)
    raise ValueError(f"kind must be 'ket', 'density' or 'unitary', got {kind!r}")


def truncation_populations(state, subsystems, kind="ket", names=None) -> dict:
    """{subsystem: population in its top basis level}, maxed over any batch axes.

    `state` may carry leading batch axes (a time series from `state_at(ts)`, a
    stack of propagators); the returned number is the worst case over all of
    them, since one excursion into the ceiling invalidates everything after it.

    `subsystems` is the ordered {name: dim} registry -- ordering must match how
    the state was built, exactly as for `trace_out`. If it does not describe a
    state of this dimension the check declines to guess and returns {}.
    """
    labels, dims = list(subsystems.keys()), list(subsystems.values())
    if not dims:
        return {}
    probs = _basis_probs(state, kind)
    if probs.shape[-1] != int(np.prod(dims)):
        return {}   # registry doesn't describe this state; not our business
    n_lead = probs.ndim - 1
    p = probs.reshape(probs.shape[:-1] + tuple(dims))

    out = {}
    for k, (label, dim) in enumerate(zip(labels, dims)):
        if names is None:
            if dim < MIN_LADDER_DIM:
                continue
        elif label not in names:
            continue
        # integer-index the top level of factor k, then sum out every other
        # factor -- what remains is the marginal population at the ceiling
        idx = [slice(None)] * (n_lead + len(dims))
        idx[n_lead + k] = dim - 1
        sel = p[tuple(idx)]
        factor_axes = tuple(range(n_lead, sel.ndim))
        marginal = sel.sum(axis=factor_axes) if factor_axes else sel
        out[label] = float(np.max(marginal))
    return out


def warn_if_truncated(state, subsystems, kind, threshold, names, seen, label=""):
    """Warn once per subsystem (tracked in the mutable set `seen`).

    Called on every `state_at` query, but each subsystem can only fire once per
    evolution: the point is to tell you the run is compromised, not to bury the
    output under one warning per requested time point.
    """
    if threshold is None or not subsystems:
        return
    pending = {k: v for k, v in
               truncation_populations(state, subsystems, kind, names).items()
               if k not in seen}
    for name, pop in pending.items():
        if pop <= threshold:
            continue
        seen.add(name)
        dim = subsystems[name]
        where = f"{label}: " if label else ""
        warnings.warn(
            f"{where}subsystem {name!r} has {pop:.3g} population in its top "
            f"level |{dim - 1}> (threshold {threshold:g}). This space is "
            f"truncated -- a^dagger|{dim - 1}> is set to zero -- so the "
            f"ceiling is now part of the dynamics and these results are not "
            f"the physics you asked for. The norm is still 1 and the solver "
            f"still converged; neither means anything here. Raise n_max for "
            f"{name!r} until this stops firing, then confirm the answer has "
            f"stopped moving. If {name!r} is not a truncated ladder, exclude "
            f"it with ladders=(...) naming only the ones that are.",
            TruncationWarning, stacklevel=3)
