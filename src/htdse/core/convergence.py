"""Answer the question a truncation/tolerance warning raises: what IS enough?

`TruncationWarning` tells you `n_max` is too small. The next thing anyone does
is re-run by hand at a few larger values and eyeball whether the answer moved.
`converged()` is that loop, done once and reported honestly.

    from htdse import converged

    def gate_error(n_max):
        H = build_model(n_max)
        with htdse.quiet():
            psi = htdse.HamiltonianEvolution(H, initial(n_max)).state_at(T)
        return 1 - htdse.fidelity(target(n_max), psi)     # a SCALAR

    result = converged(gate_error, [4, 6, 8, 10, 12], tol=1e-8)
    print(result)          # -> converged at n_max = 8

THE ONE RULE: whatever `fn` returns must be COMPARABLE ACROSS THE SWEEP. For an
`n_max` sweep this is not a formality -- the state vectors at n_max=4 and
n_max=8 live in different-dimensional Hilbert spaces and cannot be subtracted at
all. Return a scalar observable, a fidelity against a fixed target, or a reduced
density matrix on subsystems whose dimension is not being swept. The default
metric raises (rather than broadcasting, or silently comparing the wrong thing)
when shapes disagree.
"""
import numpy as np

from . import config


def _default_metric(a, b):
    """Max absolute change. Refuses mismatched shapes rather than guessing."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"converged: consecutive results have shapes {a.shape} and {b.shape}. "
            "They are not comparable, so no convergence claim is possible. This is "
            "the usual sign of returning a raw state from an n_max sweep -- those "
            "live in different-dimensional spaces. Return a scalar observable, a "
            "fidelity against a fixed target, or a reduced density matrix over "
            "subsystems you are not sweeping.")
    return float(np.max(np.abs(a - b)))


class Convergence(dict):
    """Result of a `converged()` sweep; prints as a small table."""

    def __str__(self):
        head = (f"converged at {self['parameter']} = {self['value']!r} "
                f"(change {self['delta']:.3g} <= tol {self['tol']:g})"
                if self["ok"] else
                f"NOT converged over {self['parameter']} in "
                f"{[v for v, _ in self['history']]!r} (tol {self['tol']:g})")
        rows = [f"  {v!r:>10}   {'--' if d is None else f'{d:.3g}'}"
                for v, d in self["history"]]
        return "\n".join([head, f"  {'value':>10}   change vs previous", *rows])

    __repr__ = __str__


def converged(fn, values, tol=1e-6, metric=None, parameter="value", quiet=True):
    """Sweep `values`, stop as soon as the answer stops moving.

    fn(value) -> the answer at that setting. Must be comparable across the
        sweep (see the module docstring -- this is the one real constraint).
    values: settings to try, in increasing order of cost/accuracy
        (e.g. [4, 6, 8, 10] for n_max, [1e-6, 1e-8, 1e-10] for rtol,
        [10, 20, 40, 80] for Trotter steps).
    tol: convergence threshold on the change between CONSECUTIVE answers.
    metric(a, b) -> float: how to measure that change. Default is max|a-b|.
        For states compared as vectors, prefer something phase-blind such as
        `lambda a, b: 1 - fidelity(a, b)` -- a global phase difference is not a
        physical change but will dominate a raw subtraction.
    quiet: run the sweep inside `htdse.quiet()` (default) so the solver's own
        per-integration logging does not bury the result.

    Returns a `Convergence` mapping with `ok`, `value` (first setting that met
    tol), `answer`, `delta`, and the full `history` of (value, change) pairs.

    It stops at the FIRST value meeting `tol`, which means the answer is "this
    was enough", not "this is optimal". Convergence between two consecutive
    points is evidence, not proof: a quantity can sit still and then move again
    (a resonance the sweep stepped over). Widen `values` if that is plausible.
    """
    metric = metric or _default_metric
    values = list(values)
    if len(values) < 2:
        raise ValueError("converged needs at least two values to compare")

    prev_val = config.VERBOSE
    if quiet:
        config.VERBOSE = False
    try:
        history, prev, hit = [], None, None
        for v in values:
            answer = fn(v)
            delta = None if prev is None else metric(prev, answer)
            history.append((v, delta))
            if delta is not None and delta <= tol and hit is None:
                hit = (v, answer, delta)
                break
            prev = answer
    finally:
        config.VERBOSE = prev_val

    if hit is None:
        return Convergence(ok=False, value=None, answer=None, delta=None,
                           tol=tol, parameter=parameter, history=history)
    v, answer, delta = hit
    return Convergence(ok=True, value=v, answer=answer, delta=delta,
                       tol=tol, parameter=parameter, history=history)
