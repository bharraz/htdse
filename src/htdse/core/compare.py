"""The target-vs-realized comparison loop, as one free function.

No generic `.compare()` method hides which metric is being computed -- you
pass the fidelity function explicitly (`fidelity`, `process_fidelity`,
`density_fidelity`, or your own). `compare_over` only removes the boilerplate
of walking two evolutions over a time grid and applying embed/trace adapters.
"""
import numpy as np


def _states_over(x, ts):
    """Accept an evolution (anything with .state_at) or a callable t -> array.

    Evolutions are queried with the whole grid at once: `state_at(ts)` is one
    solve over [min ts, max ts], whereas asking time-by-time restarts the
    integrator at every point (and leaves one stored segment per point).
    """
    if hasattr(x, "state_at"):
        return list(x.state_at(ts))
    return [x(t) for t in ts]


def compare_over(ts, target, realized, metric,
                 target_adapter=None, realized_adapter=None) -> np.ndarray:
    """metric(target(t), realized(t)) over a time grid; returns one value per t.

    ts:       array of times.
    target, realized: evolutions (anything with `.state_at(t)`) or callables
              t -> state/operator.
    metric:   explicit comparison function of two states/operators, e.g.
              `fidelity`, `process_fidelity`, `density_fidelity`.
    *_adapter: optional per-side function applied to each state before the
              metric -- this is where the embed/trace_out physics decision
              lives (e.g. `lambda rho: partial_trace(rho, dims, ("mode",))`,
              or a lift `lambda psi: U_embed @ psi`). The framework never
              guesses which side to adapt.
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))  # a scalar t is a 1-point grid
    a_states = _states_over(target, ts)
    b_states = _states_over(realized, ts)
    if target_adapter is not None:
        a_states = [target_adapter(a) for a in a_states]
    if realized_adapter is not None:
        b_states = [realized_adapter(b) for b in b_states]
    return np.array([metric(a, b) for a, b in zip(a_states, b_states)])
