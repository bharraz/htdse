"""The `System` protocol -- the one contract every evolution consumes.

A System is anything that answers "what are the dynamics at time t?", by
implementing `hamiltonian(t)` and/or `unitary(t)` (plus `jump_operators(t)`
if it dissipates). That is the whole interface. It is a `typing.Protocol`,
not a class hierarchy: the evolutions duck-type every attribute they read,
so inheriting is OPTIONAL and buys only convenience (the `H()` alias, a
readable `__repr__`, and NotImplementedError instead of AttributeError).

Two ways to build one, and the common case is not the object-oriented one:

  1. A function returning a `Model`. Write physics as a sum of named terms
     and let the term layer do embedding, caching, and group swapping:

         def my_drive(detune, amp, n_max) -> Model:
             return term(...) + term(...)

     `ms_lamb_dicke1`, `pauli_sum` and `term` itself are all this shape.
     A `Model` satisfies this protocol, so it goes straight into an evolution.

  2. A class implementing this protocol -- for physics that is NOT a sum of
     terms: a closed-form gate (`MSMagnus`), a wrapper (`TrotterizedSystem`),
     a bridge (`interop.qutip.as_mechanism`).

Choosing (2) costs less than it looks. Sparse support is duck-typed on the
matrix you return (return a CSR and the solver takes the sparse path), and
the truncation guard only needs a `subsystems` dict -- either expose one as
an attribute or pass `subsystems=` to the evolution. What you give up is the
`Model` *algebra*: `+`, `.replace()`, `.without()`, automatic identity
padding, and the static/dynamic materialization cache. For a closed-form
U(t) most of that is moot -- there is no sum of terms to compose anyway.
`embed()` in core/subsystems.py is still available a la carte.
"""
from typing import List, Optional, Protocol, Sequence

import numpy as np


def _summarize(v) -> str:
    """Compact repr for a system attribute -- big arrays/lists/callables
    are summarized, not dumped, so verbose solver lines stay readable."""
    if isinstance(v, np.ndarray):
        return f"<array {v.shape}>"
    if isinstance(v, (list, tuple)) and len(v) > 4:
        return f"<{type(v).__name__} len={len(v)}>"
    if callable(v):
        return f"<{getattr(v, '__name__', type(v).__name__)}>"
    return repr(v)


class System(Protocol):
    """A physical system whose dynamics an evolution can integrate.

    Implement `.hamiltonian(t)` and/or `.unitary(t)` -- whichever the physics
    naturally gives.

    H(t) -> U (time-ordered exponential) is always well-defined, so
    `HamiltonianEvolution` only needs `.hamiltonian(t)`. The reverse, U -> H,
    is not (matrix log is branch-ambiguous), so a system defined only as a
    gate (e.g. an analytic Magnus/RWA result) implements `.unitary(t)` alone --
    `UnitaryEvolution` and `DensityMatrixEvolution` consume it directly,
    skipping the ODE solve entirely.

    Optional, duck-typed hints the evolutions read if present:

    - `breakpoints()`: times where H(t) is discontinuous. The solver never
      integrates across one; it restarts the integration at each. (An adaptive
      ODE stepper assumes a smooth right-hand side, so a step straddling a jump
      in H(t) can be silently inaccurate.)
    - `piecewise_constant = True` (class attribute): H(t) is exactly constant
      between consecutive breakpoints. Schrodinger-type evolutions then skip
      the ODE solver entirely and propagate each interval exactly via the
      eigendecomposition of H (U = V e^{-iE dt} V^dagger) -- faster AND exact.
    - `subsystems`: {name: dim} of the tensor factors. Enables the truncation
      guard and per-subsystem diagnostics in `report()`. A `Model` has this
      for free; a hand-written system can set it (see `MSMagnus`) or the
      caller can pass `subsystems=` to the evolution instead.

    IMPORTANT: a system is treated as frozen once handed to an evolution.
    The evolutions memoize solved segments; mutating parameters afterwards is
    detected and rejected (build a new system/evolution instead).
    """

    piecewise_constant = False  # set True if H(t) is constant between breakpoints

    def hamiltonian(self, t: float) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no .hamiltonian(t)")

    def H(self, t: float) -> np.ndarray:
        """Alias for `hamiltonian(t)` -- H(t), the way it is written on paper.
        Dispatches to whatever `hamiltonian` the system implements."""
        return self.hamiltonian(t)

    def unitary(self, t: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no .unitary(t)")

    def jump_operators(self, t: float) -> List[np.ndarray]:
        """Lindblad jump operators L_k(t), each already scaled by sqrt(rate).
        Default: none (closed system). Only override for dissipation into a
        bath too large/uncharacterized to model as a subsystem -- a finite
        modeled subsystem should stay unitary + trace_out instead.
        """
        return []  # no dissipation

    def breakpoints(self) -> Sequence[float]:
        """Times where H(t) is discontinuous (e.g. Trotter step edges).
        Default: none (H(t) is smooth)."""
        return ()

    def __repr__(self):
        params = ", ".join(f"{k}={_summarize(v)}" for k, v in vars(self).items()
                           if not k.startswith("_"))
        text = f"{type(self).__name__}({params})"
        return text if len(text) <= 200 else text[:197] + "..."


def provides_unitary(system) -> bool:
    """True if this system actually implements its own .unitary(t)
    (rather than inheriting the not-implemented default)."""
    fn = getattr(type(system), "unitary", None)
    return fn is not None and fn is not System.unitary


def provides_hamiltonian(system) -> bool:
    """True if this system actually implements its own .hamiltonian(t)."""
    fn = getattr(type(system), "hamiltonian", None)
    return fn is not None and fn is not System.hamiltonian
