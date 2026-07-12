from typing import List, Optional, Sequence

import numpy as np

from .operator import Operator


def _summarize(v) -> str:
    """Compact repr for a mechanism attribute -- big arrays/lists/callables
    are summarized, not dumped, so verbose solver lines stay readable."""
    if isinstance(v, np.ndarray):
        return f"<array {v.shape}>"
    if isinstance(v, (list, tuple)) and len(v) > 4:
        return f"<{type(v).__name__} len={len(v)}>"
    if callable(v):
        return f"<{getattr(v, '__name__', type(v).__name__)}>"
    return repr(v)


class Mechanism:
    """Base class for a physical process mapping its own control parameters
    to realized dynamics. Subclasses implement `.hamiltonian(t)` and/or
    `.unitary(t)` -- whichever the physics naturally gives.

    H(t) -> U (time-ordered exponential) is always well-defined, so
    `HamiltonianEvolution` only needs `.hamiltonian(t)`. The reverse, U -> H,
    is not (matrix log is branch-ambiguous), so a mechanism defined only as a
    gate (e.g. an analytic Magnus/RWA result) implements `.unitary(t)` alone --
    `UnitaryEvolution` and `DensityMatrixEvolution` consume it directly,
    skipping the ODE solve entirely.

    Optional solver hints (both about discontinuous H(t) -- an adaptive ODE
    stepper assumes a smooth right-hand side, so a step straddling a jump in
    H(t) can be silently inaccurate):

    - `breakpoints()`: times where H(t) is discontinuous. The solver never
      integrates across one; it restarts the integration at each.
    - `piecewise_constant = True` (class attribute): H(t) is exactly constant
      between consecutive breakpoints. Schrodinger-type evolutions then skip
      the ODE solver entirely and propagate each interval exactly via the
      eigendecomposition of H (U = V e^{-iE dt} V^dagger) -- faster AND exact.

    IMPORTANT: a mechanism is treated as frozen once handed to an evolution.
    The evolutions memoize solved segments; mutating parameters afterwards is
    detected and rejected (build a new mechanism/evolution instead).
    """

    piecewise_constant = False  # set True if H(t) is constant between breakpoints

    def hamiltonian(self, t: float) -> Operator:
        raise NotImplementedError(f"{type(self).__name__} has no .hamiltonian(t)")

    def H(self, t: float) -> Operator:
        """Alias for `hamiltonian(t)` -- H(t), the way it is written on paper.
        Dispatches to whatever `hamiltonian` the mechanism implements."""
        return self.hamiltonian(t)

    def unitary(self, t: Optional[float] = None) -> Operator:
        raise NotImplementedError(f"{type(self).__name__} has no .unitary(t)")

    def jump_operators(self, t: float) -> List[Operator]:
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


def provides_unitary(mechanism) -> bool:
    """True if this mechanism actually implements its own .unitary(t)
    (rather than inheriting the not-implemented default)."""
    fn = getattr(type(mechanism), "unitary", None)
    return fn is not None and fn is not Mechanism.unitary


def provides_hamiltonian(mechanism) -> bool:
    """True if this mechanism actually implements its own .hamiltonian(t)."""
    fn = getattr(type(mechanism), "hamiltonian", None)
    return fn is not None and fn is not Mechanism.hamiltonian
