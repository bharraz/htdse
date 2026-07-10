"""Trotterization as a mechanism wrapper.

`TrotterizedMechanism` turns ANY mechanism's H(t) into its piecewise-constant
version: H is sampled once per step and held. Because it declares its step
edges as `breakpoints()` and sets `piecewise_constant = True`, the evolution
layer (a) never lets the adaptive ODE stepper integrate across a step edge,
and (b) propagates each step EXACTLY via the eigendecomposition of the held H
(U_step = V e^{-iE dt} V^dag) -- so Trotter evolution is both faster and free
of smooth-interpolant artifacts, while remaining just another `H(t)`.
"""
import numpy as np

from ..core.mechanism import Mechanism
from ..core.operator import Operator


class TrotterizedMechanism(Mechanism):
    """Piecewise-constant discretization of `inner` over [t_start, t_stop]
    in `n_steps` equal steps; H on each step is inner.hamiltonian at the step
    midpoint (`sample="midpoint"`, second-order accurate) or left edge
    (`sample="left"`).

    Outside [t_start, t_stop] the nearest step's H is held (querying far
    outside the discretized window is usually a sign of a setup error, but
    holding the edge value keeps continuation solves well-defined).
    """

    piecewise_constant = True

    def __init__(self, inner, t_start: float, t_stop: float, n_steps: int,
                 sample: str = "midpoint"):
        if t_stop <= t_start:
            raise ValueError("need t_stop > t_start")
        if sample not in ("midpoint", "left"):
            raise ValueError("sample must be 'midpoint' or 'left'")
        self.inner = inner
        self.t_start, self.t_stop, self.n_steps = t_start, t_stop, int(n_steps)
        self.sample = sample
        self._edges = np.linspace(t_start, t_stop, self.n_steps + 1)
        # carry the inner mechanism's subsystem structure forward, if any
        self.subsystems = dict(getattr(inner, "subsystems", {}) or {})

    def _sample_time(self, t: float) -> float:
        k = int(np.clip(np.searchsorted(self._edges, t, side="right") - 1,
                        0, self.n_steps - 1))
        if self.sample == "midpoint":
            return 0.5 * (self._edges[k] + self._edges[k + 1])
        return self._edges[k]

    def hamiltonian(self, t) -> Operator:
        return self.inner.hamiltonian(self._sample_time(t))

    def jump_operators(self, t) -> list:
        return self.inner.jump_operators(self._sample_time(t)) \
            if hasattr(self.inner, "jump_operators") else []

    def breakpoints(self):
        return self._edges  # step edges: H(t) is discontinuous exactly here

    def __repr__(self):
        return (f"TrotterizedMechanism({self.inner!r}, "
                f"[{self.t_start:g}, {self.t_stop:g}] / {self.n_steps} steps, "
                f"sample={self.sample!r})")
