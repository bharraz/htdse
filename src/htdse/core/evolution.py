import hashlib
import pickle
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from . import config
from .mechanism import provides_hamiltonian, provides_unitary
from .operator import Operator
from ..util import MAG_THRESHOLD


# ---------------------------------------------------------------------------
# validity checks -- catch the silent-wrong failure modes at construction time
# ---------------------------------------------------------------------------

def _check_hermitian(H, what="H(t0)"):
    """A non-Hermitian generator gives non-unitary dynamics that just looks
    like mysterious decay -- the most common user sign error. Cheap to catch."""
    H = np.asarray(H)
    scale = max(1.0, np.max(np.abs(H)))
    defect = np.max(np.abs(H - H.conj().T))
    if defect > MAG_THRESHOLD * scale:
        raise ValueError(f"{what} is not Hermitian (max |H - H^dag| = {defect:.3g}). "
                         "Check the mechanism for a sign/conjugation error.")


def _check_density_matrix(rho, what="rho0"):
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"{what} must be a square matrix, got shape {rho.shape}")
    defect = np.max(np.abs(rho - rho.conj().T))
    if defect > MAG_THRESHOLD * max(1.0, np.max(np.abs(rho))):
        raise ValueError(f"{what} is not Hermitian (max |rho - rho^dag| = {defect:.3g})")
    # positivity: a hand-built rho0 with a negative eigenvalue is not a state at
    # all, and nothing downstream would notice. One eigvalsh at construction.
    lo = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2))
    if lo < -MAG_THRESHOLD * max(1.0, np.max(np.abs(rho))):
        raise ValueError(f"{what} is not positive semidefinite (min eigenvalue "
                         f"{lo:.3g}) -- not a valid density matrix.")
    tr = np.trace(rho)
    if abs(tr - 1.0) > 1e-6:
        warnings.warn(f"{what} has trace {tr:.6g} (expected 1). Evolving it anyway -- "
                      "make sure that's intentional.", stacklevel=3)


def _reject_dissipative(mechanism, t0, cls_name, alternative="LindbladEvolution"):
    """Closed-system evolutions silently IGNORE jump operators -- so refuse a
    dissipative mechanism outright instead of producing wrong physics.

    Sampling `jump_operators(t0)` alone would miss a channel that switches on at
    t > t0 (a time-dependent coefficient vanishing at t0). A term-layer
    Hamiltonian declares its channels structurally, so check that registry when
    it exists; for a hand-written Mechanism, sampling at t0 is all we have."""
    structural = getattr(mechanism, "jumps", None)
    dissipative = bool(structural) if isinstance(structural, dict) else False
    if not dissipative:
        jumps = getattr(mechanism, "jump_operators", None)
        dissipative = callable(jumps) and len(jumps(t0)) > 0
    if dissipative:
        raise ValueError(
            f"{cls_name} solves closed-system dynamics, but {type(mechanism).__name__} "
            f"has jump operators -- its dissipation would be silently ignored. "
            f"Use {alternative} instead.")


def _snapshot(mechanism):
    """Digest of a mechanism's parameters, to detect mutation after binding (the
    memoized solution would silently be stale physics). Returns None when the
    state isn't picklable (e.g. lambda coefficients) -- then the guard is
    skipped and the frozen-after-binding rule is on the caller.

    Hashed, not retained: the comparison is against a 16-byte digest rather than
    a full pickle of (possibly large) array attributes. The pickling itself is
    the cost, and it is unavoidable if the check is to be sound -- see
    `check_mutation=False` on the evolution classes to opt out in hot loops."""
    if mechanism is None:
        return None
    try:
        raw = pickle.dumps({k: v for k, v in vars(mechanism).items()
                            if not k.startswith("_")})
    except Exception:
        return None
    return hashlib.blake2b(raw, digest_size=16).digest()


# ---------------------------------------------------------------------------
# the shared lazy solver
# ---------------------------------------------------------------------------

class _EighSegment:
    """Exact propagation over one interval of CONSTANT Hermitian H:
    y(t) = V e^{-i E (t - t_start)} V^dag y0, columnwise for matrix states.
    Exact (to eigendecomposition precision) -- no ODE stepping, no interpolant."""

    def __init__(self, H, t_start, y0_flat, state_dim):
        E, V = np.linalg.eigh(np.asarray(H, dtype=complex))
        self._E, self._V = E, V
        self._t0 = t_start
        self._C = V.conj().T @ y0_flat.reshape(state_dim, -1)  # eigenbasis coords

    def __call__(self, t):
        phases = np.exp(-1j * self._E * (t - self._t0))
        return (self._V @ (phases[:, None] * self._C)).reshape(-1)


class _ExtendableSolver:
    """Internal: lazily solves dy/dt = rhs(t, y) for an arbitrary linear rhs.

    Physics-agnostic -- HamiltonianEvolution/UnitaryEvolution give it a
    Schrodinger-equation rhs, LindbladEvolution a master-equation rhs. Not
    part of the public API.

    Nothing is integrated until `state_at` is first called, and only the
    range of times actually requested gets solved. Every solved interval is
    kept; a request outside what's solved extends the solution by continuing
    from the nearest solved boundary -- an exact continuation of the same ODE,
    never an extrapolation past solved data.

    Mechanism breakpoints (discontinuities in H(t)) are never integrated
    across: the solve is split there and restarted, because an adaptive
    stepper straddling a jump can silently accept an interpolant fitted to
    the wrong physics. If the mechanism declares `piecewise_constant = True`
    and the evolution allows it (Schrodinger-type equations), each interval
    is instead propagated EXACTLY via eigendecomposition -- no ODE at all.

    Verbosity: `verbose=True/False` per instance, `None` (default) follows
    the package-wide `htdse.quiet()` / `config.VERBOSE` setting. Every real
    integration performed is printed.

    `check_mutation=False` disables the stale-physics guard, which otherwise
    pickles the mechanism's parameters once per `state_at` call. That costs
    ~0.3 ms for a mechanism carrying a 200x200 array attribute, so it is worth
    turning off in an optimizer inner loop -- and only there, since you then own
    the frozen-after-binding rule yourself.
    """

    def __init__(self, rhs, initial: Operator, t0: float = 0.0,
                 rtol=1e-8, atol=1e-10, method="RK45", verbose=None,
                 mechanism=None, label=None, expm_ok=False, check_mutation=True):
        self.rhs = rhs
        self.initial = initial if isinstance(initial, Operator) else Operator(initial)
        self.t0 = t0
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.verbose = verbose
        self.mechanism = mechanism
        self.label = label or type(self).__name__
        self._expm = bool(expm_ok and getattr(mechanism, "piecewise_constant", False))
        bps = getattr(mechanism, "breakpoints", None)
        self._breakpoints = np.sort(np.asarray(bps(), dtype=float)) if callable(bps) else np.array([])
        self._segments = []  # sorted, contiguous (t_lo, t_hi, y_of_t callable) around t0
        self._seg_los = None  # ascending t_lo array, rebuilt lazily for lookup
        self._lo_t, self._hi_t = t0, t0
        self._lo_y = self._hi_y = np.asarray(self.initial, dtype=complex).reshape(-1)  # flat state
        self._check_mutation = bool(check_mutation)
        self._mech_state = _snapshot(mechanism) if self._check_mutation else None
        if (self._check_mutation and mechanism is not None
                and self._mech_state is None and self._verbose):
            print(f"[{self.label}] note: {type(mechanism).__name__} has unpicklable "
                  "parameters (e.g. lambda coefficients), so the stale-physics guard "
                  "is unavailable -- do not mutate it while this evolution is alive.")

    @property
    def _verbose(self):
        return config.VERBOSE if self.verbose is None else self.verbose

    def _check_mechanism_unchanged(self):
        if self._mech_state is None:
            return  # unpicklable parameters, or opted out -- rule is on the caller
        if _snapshot(self.mechanism) != self._mech_state:
            raise RuntimeError(
                f"[{self.label}] {type(self.mechanism).__name__}'s parameters changed "
                "after this evolution was created. The memoized solution would silently "
                "continue from stale physics -- build a new mechanism + evolution instead.")

    def _split_at_breakpoints(self, t_start, t_end):
        """Points partitioning [t_start, t_end] (either direction) so that no
        sub-interval crosses a declared discontinuity of H(t)."""
        lo, hi = min(t_start, t_end), max(t_start, t_end)
        inner = self._breakpoints[(self._breakpoints > lo) & (self._breakpoints < hi)]
        pts = np.concatenate(([t_start], inner if t_end > t_start else inner[::-1], [t_end]))
        return pts

    def _solve_one(self, t_start, t_end, y_start):
        """Solve one breakpoint-free interval; returns a y_of_t callable."""
        if self._expm:
            if self._verbose:
                print(f"[{self.label}] expm-propagating {self.mechanism!r}: "
                      f"t={t_start:.6g} -> {t_end:.6g} (piecewise-constant H, exact)")
            H = self.mechanism.hamiltonian((t_start + t_end) / 2)  # constant on interval
            return _EighSegment(H, t_start, y_start, self.initial.shape[0])
        if self._verbose:
            print(f"[{self.label}] integrating {self.mechanism!r}: "
                  f"t={t_start:.6g} -> {t_end:.6g}, method={self.method}, "
                  f"rtol={self.rtol:g}, atol={self.atol:g}")
        sol = solve_ivp(
            self.rhs, (t_start, t_end), y_start,
            method=self.method, dense_output=True, rtol=self.rtol, atol=self.atol,
        )
        if self._verbose:
            print(f"[{self.label}]   done: success={sol.success}, "
                  f"steps={len(sol.t)}, rhs evals={sol.nfev}")
        if not sol.success:
            raise RuntimeError(f"integration failed on [{t_start}, {t_end}]: {sol.message}")
        return sol.sol

    def _solve_range(self, t_start, t_end, y_start):
        """Chain solves across breakpoints; returns (segments, y at t_end)."""
        pts = self._split_at_breakpoints(t_start, t_end)
        segments = []
        y = y_start
        for a, b in zip(pts[:-1], pts[1:]):
            fn = self._solve_one(a, b, y)
            y = fn(b)
            segments.append((min(a, b), max(a, b), fn))
        return segments, y

    def _extend_to(self, t_min, t_max):
        """Grow the solved range to cover [t_min, t_max] by continuing from
        the current edges, not by resolving from scratch."""
        if t_max > self._hi_t:
            segs, y_end = self._solve_range(self._hi_t, t_max, self._hi_y)
            self._segments.extend(segs)
            self._seg_los = None
            self._hi_t, self._hi_y = t_max, y_end  # advance right edge
        if t_min < self._lo_t:
            segs, y_end = self._solve_range(self._lo_t, t_min, self._lo_y)
            self._segments = segs[::-1] + self._segments  # keep ascending order
            self._seg_los = None
            self._lo_t, self._lo_y = t_min, y_end  # advance left edge

    def _value_at(self, t):
        if t == self.t0:
            return np.asarray(self.initial, dtype=complex).reshape(-1)
        # segments are ascending and contiguous, so binary-search the one holding
        # t -- a scan is O(n_segments) per query, and a Trotter solve stores one
        # segment per step.
        if self._seg_los is None:
            self._seg_los = np.array([lo for lo, _, _ in self._segments])
        i = int(np.searchsorted(self._seg_los, t, side="right")) - 1
        if 0 <= i < len(self._segments):
            lo, hi, fn = self._segments[i]
            if lo <= t <= hi:
                return fn(t)  # within-solved-range evaluation only
        raise RuntimeError(f"t={t} not covered after extension (internal error)")

    def state_at(self, t) -> Operator:
        """Evolved state at time(s) t (scalar or array-like). Extends the
        solved range as needed; never extrapolates."""
        # Checked on EVERY query, not just on extension: a mutated mechanism is
        # stale physics even when the answer comes from an already-solved
        # segment, and returning it silently is exactly the failure this guards.
        self._check_mechanism_unchanged()
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        self._extend_to(t_arr.min(), t_arr.max())
        out = np.empty((len(t_arr),) + self.initial.shape, dtype=complex)
        for i, tt in enumerate(t_arr):
            out[i] = self._value_at(tt).reshape(self.initial.shape)
        if np.ndim(t) == 0:
            return Operator(out[0])
        return Operator(out)


def _schrodinger_rhs(mechanism, shape):
    """dX/dt = -i H(t) X, for X a ket (d,) or an operator (d,d)."""
    def rhs(t, y_flat):
        # plain ndarray: every Operator result would copy a params dict, and this
        # runs once per integrator function evaluation
        H = np.asarray(mechanism.hamiltonian(t))
        X = y_flat.reshape(shape)
        return (-1j * (H @ X)).reshape(-1)  # -i H X, flattened for the ODE solver
    return rhs


def _default_subsystems(mechanism, subsystems):
    """Explicit `subsystems=` wins; otherwise a mechanism that knows its own
    tensor structure (e.g. a term-layer Hamiltonian) supplies it."""
    if subsystems is not None:
        return dict(subsystems)
    return dict(getattr(mechanism, "subsystems", {}) or {})


# ---------------------------------------------------------------------------
# the four evolution classes -- one per equation of motion
# ---------------------------------------------------------------------------

class HamiltonianEvolution:
    """State-vector evolution: i d|psi(t)>/dt = H(t)|psi(t)>, |psi(t0)> = initial.

    `subsystems`: ordered {name: dim} of this state's tensor factors, needed
    by `trace_out`. Order must match how `initial` was built (e.g. via
    `otimes`). Defaults to the mechanism's own `.subsystems` when it has one
    (term-layer Hamiltonians always do).

    Every time-parametrized method accepts a scalar t or an array of times.
    """

    def __init__(self, mechanism, initial: Operator, t0: float = 0.0,
                 subsystems: dict | None = None, **solver_kwargs):
        initial = initial if isinstance(initial, Operator) else Operator(initial)
        _reject_dissipative(mechanism, t0, "HamiltonianEvolution")
        _check_hermitian(mechanism.hamiltonian(t0))
        self.mechanism = mechanism
        rhs = _schrodinger_rhs(mechanism, initial.shape)
        self._solver = _ExtendableSolver(rhs, initial, t0, mechanism=mechanism,
                                         label="HamiltonianEvolution", expm_ok=True,
                                         **solver_kwargs)
        self.subsystems = _default_subsystems(mechanism, subsystems)

    def state_at(self, t) -> Operator:
        return self._solver.state_at(t)

    def _require_ket(self, what):
        """`state_at` returns (n_times, d) for an array t and (d, d) for a
        matrix-valued initial (a propagator, or stacked kets evolved in one
        solve). Those are indistinguishable downstream, so anything that reads
        its input as a ket trajectory must refuse a matrix initial outright
        rather than return a plausibly-shaped wrong answer."""
        if self._solver.initial.ndim != 1:
            raise ValueError(
                f"{what} is defined for a state-vector evolution, but this one was "
                f"built with a {self._solver.initial.ndim}-d initial condition "
                f"{self._solver.initial.shape} (a propagator or a stack of kets). "
                "Evolve a single ket, or use DensityMatrixEvolution / UnitaryEvolution.")

    def trace_out(self, *names, t) -> Operator:
        """Reduced density matrix at time(s) t, tracing out the named
        subsystems. Scalar t -> (d, d); array t -> (n_times, d, d)."""
        from .subsystems import partial_trace
        self._require_ket("trace_out")
        psi = self.state_at(t)                                   # (d,) or (n, d)
        rho = psi[..., :, None] * psi.conj()[..., None, :]        # batched |psi><psi|
        return partial_trace(Operator(rho), self.subsystems, names)

    def instantaneous_eigenbasis(self, t):
        """Eigenbasis of H(t) itself (not the evolved state): H(t)|n(t)> = E_n(t)|n(t)>.

        Scalar t only. Returns (evals, evecs) sorted ascending -- evecs[:, 0]
        is the instantaneous ground state, the reference adiabaticity is
        measured against. Caveat: at a (near-)degeneracy the ordering and the
        basis within the degenerate subspace are arbitrary, so per-level
        quantities can jump discontinuously exactly where gaps close.
        """
        H = self.mechanism.hamiltonian(t)
        return np.linalg.eigh(H)  # Hermitian eigendecomposition, ascending order

    def adiabatic_populations(self, t) -> np.ndarray:
        """Population in each instantaneous eigenstate of H(t):
        |<n(t)|psi(t)>|^2, ordered by ascending E_n(t).
        Scalar t -> (dim,); array t -> (n_times, dim)."""
        self._require_ket("adiabatic_populations")
        if np.ndim(t) > 0:
            # one batched state_at (one solve, one mutation check) rather than a
            # scalar recursion that pays both per time point
            ts = np.asarray(t)
            psis = self.state_at(ts)
            return np.array([np.abs(self.instantaneous_eigenbasis(tt)[1].conj().T @ psi) ** 2
                             for tt, psi in zip(ts, psis)])
        _, evecs = self.instantaneous_eigenbasis(t)
        psi = self.state_at(t)
        return np.abs(evecs.conj().T @ psi) ** 2  # overlap with each eigenvector

    def adiabatic_fidelity(self, t):
        """Population remaining in the instantaneous ground state -- 1 for a
        perfectly adiabatic ramp, less for diabatic leakage. Scalar t -> float;
        array t -> array. See the degeneracy caveat on instantaneous_eigenbasis."""
        pops = self.adiabatic_populations(t)
        return float(pops[0]) if np.ndim(t) == 0 else pops[:, 0]


class UnitaryEvolution:
    """Propagator evolution: i d/dt U(t) = H(t) U(t), U(t0) = initial (usually I).

    Pass either `initial` (an existing propagator to continue) or `dim`
    (to start from the dim x dim identity at t0).

    A mechanism that implements its own `.unitary(t)` (an analytic Magnus/RWA
    result defined as a gate) is consumed directly -- no ODE solve, no U -> H
    inversion. In that case `initial` must be omitted or I: composing an
    analytic U(t, t0) with a different starting propagator is the mechanism's
    business, not something to guess here.
    """

    def __init__(self, mechanism, initial: Operator | None = None, dim: int | None = None,
                 t0: float = 0.0, **solver_kwargs):
        _reject_dissipative(mechanism, t0, "UnitaryEvolution")
        self.mechanism = mechanism
        self._analytic = provides_unitary(mechanism) and not provides_hamiltonian(mechanism)
        if self._analytic:
            if initial is not None and not np.allclose(np.asarray(initial),
                                                       np.eye(initial.shape[0])):
                raise ValueError("mechanism provides an analytic unitary; a non-identity "
                                 "`initial` propagator can't be composed with it here")
            if t0 != 0:
                # .unitary(t) is U(t, its_own_origin); re-anchoring it at t0 means
                # U(t) U(t0)^dag, which is only the propagator from t0 when H
                # commutes with itself at different times. The mechanism's call.
                raise ValueError(
                    f"mechanism provides an analytic unitary from its own origin, so "
                    f"t0={t0} would be silently ignored. Build the mechanism with the "
                    f"origin you want, or evolve its .hamiltonian(t) instead.")
            self._solver = None
            return
        if initial is None:
            if dim is None:
                raise ValueError("UnitaryEvolution needs either initial or dim")
            initial = Operator(np.eye(dim, dtype=complex))  # U(t0) = I
        elif dim is not None and np.shape(initial)[0] != dim:
            raise ValueError(f"UnitaryEvolution got both `initial` (dimension "
                             f"{np.shape(initial)[0]}) and dim={dim}, which disagree; "
                             f"`initial` wins, so drop `dim` or make them match")
        initial = initial if isinstance(initial, Operator) else Operator(initial)
        _check_hermitian(mechanism.hamiltonian(t0))
        rhs = _schrodinger_rhs(mechanism, initial.shape)
        self._solver = _ExtendableSolver(rhs, initial, t0, mechanism=mechanism,
                                         label="UnitaryEvolution", expm_ok=True,
                                         **solver_kwargs)

    def unitary_at(self, t) -> Operator:
        """The propagator U(t) (scalar t) or a stack of them (array t)."""
        if self._analytic:
            if np.ndim(t) == 0:
                return Operator(self.mechanism.unitary(t))
            return Operator(np.array([np.asarray(self.mechanism.unitary(tt))
                                      for tt in np.asarray(t)]))
        return self._solver.state_at(t)

    # kept as an alias: every evolution class answers state_at
    state_at = unitary_at

    def unitarity_defect(self, t) -> float:
        """max |U(t)^dag U(t) - I| -- how far numerical error has drifted the
        propagator off the unitary group. A solver-accuracy diagnostic."""
        U = np.asarray(self.unitary_at(t))
        d = U.shape[-1]
        return float(np.max(np.abs(np.swapaxes(U.conj(), -1, -2) @ U - np.eye(d))))


class DensityMatrixEvolution:
    """Closed-system (no dissipation) density matrix evolution:

        rho(t) = U(t) rho0 U(t)^dagger

    Computed by evolving the propagator U(t) (a UnitaryEvolution -- or the
    mechanism's own analytic `.unitary(t)` if it has one) and conjugating on
    demand. Exact up to the accuracy of U itself; rho inherits U's solver
    error twice (U and U^dag), so `unitarity_defect(t)` is exposed as the
    relevant diagnostic. Once a mechanism has dissipation (jump operators)
    the conjugation identity breaks and LindbladEvolution is required --
    passing a dissipative mechanism here raises.
    """

    def __init__(self, mechanism, rho0: Operator, t0: float = 0.0,
                 subsystems: dict | None = None, **solver_kwargs):
        _reject_dissipative(mechanism, t0, "DensityMatrixEvolution")
        _check_density_matrix(rho0)
        self.mechanism = mechanism
        self.rho0 = rho0 if isinstance(rho0, Operator) else Operator(rho0)
        dim = self.rho0.shape[0]
        self._U = UnitaryEvolution(mechanism, dim=dim, t0=t0, **solver_kwargs)
        self.subsystems = _default_subsystems(mechanism, subsystems)

    def state_at(self, t) -> Operator:
        U = self._U.unitary_at(t)
        if U.ndim == 2:
            return Operator(U @ self.rho0 @ U.conj().T)  # single time: U rho0 U^dagger
        rho_t = np.einsum("nij,jk,nlk->nil", U, self.rho0, U.conj())  # batched over time axis n
        return Operator(rho_t)

    def unitarity_defect(self, t) -> float:
        return self._U.unitarity_defect(t)

    def trace_out(self, *names, t) -> Operator:
        """rho at time(s) t, tracing out the named subsystems (batched over t)."""
        from .subsystems import partial_trace
        rho = self.state_at(t)
        return partial_trace(rho, self.subsystems, names)


def _lindblad_rhs(mechanism, dim):
    """d(rho)/dt = -i[H(t),rho] + sum_k ( L_k rho L_k^dagger - 1/2{L_k^dagger L_k, rho} )."""
    def rhs(t, y_flat):
        rho = y_flat.reshape(dim, dim)
        H = np.asarray(mechanism.hamiltonian(t))
        drho = -1j * (H @ rho - rho @ H)  # coherent part: -i[H, rho]
        for L in mechanism.jump_operators(t):
            L = np.asarray(L)
            Ld = L.conj().T
            LdL = Ld @ L  # once, not once per anticommutator half
            drho += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)  # dissipator D[L]
        return drho.reshape(-1)
    return rhs


class LindbladEvolution:
    """Open-system density matrix evolution via the Lindblad master equation:

        d(rho)/dt = -i[H(t), rho] + sum_k ( L_k rho L_k^dagger - 1/2{L_k^dagger L_k, rho} )

    Needed whenever a mechanism has jump operators -- dissipation into a bath
    too large/uncharacterized to model as a subsystem. Genuinely different
    from HamiltonianEvolution/UnitaryEvolution's dX/dt = -iHX: not obtainable
    via conjugation by a propagator the way DensityMatrixEvolution is.

    Forward-only: `state_at(t)` for t < t0 is rejected. Trace preservation
    holds integrating either direction, but positivity is only guaranteed by
    the forward semigroup -- backward integration can (and does) produce a
    matrix with negative eigenvalues, i.e. not a valid density matrix.
    """

    def __init__(self, mechanism, rho0: Operator, t0: float = 0.0,
                 subsystems: dict | None = None, **solver_kwargs):
        _check_density_matrix(rho0)
        _check_hermitian(mechanism.hamiltonian(t0))
        self.mechanism = mechanism
        self.rho0 = rho0 if isinstance(rho0, Operator) else Operator(rho0)
        self.t0 = t0
        dim = self.rho0.shape[0]
        rhs = _lindblad_rhs(mechanism, dim)
        self._solver = _ExtendableSolver(rhs, self.rho0, t0, mechanism=mechanism,
                                         label="LindbladEvolution", expm_ok=False,
                                         **solver_kwargs)
        self.subsystems = _default_subsystems(mechanism, subsystems)

    def state_at(self, t) -> Operator:
        if np.any(np.asarray(t) < self.t0):
            raise ValueError(
                f"LindbladEvolution.state_at: requested t < t0 ({self.t0}). Backward "
                "integration of a Lindbladian isn't guaranteed positive -- would "
                "silently return a non-physical density matrix. Not supported."
            )
        return self._solver.state_at(t)

    def trace_out(self, *names, t) -> Operator:
        """rho at time(s) t, tracing out the named subsystems (batched over t)."""
        from .subsystems import partial_trace
        rho = self.state_at(t)
        return partial_trace(rho, self.subsystems, names)
