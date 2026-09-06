import hashlib
import pickle
from collections.abc import Mapping
import warnings

import numpy as np
from scipy import sparse as _sp

from . import config
from .qutip_backend import compile_system, qobj, solve, solver_options
from .system import provides_hamiltonian, provides_unitary
from .truncation import resolve_threshold, warn_if_truncated
from ..util import MAG_THRESHOLD


# ---------------------------------------------------------------------------
# validity checks -- catch the silent-wrong failure modes at construction time
# ---------------------------------------------------------------------------

def _dense(H) -> np.ndarray:
    """Densify a possibly-sparse operator (sparse systems hand out CSR)."""
    return H.toarray() if _sp.issparse(H) else np.asarray(H)


def _h_of(system):
    """The solver's H(t) accessor.

    A System's PUBLIC `hamiltonian(t)` always densifies, because a user asking
    to see the matrix wants one they can plot and index. The solver wants the
    native storage -- keeping a sparse model sparse is the entire point -- so
    it prefers `_h_native` when the system offers one. A hand-written system
    has no `_h_native`, and whatever it returns (dense or CSR) is handled."""
    return getattr(system, "_h_native", getattr(system, "hamiltonian"))


def _check_hermitian(H, what="H(t0)"):
    """A non-Hermitian generator gives non-unitary dynamics that just looks
    like mysterious decay -- the most common user sign error. Cheap to catch.
    Works on dense and scipy-sparse H alike (the sparse branch never densifies:
    abs/max/subtraction all stay sparse)."""
    if _sp.issparse(H):
        scale = max(1.0, abs(H).max() if H.nnz else 0.0)
        diff = H - H.conj().T
        defect = abs(diff).max() if diff.nnz else 0.0
        if defect > MAG_THRESHOLD * scale:
            raise ValueError(f"{what} is not Hermitian (max |H - H^dag| = {defect:.3g}). "
                             "Check the system for a sign/conjugation error.")
        return
    H = np.asarray(H)
    scale = max(1.0, np.max(np.abs(H)))
    defect = np.max(np.abs(H - H.conj().T))
    if defect > MAG_THRESHOLD * scale:
        raise ValueError(f"{what} is not Hermitian (max |H - H^dag| = {defect:.3g}). "
                         "Check the system for a sign/conjugation error.")


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


def _is_dissipative(system, t0) -> bool:
    """Whether `system` carries jump operators at (or structurally, regardless
    of) `t0`. Sampling `jump_operators(t0)` alone would miss a channel that
    switches on at t > t0 (a time-dependent coefficient vanishing at t0). A
    term-built System declares its channels structurally, so check that
    registry when it exists; for a hand-written System, sampling at t0 is
    all we have."""
    structural = getattr(system, "jumps", None)
    dissipative = bool(structural) if isinstance(structural, Mapping) else False
    if not dissipative:
        jumps = getattr(system, "jump_operators", None)
        dissipative = callable(jumps) and len(jumps(t0)) > 0
    return dissipative


def _reject_dissipative(system, t0, cls_name, alternative="LindbladEvolution"):
    """Closed-system evolutions silently IGNORE jump operators -- so refuse a
    dissipative system outright instead of producing wrong physics."""
    if _is_dissipative(system, t0):
        raise ValueError(
            f"{cls_name} solves closed-system dynamics, but {type(system).__name__} "
            f"has jump operators -- its dissipation would be silently ignored. "
            f"Use {alternative} instead.")


def _snapshot(system):
    """Digest of a system's parameters, to detect mutation after binding (the
    memoized solution would silently be stale physics). Returns None when the
    state isn't picklable (e.g. lambda coefficients) -- then the guard is
    skipped and the frozen-after-binding rule is on the caller.

    Hashed, not retained: the comparison is against a 16-byte digest rather than
    a full pickle of (possibly large) array attributes. The pickling itself is
    the cost, and it is unavoidable if the check is to be sound -- see
    `check_mutation=False` on the evolution classes to opt out in hot loops."""
    if system is None:
        return None
    if getattr(system, "_immutable", False):
        return None
    try:
        raw = pickle.dumps({k: v for k, v in vars(system).items()
                            if not k.startswith("_")})
    except Exception:
        return None
    return hashlib.blake2b(raw, digest_size=16).digest()


# ---------------------------------------------------------------------------
# the shared lazy solver
# ---------------------------------------------------------------------------

class _QutipSolver:
    """Lazy NumPy-facing wrapper around the single QuTiP solver backend."""

    def __init__(self, initial, t0: float = 0.0,
                 rtol=1e-8, atol=1e-10, method="dop853", verbose=None,
                 system=None, label=None, expm_ok=False, check_mutation=True,
                 dissipative=False):
        self.initial = np.asarray(initial)
        self.t0 = t0
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.verbose = verbose
        self.system = system
        self.label = label or type(self).__name__
        self._expm = bool(expm_ok and getattr(system, "piecewise_constant", False))
        bps = getattr(system, "breakpoints", None)
        self._breakpoints = np.sort(np.asarray(bps(), dtype=float)) if callable(bps) else np.array([])
        self._solver_runs = 0
        self._segments = []
        self._lo_t, self._hi_t = t0, t0
        self._values = {float(t0): np.asarray(self.initial, dtype=complex).copy()}
        self._dissipative = bool(dissipative)
        # Compilation copies every constant operator into a Qobj.  The backend
        # therefore owns an immutable numerical snapshot rather than caller
        # arrays or scheduled-tone containers.
        self._H, self._c_ops = compile_system(system, include_jumps=dissipative)
        self._check_mutation = bool(check_mutation)
        self._system_state = _snapshot(system) if self._check_mutation else None
        if (self._check_mutation and system is not None
                and self._system_state is None and self._verbose):
            print(f"[{self.label}] note: {type(system).__name__} has unpicklable "
                  "parameters (e.g. lambda coefficients), so the stale-physics guard "
                  "is unavailable -- do not mutate it while this evolution is alive.")

    @property
    def _verbose(self):
        return config.VERBOSE if self.verbose is None else self.verbose

    def _check_system_unchanged(self):
        if self._system_state is None:
            return  # unpicklable parameters, or opted out -- rule is on the caller
        if _snapshot(self.system) != self._system_state:
            raise RuntimeError(
                f"[{self.label}] {type(self.system).__name__}'s parameters changed "
                "after this evolution was created. The memoized solution would silently "
                "continue from stale physics -- build a new system + evolution instead.")

    def _solve_grid(self, requested, direction):
        end = max(requested) if direction > 0 else min(requested)
        lo, hi = sorted((self.t0, end))
        inner = self._breakpoints[(self._breakpoints > lo) & (self._breakpoints < hi)]
        grid = np.unique(np.concatenate(([self.t0], requested, inner)))
        if direction < 0:
            grid = grid[::-1]
        if self._verbose:
            selected = "diag" if self._expm else solver_options(
                self.method, self.rtol, self.atol)["method"]
            print(f"[{self.label}] QuTiP integrating {self.system!r}: "
                  f"t={self.t0:.6g} -> {end:.6g}, method={selected}, "
                  f"rtol={self.rtol:g}, atol={self.atol:g}")

        if self._expm:
            # A Trotterized provider is constant only between breakpoints.
            # Restart QuTiP's diagonal integrator at every boundary so no
            # adaptive step can straddle a Hamiltonian jump.
            state = np.asarray(self.initial, dtype=complex)
            values = {float(self.t0): state.copy()}
            for a, b in zip(grid[:-1], grid[1:]):
                midpoint = 0.5 * (a + b)
                H = qobj(_h_of(self.system)(midpoint),
                         getattr(self.system, "subsystems", {}))
                segment = solve(H, state, [a, b],
                                subsystems=getattr(self.system, "subsystems", {}),
                                method="diag", rtol=self.rtol, atol=self.atol)
                state = segment[-1]
                values[float(b)] = state.copy()
        else:
            states = solve(self._H, self.initial, grid,
                           subsystems=getattr(self.system, "subsystems", {}),
                           c_ops=self._c_ops if self._dissipative else (),
                           method=self.method, rtol=self.rtol, atol=self.atol)
            values = {float(t): state for t, state in zip(grid, states)}
        self._values.update(values)
        self._solver_runs += max(1, len(grid) - 1) if self._expm else 1
        self._segments.extend((min(a, b), max(a, b), None)
                              for a, b in zip(grid[:-1], grid[1:]))
        self._lo_t = min(self._lo_t, float(end))
        self._hi_t = max(self._hi_t, float(end))

    def state_at(self, t) -> np.ndarray:
        """Evolved state at time(s) t (scalar or array-like). Extends the
        solved range as needed; never extrapolates."""
        # Checked on EVERY query, not just on extension: a mutated system is
        # stale physics even when the answer comes from an already-solved
        # segment, and returning it silently is exactly the failure this guards.
        self._check_system_unchanged()
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        missing = np.asarray([tt for tt in np.unique(t_arr)
                              if float(tt) not in self._values])
        forward = missing[missing > self.t0]
        backward = missing[missing < self.t0]
        if len(forward):
            self._solve_grid(forward, +1)
        if len(backward):
            self._solve_grid(backward, -1)
        out = np.empty((len(t_arr),) + self.initial.shape, dtype=complex)
        for i, tt in enumerate(t_arr):
            out[i] = self._values[float(tt)].reshape(self.initial.shape)
        if np.ndim(t) == 0:
            return np.asarray(out[0])
        return np.asarray(out)
def _default_subsystems(system, subsystems):
    """Explicit `subsystems=` wins; otherwise a system that knows its own
              tensor structure (e.g. a term-built System) supplies it."""
    if subsystems is not None:
        return dict(subsystems)
    return dict(getattr(system, "subsystems", {}) or {})


class Report(dict):
    """What a solve actually did, as a dict that prints like a lab note.

    `ev.report()["truncation"]` for the numbers, `print(ev.report())` for a
    human. See `_Reportable.report`."""

    _ORDER = ("system", "equation", "solved_range", "segments", "propagation",
              "solver_runs", "breakpoints", "mutation_guard",
              "truncation", "unitarity_defect", "trace")

    def __str__(self):
        width = max(len(k) for k in self) if self else 0
        lines = [f"{self.get('equation', 'evolution')} report"]
        for k in self._ORDER:
            if k not in self or k == "equation":
                continue
            v = self[k]
            if isinstance(v, float):
                v = f"{v:.6g}"
            elif isinstance(v, dict):
                v = ", ".join(f"{n}={p:.3g}" for n, p in v.items()) or "(none checked)"
            lines.append(f"  {k:<{width}}  {v}")
        return "\n".join(lines)

    __repr__ = __str__


class _Reportable:
    """`report()` for the evolution classes -- one honest summary of a solve.

    Everything here is already computed or tracked somewhere; the point is that
    "was this run trustworthy?" should be one call rather than a scavenger hunt
    across warnings, `unitarity_defect`, and the verbose log."""

    _report_kind = "ket"      # overridden per class; drives the truncation check

    def report(self, t=None) -> Report:
        """Diagnostics for this evolution, as of the times solved so far.

        t: where to evaluate the state-dependent checks (truncation, unitarity,
        trace). Defaults to the far edge of what has been solved, so calling
        `report()` costs nothing extra and never triggers a fresh solve. Passing
        a t outside the solved range WILL extend the solve, exactly like
        `state_at` would.

        Reads, it does not judge: a large `unitarity_defect` or a nonzero
        `truncation` entry means the numbers upstream are suspect, and this
        tells you so without deciding what to do about it.
        """
        solver = getattr(self, "_solver", None)
        rep = Report(system=repr(getattr(self, "system", None)),
                     equation=type(self).__name__)

        if solver is None:      # analytic-unitary path: no ODE ever runs
            rep["propagation"] = "analytic .unitary(t), no solve"
            rep["solved_range"] = "n/a"
        else:
            solved = (solver._lo_t, solver._hi_t)
            rep["solved_range"] = ("nothing solved yet" if solved[0] == solved[1]
                                   else f"[{solved[0]:.6g}, {solved[1]:.6g}]")
            rep["segments"] = len(solver._segments)
            rep["propagation"] = (
                "QuTiP diag, exact per piecewise-constant interval" if solver._expm
                else f"QuTiP {solver_options(solver.method, solver.rtol, solver.atol)['method']}, "
                     f"rtol={solver.rtol:g}, atol={solver.atol:g}")
            rep["solver_runs"] = solver._solver_runs
            rep["breakpoints"] = len(solver._breakpoints)
            rep["mutation_guard"] = (
                "disabled (check_mutation=False)" if not solver._check_mutation
                else "not needed (immutable System)" if getattr(self.system, "_immutable", False)
                else "active" if solver._system_state is not None
                else "UNAVAILABLE -- system has unpicklable parameters "
                     "(e.g. lambda coefficients); do not mutate it")
            if t is None and solved[0] != solved[1]:
                t = solver._hi_t

        if t is None:
            return rep

        state = self.state_at(t)
        rep["at_t"] = float(t)
        subs = getattr(self, "subsystems", {}) or {}
        if subs:
            from .truncation import truncation_populations
            rep["truncation"] = truncation_populations(
                state, subs, self._report_kind,
                getattr(self, "_ladders", None))
        if hasattr(self, "unitarity_defect"):
            rep["unitarity_defect"] = self.unitarity_defect(t)
        if self._report_kind == "density":
            rep["trace"] = float(np.real(np.trace(np.asarray(state))))
        return rep


# ---------------------------------------------------------------------------
# the four evolution classes -- one per equation of motion
# ---------------------------------------------------------------------------

class HamiltonianEvolution(_Reportable):
    """State-vector evolution: i d|psi(t)>/dt = H(t)|psi(t)>, |psi(t0)> = initial.

    `subsystems`: ordered {name: dim} of this state's tensor factors, needed
    by `trace_out`. Order must match how `initial` was built (e.g. via
    `otimes`). Defaults to the system's own `.subsystems` when it has one
    (term-built Systems always do).

    Every time-parametrized method accepts a scalar t or an array of times.
    """

    def __init__(self, system, initial, t0: float = 0.0,
                 subsystems: dict | None = None, truncation=None,
                 ladders=None, **solver_kwargs):
        initial = np.asarray(initial)
        _reject_dissipative(system, t0, "HamiltonianEvolution")
        _check_hermitian(_h_of(system)(t0))
        self.system = system
        self._solver = _QutipSolver(initial, t0, system=system,
                                    label="HamiltonianEvolution", expm_ok=True,
                                    **solver_kwargs)
        self.subsystems = _default_subsystems(system, subsystems)
        self._truncation, self._ladders = truncation, ladders
        self._trunc_seen = set()

    def state_at(self, t) -> np.ndarray:
        state = self._solver.state_at(t)
        # a matrix initial is a propagator / stack of kets, not a ket trajectory
        kind = "ket" if self._solver.initial.ndim == 1 else "unitary"
        warn_if_truncated(state, self.subsystems, kind,
                          resolve_threshold(self._truncation), self._ladders,
                          self._trunc_seen, "HamiltonianEvolution")
        return state

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

    def trace_out(self, *names, t) -> np.ndarray:
        """Reduced density matrix at time(s) t, tracing out the named
        subsystems. Scalar t -> (d, d); array t -> (n_times, d, d)."""
        from .subsystems import partial_trace
        self._require_ket("trace_out")
        psi = self.state_at(t)                                   # (d,) or (n, d)
        rho = psi[..., :, None] * psi.conj()[..., None, :]        # batched |psi><psi|
        return partial_trace(np.asarray(rho), self.subsystems, names)

    def instantaneous_eigenbasis(self, t):
        """Eigenbasis of H(t) itself (not the evolved state): H(t)|n(t)> = E_n(t)|n(t)>.

        Scalar t only. Returns (evals, evecs) sorted ascending -- evecs[:, 0]
        is the instantaneous ground state, the reference adiabaticity is
        measured against. Caveat: at a (near-)degeneracy the ordering and the
        basis within the degenerate subspace are arbitrary, so per-level
        quantities can jump discontinuously exactly where gaps close.
        """
        # densified even for a sparse system: a full eigendecomposition is
        # inherently dense (O(d^2) memory) -- fine as a diagnostic at moderate
        # dim, not something to call at dimensions only sparse can evolve
        H = _dense(self.system.hamiltonian(t))
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


class UnitaryEvolution(_Reportable):
    """Propagator evolution: i d/dt U(t) = H(t) U(t), U(t0) = initial (usually I).

    Pass either `initial` (an existing propagator to continue) or `dim`
    (to start from the dim x dim identity at t0).

    A system that implements its own `.unitary(t)` (an analytic Magnus/RWA
    result defined as a gate) is consumed directly -- no ODE solve, no U -> H
    inversion. In that case `initial` must be omitted or I: composing an
    analytic U(t, t0) with a different starting propagator is the system's
    business, not something to guess here.
    """

    _report_kind = "unitary"

    def __init__(self, system, initial=None, dim: int | None = None,
                 t0: float = 0.0, subsystems: dict | None = None,
                 truncation=None, ladders=None, **solver_kwargs):
        _reject_dissipative(system, t0, "UnitaryEvolution")
        self.system = system
        self.subsystems = _default_subsystems(system, subsystems)
        self._truncation, self._ladders = truncation, ladders
        self._trunc_seen = set()
        self._analytic = provides_unitary(system) and not provides_hamiltonian(system)
        if self._analytic:
            if initial is not None and not np.allclose(np.asarray(initial),
                                                       np.eye(initial.shape[0])):
                raise ValueError("system provides an analytic unitary; a non-identity "
                                 "`initial` propagator can't be composed with it here")
            if t0 != 0:
                # .unitary(t) is U(t, its_own_origin); re-anchoring it at t0 means
                # U(t) U(t0)^dag, which is only the propagator from t0 when H
                # commutes with itself at different times. The system's call.
                raise ValueError(
                    f"system provides an analytic unitary from its own origin, so "
                    f"t0={t0} would be silently ignored. Build the system with the "
                    f"origin you want, or evolve its .hamiltonian(t) instead.")
            self._solver = None
            return
        if initial is None:
            if dim is None:
                raise ValueError("UnitaryEvolution needs either initial or dim")
            initial = np.eye(dim, dtype=complex)  # U(t0) = I
        elif dim is not None and np.shape(initial)[0] != dim:
            raise ValueError(f"UnitaryEvolution got both `initial` (dimension "
                             f"{np.shape(initial)[0]}) and dim={dim}, which disagree; "
                             f"`initial` wins, so drop `dim` or make them match")
        initial = np.asarray(initial)
        _check_hermitian(_h_of(system)(t0))
        self._solver = _QutipSolver(initial, t0, system=system,
                                    label="UnitaryEvolution", expm_ok=True,
                                    **solver_kwargs)

    def unitary_at(self, t) -> np.ndarray:
        """The propagator U(t) (scalar t) or a stack of them (array t)."""
        if self._analytic:
            if np.ndim(t) == 0:
                U = _dense(self.system.unitary(t))
            else:
                U = np.array([_dense(self.system.unitary(tt))
                              for tt in np.asarray(t)])
        else:
            U = self._solver.state_at(t)
        warn_if_truncated(U, self.subsystems, "unitary",
                          resolve_threshold(self._truncation), self._ladders,
                          self._trunc_seen, "UnitaryEvolution")
        return U

    # kept as an alias: every evolution class answers state_at
    state_at = unitary_at

    def unitarity_defect(self, t) -> float:
        """max |U(t)^dag U(t) - I| -- how far numerical error has drifted the
        propagator off the unitary group. A solver-accuracy diagnostic."""
        U = np.asarray(self.unitary_at(t))
        d = U.shape[-1]
        return float(np.max(np.abs(np.swapaxes(U.conj(), -1, -2) @ U - np.eye(d))))


class DensityMatrixEvolution(_Reportable):
    """Closed-system (no dissipation) density matrix evolution:

        rho(t) = U(t) rho0 U(t)^dagger

    Computed by evolving the propagator U(t) (a UnitaryEvolution -- or the
    system's own analytic `.unitary(t)` if it has one) and conjugating on
    demand. Exact up to the accuracy of U itself; rho inherits U's solver
    error twice (U and U^dag), so `unitarity_defect(t)` is exposed as the
    relevant diagnostic. Once a system has dissipation (jump operators)
    the conjugation identity breaks and LindbladEvolution is required --
    passing a dissipative system here raises.
    """

    _report_kind = "density"

    def __init__(self, system, rho0, t0: float = 0.0,
                 subsystems: dict | None = None, truncation=None,
                 ladders=None, **solver_kwargs):
        _reject_dissipative(system, t0, "DensityMatrixEvolution")
        _check_density_matrix(rho0)
        self.system = system
        self.rho0 = np.asarray(rho0)
        dim = self.rho0.shape[0]
        # the inner propagator's own guard is off: what matters physically is
        # the ceiling population of rho, which depends on rho0, and warning
        # about U as well would fire twice for one problem
        self._U = UnitaryEvolution(system, dim=dim, t0=t0,
                                   truncation=False, **solver_kwargs)
        self.subsystems = _default_subsystems(system, subsystems)
        self._truncation, self._ladders = truncation, ladders
        self._trunc_seen = set()

    def state_at(self, t) -> np.ndarray:
        U = self._U.unitary_at(t)
        if U.ndim == 2:
            rho = np.asarray(U @ self.rho0 @ U.conj().T)  # single time: U rho0 U^dagger
        else:
            rho = np.asarray(np.einsum("nij,jk,nlk->nil", U, self.rho0, U.conj()))  # batched over time axis n
        warn_if_truncated(rho, self.subsystems, "density",
                          resolve_threshold(self._truncation), self._ladders,
                          self._trunc_seen, "DensityMatrixEvolution")
        return rho

    def unitarity_defect(self, t) -> float:
        return self._U.unitarity_defect(t)

    def trace_out(self, *names, t) -> np.ndarray:
        """rho at time(s) t, tracing out the named subsystems (batched over t)."""
        from .subsystems import partial_trace
        rho = self.state_at(t)
        return partial_trace(rho, self.subsystems, names)


class LindbladEvolution(_Reportable):
    """Open-system density matrix evolution via the Lindblad master equation:

        d(rho)/dt = -i[H(t), rho] + sum_k ( L_k rho L_k^dagger - 1/2{L_k^dagger L_k, rho} )

    Needed whenever a system has jump operators -- dissipation into a bath
    too large/uncharacterized to model as a subsystem. Genuinely different
    from HamiltonianEvolution/UnitaryEvolution's dX/dt = -iHX: not obtainable
    via conjugation by a propagator the way DensityMatrixEvolution is.

    Forward-only: `state_at(t)` for t < t0 is rejected. Trace preservation
    holds integrating either direction, but positivity is only guaranteed by
    the forward semigroup -- backward integration can (and does) produce a
    matrix with negative eigenvalues, i.e. not a valid density matrix.
    """

    _report_kind = "density"

    def __init__(self, system, rho0, t0: float = 0.0,
                 subsystems: dict | None = None, truncation=None,
                 ladders=None, **solver_kwargs):
        _check_density_matrix(rho0)
        _check_hermitian(_h_of(system)(t0))
        self.system = system
        self.rho0 = np.asarray(rho0)
        self.t0 = t0
        self._solver = _QutipSolver(self.rho0, t0, system=system,
                                    label="LindbladEvolution", expm_ok=False,
                                    dissipative=True, **solver_kwargs)
        self.subsystems = _default_subsystems(system, subsystems)
        self._truncation, self._ladders = truncation, ladders
        self._trunc_seen = set()

    def state_at(self, t) -> np.ndarray:
        if np.any(np.asarray(t) < self.t0):
            raise ValueError(
                f"LindbladEvolution.state_at: requested t < t0 ({self.t0}). Backward "
                "integration of a Lindbladian isn't guaranteed positive -- would "
                "silently return a non-physical density matrix. Not supported."
            )
        rho = self._solver.state_at(t)
        # heating is the canonical way to hit the ceiling: a mode driven by a
        # thermal bath climbs the ladder indefinitely, so n_max chosen from the
        # coherent dynamics alone is routinely too small here
        warn_if_truncated(rho, self.subsystems, "density",
                          resolve_threshold(self._truncation), self._ladders,
                          self._trunc_seen, "LindbladEvolution")
        return rho

    def trace_out(self, *names, t) -> np.ndarray:
        """rho at time(s) t, tracing out the named subsystems (batched over t)."""
        from .subsystems import partial_trace
        rho = self.state_at(t)
        return partial_trace(rho, self.subsystems, names)


# ---------------------------------------------------------------------------
# facade -- the one-call shortcut, for when you just want the answer
# ---------------------------------------------------------------------------

def evolve(system, initial, t, t0: float = 0.0, **kwargs):
    """Solve the equation of motion matching `initial`, and hand back
    `state_at(t)` directly -- no Evolution object to construct or query.

    `initial` a ket (1-D) -> HamiltonianEvolution; a density matrix (2-D) ->
    DensityMatrixEvolution, or LindbladEvolution if `system` carries jump
    operators (dissipation needs a mixed state -- a ket with a dissipative
    system raises rather than silently ignoring the dissipation, same rule
    the classes themselves enforce).

    Extra keyword arguments (`rtol=`, `atol=`, `method=`, `verbose=`,
    `subsystems=`, `truncation=`, ...) pass straight through to whichever
    class gets picked. This is the shortcut for a one-shot number; reach for
    the class directly when you need `report()`, `trace_out()`, or repeated
    queries without re-solving.
    """
    arr = np.asarray(initial)
    if arr.ndim == 1:
        if _is_dissipative(system, t0):
            raise ValueError(
                "evolve(): system has jump operators, but `initial` is a ket -- "
                "dissipation needs a mixed state. Pass a density matrix (e.g. "
                "np.outer(psi, psi.conj())) so this routes to LindbladEvolution.")
        ev = HamiltonianEvolution(system, initial, t0=t0, **kwargs)
    elif arr.ndim == 2:
        cls = LindbladEvolution if _is_dissipative(system, t0) else DensityMatrixEvolution
        ev = cls(system, initial, t0=t0, **kwargs)
    else:
        raise ValueError(f"evolve() needs a ket (d,) or density matrix (d,d), "
                         f"got shape {arr.shape}")
    return ev.state_at(t)


def propagator(system, dim, t, t0: float = 0.0, **kwargs):
    """`UnitaryEvolution(system, dim=dim, t0=t0, **kwargs).unitary_at(t)` in
    one call -- the propagator shortcut alongside `evolve()`. `dim` is
    unused (and may be omitted) when `system` provides its own analytic
    `.unitary(t)`, exactly as for `UnitaryEvolution` itself."""
    return UnitaryEvolution(system, dim=dim, t0=t0, **kwargs).unitary_at(t)
