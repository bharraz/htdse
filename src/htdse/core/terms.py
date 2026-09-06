"""The composable System layer.

A `System` here is NOT a matrix -- it is a sum of symbolic *terms*, each
term being

    coefficient (a number, or a callable f(t))  x  local operators on NAMED
    subsystems (e.g. {"spin": sigma_plus, "mode": a})

plus a registry {subsystem name: dimension}. The dense matrix on the joint
space is a *materialization* computed only when an evolution asks for
`.hamiltonian(t)` -- which makes a `System` directly consumable by evolution.

Because terms carry subsystem *names*, composition is literal:

    H_atom = term(0.5 * w0 * sigma_z, on="spin")  # term factory function (bottom of file)
    H_mode = term(w * number_op, on="mode")
    H_jc   = plus_hc(term({"spin": sigma_plus, "mode": a}, coeff=g))
    H      = H_atom + H_mode + H_jc          # names do the embedding

`+` takes the union of the registries (same name must mean same dimension;
matching is by name only, never by size) and merges the named term groups.
The spin term stays 2-dim in its definition, the mode term stays
(n_max+1)-dim; the joint matrix only exists at solve time.

Named groups are the swap-out handle:

    model    = atom + mode + term(..., name="drive")
    realized = replace(system, drive=noisy_drive)   # same system, one entry swapped

Storage is a backend detail, not a type: `H.sparse()` returns the same System
flagged to materialize as scipy CSR, which QuTiP preserves internally. You
never handle a CSR yourself --
`hamiltonian(t)` and `jump_operators(t)` always hand back plain numpy arrays;
only the solver sees the native storage. The flag is sticky under composition.

Worth it from a joint dimension of a few hundred up (measured crossover ~200;
5x faster at 256, 124x at 1024). Below that, dense wins on fixed overhead.
Nothing switches by itself -- a System past the threshold raises a one-time
`SparseSuggestion` warning and leaves the decision to you.

Physics caveats the framework cannot check for you:
- Addition is literal. All terms must be written in the same frame (lab vs.
  rotating); terms may carry a `frame` tag and mixing distinct tags warns.
- Composing a dipole coupling g*sx(x)(a+adag) gives the quantum RABI model;
  Jaynes-Cummings is the post-RWA interaction s+a + s-adag, which you write
  directly as its own term. The framework composes; you choose the
  approximation.
"""
import itertools
import warnings
from types import MappingProxyType
from typing import Callable, Union

import numpy as np
from scipy import sparse as _sp

from .subsystems import embed

_anon_counter = itertools.count()  # unique keys for unnamed term groups

Coefficient = Union[complex, float, Callable[[float], complex]]

# A dense copy costs 16 bytes per entry, so at the dimensions where sparse
# storage earns its keep it is not merely slow but impossible. Refuse with the
# number rather than let numpy raise MemoryError three frames deeper.
MAX_DENSE_BYTES = 2 * 1024 ** 3  # 2 GiB

# When to SUGGEST sparse. A dense H(t) rebuild copies the whole dim^2 matrix
# every call no matter how empty it is, while sparse tracks nnz -- so the
# crossover is set by dimension, not fill. Measured at ~200 (below it sparse
# loses on fixed overhead, ~3.7x slower at dim 128; above it sparse wins by
# 5x at 256 and 124x at 1024). Fill only matters as a veto: an actually-dense
# matrix gains nothing.
SPARSE_HINT_DIM = 256
SPARSE_HINT_MAX_FILL = 0.5


class SparseSuggestion(UserWarning):
    """Raised once by a dense System large enough that `.sparse()` would pay.
    Performance advice, never a correctness problem -- silence with
    `warnings.simplefilter("ignore", SparseSuggestion)`."""


def _densify(M, dim):
    """A possibly-sparse operator -> a plain ndarray, or a clear refusal."""
    if not _sp.issparse(M):
        return np.asarray(M)
    need = dim * dim * 16
    if need > MAX_DENSE_BYTES:
        raise MemoryError(
            f"this operator is {dim}x{dim}, which is {need / 1024 ** 3:.1f} GB as a "
            f"dense array (limit {MAX_DENSE_BYTES / 1024 ** 3:.0f} GB). Nothing is "
            f"wrong with the model -- the solver keeps it sparse and evolves it "
            f"fine. It is materializing the whole matrix in one piece that does "
            f"not fit, so evolve it rather than inspecting H(t) directly.")
    return M.toarray()


def _readonly_array(value):
    """Own an operator array and make the owned copy immutable."""
    out = np.array(value, dtype=complex, copy=True)
    out.setflags(write=False)
    return out


class _Term:
    """One product term: coeff (scalar or f(t)) x local ops on named subsystems.

    `ops` maps a subsystem name (str) -- or a tuple of names, for a joint
    operator that doesn't factor -- to a matrix. `dims` records the dimension
    of every subsystem the term touches.

    `frame`: an optional free-text tag (e.g. "lab", "rotating@w0"), carrying no
    physics itself -- `+` is literal matrix addition, so it can't detect two
    terms written under different frame assumptions being combined into
    something meaningless. Mixing distinct tags in one System warns at
    materialization; this is the only thing `frame` does.
    """

    def __init__(self, coeff: Coefficient, ops: dict, dims: dict, frame=None):
        object.__setattr__(self, "coeff", coeff)
        object.__setattr__(self, "ops", MappingProxyType({
            k if isinstance(k, tuple) else (k,): _readonly_array(v)
            for k, v in ops.items()}))
        object.__setattr__(self, "dims", MappingProxyType(dict(dims)))
        object.__setattr__(self, "frame", frame)
        for key, mat in self.ops.items():
            d = int(np.prod([self.dims[n] for n in key]))
            if mat.shape != (d, d):
                raise ValueError(f"operator on {key} has shape {mat.shape}, "
                                 f"expected ({d}, {d}) from dims {self.dims}")

    def __setattr__(self, name, value):
        raise AttributeError("contribution records are immutable")

    @property
    def is_static(self) -> bool:
        return not callable(self.coeff)

    def coeff_at(self, t) -> complex:
        return self.coeff(t) if callable(self.coeff) else self.coeff

    def involved(self) -> tuple:
        return tuple(n for key in self.ops for n in key)

    def local_matrix(self, sparse: bool = False):
        """Kronecker product of this term's ops, in their stated order --
        the operator on just the subsystems the term touches. `sparse=True`
        returns a scipy CSR matrix (kron chain stays sparse throughout)."""
        mats = list(self.ops.values())
        if sparse:
            out = _sp.csr_matrix(mats[0])
            for m in mats[1:]:
                out = _sp.kron(out, _sp.csr_matrix(m), format="csr")
            return out
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def scaled(self, c: Coefficient) -> "_Term":
        if callable(c) and callable(self.coeff):
            f, g = c, self.coeff
            coeff = lambda t: f(t) * g(t)
        elif callable(c):
            g0 = self.coeff
            coeff = lambda t: c(t) * g0
        elif callable(self.coeff):
            g1 = self.coeff
            coeff = lambda t: c * g1(t)
        else:
            coeff = c * self.coeff
        return _Term(coeff, dict(self.ops), self.dims, self.frame)

    def dag(self) -> "_Term":
        """Hermitian conjugate: (A (x) B)^dag = A^dag (x) B^dag, coeff conjugated."""
        ops = {k: m.conj().T for k, m in self.ops.items()}
        if callable(self.coeff):
            f = self.coeff
            coeff = lambda t: np.conj(f(t))
        else:
            coeff = np.conj(self.coeff)
        return _Term(coeff, ops, self.dims, self.frame)


class _OperatorTerm:
    """Private named contribution whose operator is evaluated as ``operator(t)``."""

    def __init__(self, operator: Callable[[float], np.ndarray], on, dims, frame=None):
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "on", (on,) if isinstance(on, str) else tuple(on))
        object.__setattr__(self, "dims", MappingProxyType(dict(dims)))
        object.__setattr__(self, "frame", frame)

    def __setattr__(self, name, value):
        raise AttributeError("contribution records are immutable")

    def at(self, t, sparse=False, target_dims=None):
        value = self.operator(t)
        shape = value.shape if _sp.issparse(value) else np.shape(value)
        expected = int(np.prod([self.dims[n] for n in self.on]))
        if shape != (expected, expected):
            raise ValueError(f"operator-valued contribution on {self.on} has shape "
                             f"{shape}, expected ({expected}, {expected})")
        if _sp.issparse(value) and not sparse:
            value = value.toarray()
        elif not _sp.issparse(value):
            value = _readonly_array(value)
        embedded = embed(value, target_dims or self.dims, self.on)
        if sparse and not _sp.issparse(embedded):
            embedded = _sp.csr_matrix(embedded)
        return embedded if sparse else np.asarray(embedded)

    def scaled(self, c):
        fn = self.operator
        factor = c if callable(c) else lambda t: c
        return _OperatorTerm(lambda t: factor(t) * fn(t), self.on, self.dims, self.frame)

    def dag(self):
        fn = self.operator
        def conjugate_transpose(t):
            value = fn(t)
            return value.conj().T if _sp.issparse(value) else np.asarray(value).conj().T
        return _OperatorTerm(conjugate_transpose,
                             self.on, self.dims, self.frame)


def _merge_registry(a: dict, b: dict) -> dict:
    """Union of two {name: dim} registries, first-appearance order.
    Same name MUST mean same dimension -- matching is by name only, never by
    size, so a collision is a modeling error and raises."""
    out = dict(a)
    for name, dim in b.items():
        if name in out and out[name] != dim:
            raise ValueError(f"subsystem {name!r} has dimension {out[name]} on one "
                             f"side and {dim} on the other -- same name must mean "
                             f"the same physical subsystem")
        out.setdefault(name, dim)
    return out


class System:
    """A sum of named groups of terms + a subsystem registry. Satisfies the
    dynamics value consumed by the evolution classes.

    Treat instances as immutable: every operation (+, *, dag, replace, ...)
    returns a new System. See the module docstring for the full system.
    """

    def __init__(self, subsystems: dict | None = None, groups: dict | None = None,
                 jumps: dict | None = None, sparse: bool = False,
                 breakpoints=()):
        object.__setattr__(self, "subsystems", MappingProxyType(dict(subsystems or {})))
        object.__setattr__(self, "groups", MappingProxyType(
            {k: tuple(v) for k, v in (groups or {}).items()}))
        object.__setattr__(self, "jumps", MappingProxyType(
            {k: tuple(v) for k, v in (jumps or {}).items()}))
        object.__setattr__(self, "is_sparse", bool(sparse))
        object.__setattr__(self, "_breakpoints", tuple(sorted(set(float(t) for t in breakpoints))))
        object.__setattr__(self, "_cache", None)
        object.__setattr__(self, "_hinted", False)
        object.__setattr__(self, "_immutable", True)

    def __setattr__(self, name, value):
        if name.startswith("_") and name in {"_cache", "_hinted"}:
            object.__setattr__(self, name, value)
            return
        raise AttributeError("System values are immutable; build a transformed System")

    def H(self, t):
        """Paper-style alias for the Hamiltonian accessor."""
        return self.hamiltonian(t)

    def breakpoints(self):
        """Discontinuities at which an evolution must restart integration."""
        return np.asarray(self._breakpoints, dtype=float)

    # ---- composition ----------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self  # so sum([...]) works
        if not isinstance(other, System):
            return NotImplemented
        subsystems = _merge_registry(self.subsystems, other.subsystems)
        groups = {k: tuple(v) for k, v in self.groups.items()}
        for k, terms in other.groups.items():
            groups.setdefault(k, ())
            groups[k] = groups[k] + tuple(terms)
        jumps = {k: tuple(v) for k, v in self.jumps.items()}
        for k, terms in other.jumps.items():
            jumps.setdefault(k, ())
            jumps[k] = jumps[k] + tuple(terms)
        # sparse is sticky under composition: either side sparse => sum sparse
        return System(subsystems, groups, jumps, sparse=self.is_sparse or other.is_sparse,
                      breakpoints=self._breakpoints + other._breakpoints)

    __radd__ = __add__

    def sparse(self, flag: bool = True) -> "System":
        """Return this System flagged to materialize as scipy sparse (CSR).

        Same physics, different storage: every embedded term matrix and the
        static sum become CSR, and the evolution classes use sparse
        operators through QuTiP. This does NOT change what you get back --
        `hamiltonian(t)` and `jump_operators(t)` still return plain numpy
        arrays either way; the CSR stays on the solver side.

        Worth it from a joint dimension of a few hundred up: the measured
        crossover is ~200 (dense H(t) recopies all dim^2 entries per call
        regardless of how empty it is, while sparse tracks nnz), giving 5x at
        dim 256 and 124x at 1024. Below ~200, dense wins on fixed overhead.
        A dense System past the threshold says so once; it never switches
        itself.

        The flag is sticky under composition: `H.sparse() + other` is sparse.
        `H.sparse(False)` (or on any composition of sparse models) toggles back
        to dense."""
        return System(self.subsystems, self.groups, self.jumps, sparse=flag,
                      breakpoints=self._breakpoints)

    def _reject_jumps(self, op: str):
        """Scaling/negating/subtracting a DISSIPATIVE model has no agreed
        meaning: L_k enters the GKSL equation quadratically, so `2*H` would
        scale coherent terms while leaving rates alone, and `-H` (or `H1 - H2`,
        which is `H1 + H2*(-1)`) would carry the jumps through unscaled and
        merge them in -- silently doubling every rate on `h - h`. Rather than
        pick a convention, refuse. Strip the channels explicitly first."""
        if self.jumps:
            raise ValueError(
                f"cannot {op} a System carrying jump operators "
                f"{sorted(self.jumps)}: dissipation does not scale with the "
                f"coherent part, and negation/subtraction would merge the "
                f"channels in unscaled. Drop them first with "
                f".without({', '.join(repr(k) for k in sorted(self.jumps))}).")

    def __mul__(self, c: Coefficient):
        """Scale every HAMILTONIAN term's coefficient by a scalar or f(t).
        Refuses a System carrying jump operators (see `_reject_jumps`)."""
        self._reject_jumps("scale")
        groups = {k: tuple(term.scaled(c) for term in v) for k, v in self.groups.items()}
        return System(self.subsystems, groups, self.jumps, sparse=self.is_sparse,
                      breakpoints=self._breakpoints)

    __rmul__ = __mul__

    def __neg__(self):
        self._reject_jumps("negate")
        return self * (-1.0)

    def __sub__(self, other):
        if not isinstance(other, System):
            return NotImplemented
        self._reject_jumps("subtract from")
        other._reject_jumps("subtract")
        return self + (other * (-1.0))

    def dag(self) -> "System":
        """Hermitian conjugate of every Hamiltonian term: (A x B)^dag with the
        coefficient conjugated, groups keeping their names. This is the `h.c.`
        of a paper Hamiltonian -- `H_int + H_int.dag()` completes a coupling
        written one-way (or use `plus_hc(H_int)` for the same thing in one call).

        Caveat: jump operators are DROPPED, not conjugated -- L^dag is a
        physically different channel, and carrying jumps through `h + h.dag()`
        would silently double every dissipation rate."""
        groups = {k: tuple(term.dag() for term in v) for k, v in self.groups.items()}
        return System(self.subsystems, groups, sparse=self.is_sparse,
                      breakpoints=self._breakpoints)

    def replace(self, **named) -> "System":
        """Swap out named term groups wholesale: the composable-error workflow.

            realized = replace(system, drive=noisy_drive)

        Each value is a System; ALL its terms (and jumps, and any new
        subsystems it introduces) land under the replaced name. The group must
        already exist -- replacing an unknown name is almost always a typo, so
        it raises. To add a group, use `+`; to delete one, use `without()`."""
        subsystems = dict(self.subsystems)
        groups = {k: tuple(v) for k, v in self.groups.items()}
        jumps = {k: tuple(v) for k, v in self.jumps.items()}
        for name, replacement in named.items():
            if name not in groups and name not in jumps:
                raise KeyError(f"no term group named {name!r}; have "
                               f"{sorted(set(groups) | set(jumps))}")
            if not isinstance(replacement, System):
                raise TypeError(f"replacement for {name!r} must be a System")
            subsystems = _merge_registry(subsystems, replacement.subsystems)
            groups[name] = tuple(t for terms in replacement.groups.values() for t in terms)
            if not groups[name]:
                del groups[name]
            new_jumps = tuple(t for terms in replacement.jumps.values() for t in terms)
            if new_jumps:
                jumps[name] = new_jumps
            elif name in jumps:
                del jumps[name]
        return System(subsystems, groups, jumps, sparse=self.is_sparse,
                      breakpoints=self._breakpoints + tuple(
                          bp for replacement in named.values()
                          for bp in getattr(replacement, "_breakpoints", ())))

    def without(self, *names) -> "System":
        """Drop named term groups (from both H terms and jumps)."""
        for name in names:
            if name not in self.groups and name not in self.jumps:
                raise KeyError(f"no term group named {name!r}")
        groups = {k: v for k, v in self.groups.items() if k not in names}
        jumps = {k: v for k, v in self.jumps.items() if k not in names}
        return System(self.subsystems, groups, jumps, sparse=self.is_sparse,
                      breakpoints=self._breakpoints)

    def group(self, name) -> "System":
        """Extract one named group as its own System (same registry)."""
        groups, jumps = {}, {}
        if name in self.groups:
            groups[name] = self.groups[name]
        if name in self.jumps:
            jumps[name] = self.jumps[name]
        if not groups and not jumps:
            raise KeyError(f"no term group named {name!r}")
        return System(self.subsystems, groups, jumps, sparse=self.is_sparse,
                      breakpoints=self._breakpoints)

    # ---- materialization (the System protocol) --------------------------

    @property
    def dim(self) -> int:
        d = 1
        for v in self.subsystems.values():
            d *= v
        return d

    def _embed(self, term: _Term):
        """Embed one term into the joint space: dense ndarray, or CSR when
        this System is flagged sparse (embed() stays sparse throughout)."""
        if self.is_sparse:
            return embed(term.local_matrix(sparse=True), self.subsystems,
                         term.involved())
        return np.asarray(embed(term.local_matrix(), self.subsystems, term.involved()))

    def _materialize(self):
        """Embed every term once (embedding is time-independent), sum the
        static ones, and keep (coeff_fn, matrix) for the time-dependent ones."""
        if self._cache is not None:
            return self._cache
        frames = {t.frame for terms in list(self.groups.values()) + list(self.jumps.values())
                  for t in terms if t.frame is not None}
        if len(frames) > 1:
            warnings.warn(f"composing terms tagged with different frames {sorted(frames)} "
                          "-- literal addition of Systems written in different "
                          "frames is not physically meaningful", stacklevel=3)
        if self.is_sparse:
            static = _sp.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            static = np.zeros((self.dim, self.dim), dtype=complex)
        dynamic = []
        for terms in self.groups.values():
            for term in terms:
                if isinstance(term, _OperatorTerm):
                    dynamic.append(("operator", term))
                    continue
                mat = self._embed(term)
                if term.is_static:
                    static = static + term.coeff_at(0.0) * mat
                else:
                    dynamic.append(("coefficient", term.coeff, mat))
        jump_static = []
        jump_dynamic = []
        for terms in self.jumps.values():
            for term in terms:
                if isinstance(term, _OperatorTerm):
                    jump_dynamic.append(("operator", term))
                    continue
                mat = self._embed(term)
                if term.is_static:
                    jump_static.append(term.coeff_at(0.0) * mat)
                else:
                    jump_dynamic.append(("coefficient", term.coeff, mat))
        self._cache = (static, dynamic, jump_static, jump_dynamic)
        self._suggest_sparse(static, dynamic)
        return self._cache

    def _suggest_sparse(self, static, dynamic):
        """Warn once if this dense System is big enough that `.sparse()` would
        pay. Advice only -- nothing switches by itself.

        A warning rather than a print so it carries a source line (pointing at
        the code that built the System) and obeys the usual warning filters.
        `quiet()` does NOT silence it: it is about your model, not solver
        chatter. Use `warnings.simplefilter("ignore", SparseSuggestion)`."""
        if self.is_sparse or self._hinted or self.dim < SPARSE_HINT_DIM:
            return
        if any(item[0] == "operator" for item in dynamic):
            self._hinted = True
            return
        # The union pattern over static + every dynamic piece. Time-independent:
        # a coefficient f(t) scales a piece, it can never create a nonzero where
        # that piece's matrix has a structural zero. Computed once, at most.
        pattern = np.abs(static)
        for item in dynamic:
            if item[0] == "coefficient":
                pattern = pattern + np.abs(item[2])
        fill = np.count_nonzero(pattern) / float(self.dim * self.dim)
        self._hinted = True
        if fill > SPARSE_HINT_MAX_FILL:
            return
        warnings.warn(
            f"this System is {self.dim}-dim and {100 * fill:.1f}% filled -- "
            f".sparse() would cut H(t) rebuild time substantially (dense "
            f"recopies all {self.dim * self.dim:,} entries per call). Storage "
            f"stays internal either way; hamiltonian(t) returns a plain array.",
            SparseSuggestion, stacklevel=4)

    def _h_native(self, t):
        """H(t) in this System's native storage -- CSR when sparse-flagged.

        The SOLVER's accessor. Sparse storage is a backend decision: keeping it
        sparse here is the whole point of the flag, and the evolution classes
        consume CSR directly (csr @ dense -> dense). Users get `hamiltonian(t)`,
        which is always a plain array."""
        # `static` is the memoized sum; never hand it out or accumulate into it,
        # or a caller's in-place edit of H(t) would silently corrupt the cache.
        # (Stacking the dynamic terms into one (K,d,d) contraction was measured
        # SLOWER than this loop at realistic sizes -- the cost is in evaluating
        # the K coefficient callables, not in the matrix algebra.)
        static, dynamic, _, _ = self._materialize()
        H = static.copy()
        if self.is_sparse:
            for item in dynamic:
                if item[0] == "operator":
                    H = H + item[1].at(t, sparse=True, target_dims=self.subsystems)
                else:
                    _, coeff, mat = item
                    H = H + coeff(t) * mat  # csr addition allocates; no in-place form
            return H
        for item in dynamic:
            if item[0] == "operator":
                H += item[1].at(t, sparse=False, target_dims=self.subsystems)
            else:
                _, coeff, mat = item
                H += coeff(t) * mat  # in-place into the copy of `static`
        return np.asarray(H)

    def _jumps_native(self, t) -> list:
        """Jump operators at time t in native storage. Solver-facing; see
        `_h_native`."""
        _, _, jump_static, jump_dynamic = self._materialize()
        if self.is_sparse:
            out = [L.copy() for L in jump_static]
            for item in jump_dynamic:
                out.append(item[1].at(t, sparse=True, target_dims=self.subsystems)
                           if item[0] == "operator"
                            else item[2] * item[1](t))
            return out
        out = [np.asarray(L) for L in jump_static]
        for item in jump_dynamic:
            out.append(item[1].at(t, sparse=False, target_dims=self.subsystems)
                       if item[0] == "operator"
                       else np.asarray(item[1](t) * item[2]))
        return out

    def hamiltonian(self, t):
        """H(t) as a plain numpy array -- ALWAYS, sparse-flagged or not.

        Storage is a backend concern: you asked to see the matrix, so you get
        one you can plot, index and compare without knowing what CSR is. The
        solver takes the sparse path regardless (see `_h_native`)."""
        return _densify(self._h_native(t), self.dim)

    def jump_operators(self, t) -> list:
        """Jump operators L_k(t) as plain numpy arrays (see `hamiltonian`)."""
        return [_densify(L, self.dim) for L in self._jumps_native(t)]

    # ---- inspection ------------------------------------------------------

    def __repr__(self):
        subs = ", ".join(f"{n}:{d}" for n, d in self.subsystems.items())
        gs = ", ".join(f"{k}[{len(v)}]" for k, v in self.groups.items())
        js = ", ".join(f"{k}[{len(v)}]" for k, v in self.jumps.items())
        parts = [f"subsystems=({subs})", f"terms=({gs})"]
        if js:
            parts.append(f"jumps=({js})")
        if self.is_sparse:
            parts.append("sparse")
        return f"System({', '.join(parts)})"


def _build_ops_and_dims(op, on, dims):
    """Normalize the flexible `term()` argument forms into (ops, dims)."""
    if isinstance(op, dict):
        ops = op
    else:
        if on is None:
            raise ValueError("a bare matrix needs `on=` naming its subsystem(s)")
        ops = {on if isinstance(on, (str, tuple)) else tuple(on): op}
    out_dims = {}
    for key, mat in ops.items():
        mat = np.asarray(mat)
        names = (key,) if isinstance(key, str) else tuple(key)
        if len(names) == 1:
            out_dims[names[0]] = mat.shape[0]  # single factor: dim from the matrix
        else:
            # joint (non-factoring) operator: the split of its dimension across
            # the named subsystems is ambiguous, so it must be given explicitly
            if dims is None or any(n not in dims for n in names):
                raise ValueError(f"joint operator on {names} needs explicit "
                                 f"dims={{name: dim}} for those subsystems")
            for n in names:
                out_dims[n] = dims[n]
    return ops, out_dims


def term(op, on=None, coeff: Coefficient = 1.0, name: str | None = None,
         frame: str | None = None, dims: dict | None = None) -> System:
    """Build a one-term System -- the atom everything composes from.

    op:    a matrix (with `on=` naming its subsystem), or a dict
           {name: matrix} for a product across several subsystems
           (e.g. {"spin": sigma_plus, "mode": a}), or {(n1, n2): matrix} for
           a joint operator that doesn't factor (needs `dims=`).
    coeff: scalar, or callable f(t) for time-dependent control.
    name:  the term-group name -- the handle `replace()` swaps by. Unnamed
           terms get a unique auto-name (composable, but not swappable).
    frame: optional tag ("lab", "rotating@w0", ...); mixing distinct tags in
           one System warns at materialization.
    """
    ops, term_dims = _build_ops_and_dims(op, on, dims)
    key = name if name is not None else f"term{next(_anon_counter)}"
    t = _Term(coeff, ops, term_dims, frame)
    return System(term_dims, groups={key: [t]})


def _operator(operator, on, dims, name=None, sparse=False, frame=None) -> System:
    """Construct a System from a private operator-valued contribution."""
    key = name if name is not None else f"term{next(_anon_counter)}"
    contribution = _OperatorTerm(operator, on, dims, frame)
    return System(dims, groups={key: (contribution,)}, sparse=sparse)


def jump(op, on=None, coeff: Coefficient = 1.0, name: str | None = None,
         dims: dict | None = None) -> System:
    """Build a System carrying one Lindblad jump operator (and no
    coherent term). The materialized L is coeff * (embedded op) -- keep the
    sqrt(rate) convention: pass coeff=np.sqrt(gamma).

    Composes with `+` exactly like coherent terms, so a dissipative component
    is just another named group you can `replace()` or `without()`."""
    ops, term_dims = _build_ops_and_dims(op, on, dims)
    key = name if name is not None else f"jump{next(_anon_counter)}"
    t = _Term(coeff, ops, term_dims, None)
    return System(term_dims, jumps={key: [t]})


def plus_hc(h: System) -> System:
    """h + h.dag() -- the ubiquitous `X + h.c.` pattern in one call.

    Any jump operators on `h` ride through exactly once (see `dag`): only the
    coherent terms are conjugated and added."""
    return h + h.dag()


def hc(h: System) -> System:
    """JUST the Hermitian conjugate -- `h.dag()` as a free function, so it
    sits next to `plus_hc` instead of being the one operation on this page
    you have to reach for a method to get. `h + hc(h)` == `plus_hc(h)`."""
    return h.dag()


def replace(system: System, **named) -> System:
    """Return ``system`` with named contribution groups replaced."""
    return system.replace(**named)


def without(system: System, *names) -> System:
    """Return ``system`` without the named contribution groups."""
    return system.without(*names)


def group(system: System, name) -> System:
    """Return one named contribution group as a new System."""
    return system.group(name)
