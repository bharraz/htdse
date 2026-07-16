"""The composable Model layer.

A `Model` here is NOT a matrix -- it is a sum of symbolic *terms*, each
term being

    coefficient (a number, or a callable f(t))  x  local operators on NAMED
    subsystems (e.g. {"spin": sigma_plus, "mode": a})

plus a registry {subsystem name: dimension}. The dense matrix on the joint
space is a *materialization* computed only when an evolution asks for
`.hamiltonian(t)` -- which makes a `Model` a drop-in `Mechanism`.

Because terms carry subsystem *names*, composition is literal:

    H_atom = term(0.5 * w0 * sigma_z, on="spin")
    H_mode = term(w * number_op, on="mode")
    H_jc   = hconj(term({"spin": sigma_plus, "mode": a}, coeff=g))
    H      = H_atom + H_mode + H_jc          # names do the embedding

`+` takes the union of the registries (same name must mean same dimension;
matching is by name only, never by size) and merges the named term groups.
The spin term stays 2-dim in its definition, the mode term stays
(n_max+1)-dim; the joint matrix only exists at solve time.

Named groups are the swap-out handle:

    model    = atom + mode + term(..., name="drive")
    realized = model.replace(drive=noisy_drive)   # same model, one entry swapped

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
from typing import Callable, Union

import numpy as np

from .mechanism import Mechanism
from .operator import Operator
from .subsystems import embed

_anon_counter = itertools.count()  # unique keys for unnamed term groups

Coefficient = Union[complex, float, Callable[[float], complex]]


class Term:
    """One product term: coeff (scalar or f(t)) x local ops on named subsystems.

    `ops` maps a subsystem name (str) -- or a tuple of names, for a joint
    operator that doesn't factor -- to a matrix. `dims` records the dimension
    of every subsystem the term touches.
    """

    def __init__(self, coeff: Coefficient, ops: dict, dims: dict, frame=None):
        self.coeff = coeff
        self.ops = {k if isinstance(k, tuple) else (k,): np.asarray(v, dtype=complex)
                    for k, v in ops.items()}
        self.dims = dict(dims)
        self.frame = frame
        for key, mat in self.ops.items():
            d = int(np.prod([self.dims[n] for n in key]))
            if mat.shape != (d, d):
                raise ValueError(f"operator on {key} has shape {mat.shape}, "
                                 f"expected ({d}, {d}) from dims {self.dims}")

    @property
    def is_static(self) -> bool:
        return not callable(self.coeff)

    def coeff_at(self, t) -> complex:
        return self.coeff(t) if callable(self.coeff) else self.coeff

    def involved(self) -> tuple:
        return tuple(n for key in self.ops for n in key)

    def local_matrix(self) -> np.ndarray:
        """Kronecker product of this term's ops, in their stated order --
        the operator on just the subsystems the term touches."""
        mats = list(self.ops.values())
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def scaled(self, c: Coefficient) -> "Term":
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
        return Term(coeff, dict(self.ops), self.dims, self.frame)

    def dag(self) -> "Term":
        """Hermitian conjugate: (A (x) B)^dag = A^dag (x) B^dag, coeff conjugated."""
        ops = {k: m.conj().T for k, m in self.ops.items()}
        if callable(self.coeff):
            f = self.coeff
            coeff = lambda t: np.conj(f(t))
        else:
            coeff = np.conj(self.coeff)
        return Term(coeff, ops, self.dims, self.frame)


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


class Model(Mechanism):
    """A sum of named groups of Terms + a subsystem registry. Satisfies the
    Mechanism protocol (`hamiltonian(t)`, `jump_operators(t)`), so it plugs
    straight into any evolution class.

    Treat instances as immutable: every operation (+, *, dag, replace, ...)
    returns a new Model. See the module docstring for the full model.
    """

    def __init__(self, subsystems: dict | None = None, groups: dict | None = None,
                 jumps: dict | None = None):
        self.subsystems = dict(subsystems or {})   # {name: dim}, canonical order
        self.groups = {k: list(v) for k, v in (groups or {}).items()}  # H terms
        self.jumps = {k: list(v) for k, v in (jumps or {}).items()}    # Lindblad terms
        self._cache = None      # built lazily by _materialize()
        self._cache_key = None  # structure _cache was built from

    # ---- composition ----------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self  # so sum([...]) works
        if not isinstance(other, Model):
            return NotImplemented
        subsystems = _merge_registry(self.subsystems, other.subsystems)
        groups = {k: list(v) for k, v in self.groups.items()}
        for k, terms in other.groups.items():
            groups.setdefault(k, [])
            groups[k] = groups[k] + list(terms)
        jumps = {k: list(v) for k, v in self.jumps.items()}
        for k, terms in other.jumps.items():
            jumps.setdefault(k, [])
            jumps[k] = jumps[k] + list(terms)
        return Model(subsystems, groups, jumps)

    __radd__ = __add__

    def _reject_jumps(self, op: str):
        """Scaling/negating/subtracting a DISSIPATIVE model has no agreed
        meaning: L_k enters the GKSL equation quadratically, so `2*H` would
        scale coherent terms while leaving rates alone, and `-H` (or `H1 - H2`,
        which is `H1 + H2*(-1)`) would carry the jumps through unscaled and
        merge them in -- silently doubling every rate on `h - h`. Rather than
        pick a convention, refuse. Strip the channels explicitly first."""
        if self.jumps:
            raise ValueError(
                f"cannot {op} a Model carrying jump operators "
                f"{sorted(self.jumps)}: dissipation does not scale with the "
                f"coherent part, and negation/subtraction would merge the "
                f"channels in unscaled. Drop them first with "
                f".without({', '.join(repr(k) for k in sorted(self.jumps))}).")

    def __mul__(self, c: Coefficient):
        """Scale every HAMILTONIAN term's coefficient by a scalar or f(t).
        Refuses a Model carrying jump operators (see `_reject_jumps`)."""
        self._reject_jumps("scale")
        groups = {k: [term.scaled(c) for term in v] for k, v in self.groups.items()}
        return Model(self.subsystems, groups, self.jumps)

    __rmul__ = __mul__

    def __neg__(self):
        self._reject_jumps("negate")
        return self * (-1.0)

    def __sub__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        self._reject_jumps("subtract from")
        other._reject_jumps("subtract")
        return self + (other * (-1.0))

    def dag(self) -> "Model":
        """Hermitian conjugate of every Hamiltonian term (groups keep their
        names). Handy for writing `H_int + H_int.dag()` for `... + h.c.`

        Jump operators are DROPPED, not conjugated: L^dag is a different
        channel, and `h + h.dag()` is meant to complete a coherent term. Were
        the jumps carried through, `hconj(coherent + jump(...))` would merge two
        identical jump dicts and silently double every dissipation rate."""
        groups = {k: [term.dag() for term in v] for k, v in self.groups.items()}
        return Model(self.subsystems, groups)

    def replace(self, **named) -> "Model":
        """Swap out named term groups wholesale: the composable-error workflow.

            realized = model.replace(drive=noisy_drive)

        Each value is a Model; ALL its terms (and jumps, and any new
        subsystems it introduces) land under the replaced name. The group must
        already exist -- replacing an unknown name is almost always a typo, so
        it raises. To add a group, use `+`; to delete one, use `without()`."""
        subsystems = dict(self.subsystems)
        groups = {k: list(v) for k, v in self.groups.items()}
        jumps = {k: list(v) for k, v in self.jumps.items()}
        for name, replacement in named.items():
            if name not in groups and name not in jumps:
                raise KeyError(f"no term group named {name!r}; have "
                               f"{sorted(set(groups) | set(jumps))}")
            if not isinstance(replacement, Model):
                raise TypeError(f"replacement for {name!r} must be a Model")
            subsystems = _merge_registry(subsystems, replacement.subsystems)
            groups[name] = [t for terms in replacement.groups.values() for t in terms]
            if not groups[name]:
                del groups[name]
            new_jumps = [t for terms in replacement.jumps.values() for t in terms]
            if new_jumps:
                jumps[name] = new_jumps
            elif name in jumps:
                del jumps[name]
        return Model(subsystems, groups, jumps)

    def without(self, *names) -> "Model":
        """Drop named term groups (from both H terms and jumps)."""
        for name in names:
            if name not in self.groups and name not in self.jumps:
                raise KeyError(f"no term group named {name!r}")
        groups = {k: v for k, v in self.groups.items() if k not in names}
        jumps = {k: v for k, v in self.jumps.items() if k not in names}
        return Model(self.subsystems, groups, jumps)

    def group(self, name) -> "Model":
        """Extract one named group as its own Model (same registry)."""
        out = Model(self.subsystems)
        if name in self.groups:
            out.groups[name] = list(self.groups[name])
        if name in self.jumps:
            out.jumps[name] = list(self.jumps[name])
        if not out.groups and not out.jumps:
            raise KeyError(f"no term group named {name!r}")
        return out

    # ---- materialization (the Mechanism protocol) -----------------------

    @property
    def dim(self) -> int:
        d = 1
        for v in self.subsystems.values():
            d *= v
        return d

    def _embed(self, term: Term) -> np.ndarray:
        return np.asarray(embed(term.local_matrix(), self.subsystems, term.involved()))

    def _structure(self):
        """Cheap fingerprint of what `_cache` was built from. Instances are
        meant to be immutable, but `H.groups["drive"].append(...)` (or
        `... [0] = other_term`) is easy to write and would otherwise keep
        serving the stale cache.

        Identity-level, not value-level: this catches terms added, removed, or
        swapped, but NOT a Term mutated in place (`t.coeff = ...`), which keeps
        its id. Rebuild the Model rather than edit a Term."""
        return (tuple((k, tuple(id(t) for t in v)) for k, v in self.groups.items()),
                tuple((k, tuple(id(t) for t in v)) for k, v in self.jumps.items()),
                tuple(self.subsystems.items()))

    def _materialize(self):
        """Embed every term once (embedding is time-independent), sum the
        static ones, and keep (coeff_fn, matrix) for the time-dependent ones."""
        key = self._structure()
        if self._cache is not None and self._cache_key == key:
            return self._cache
        frames = {t.frame for terms in list(self.groups.values()) + list(self.jumps.values())
                  for t in terms if t.frame is not None}
        if len(frames) > 1:
            warnings.warn(f"composing terms tagged with different frames {sorted(frames)} "
                          "-- literal addition of Models written in different "
                          "frames is not physically meaningful", stacklevel=3)
        static = np.zeros((self.dim, self.dim), dtype=complex)
        dynamic = []
        for terms in self.groups.values():
            for term in terms:
                mat = self._embed(term)
                if term.is_static:
                    static = static + term.coeff_at(0.0) * mat
                else:
                    dynamic.append((term.coeff, mat))
        jump_static = []
        jump_dynamic = []
        for terms in self.jumps.values():
            for term in terms:
                mat = self._embed(term)
                if term.is_static:
                    jump_static.append(term.coeff_at(0.0) * mat)
                else:
                    jump_dynamic.append((term.coeff, mat))
        self._cache = (static, dynamic, jump_static, jump_dynamic)
        self._cache_key = key
        return self._cache

    def hamiltonian(self, t) -> Operator:
        # `static` is the memoized sum; never hand it out or accumulate into it,
        # or a caller's in-place edit of H(t) would silently corrupt the cache.
        # (Stacking the dynamic terms into one (K,d,d) contraction was measured
        # SLOWER than this loop at realistic sizes -- the cost is in evaluating
        # the K coefficient callables, not in the matrix algebra.)
        static, dynamic, _, _ = self._materialize()
        H = static.copy()
        for coeff, mat in dynamic:
            H += coeff(t) * mat
        return Operator(H)

    def jump_operators(self, t) -> list:
        _, _, jump_static, jump_dynamic = self._materialize()
        return ([Operator(L) for L in jump_static]
                + [Operator(coeff(t) * mat) for coeff, mat in jump_dynamic])

    # ---- inspection ------------------------------------------------------

    def __repr__(self):
        subs = ", ".join(f"{n}:{d}" for n, d in self.subsystems.items())
        gs = ", ".join(f"{k}[{len(v)}]" for k, v in self.groups.items())
        js = ", ".join(f"{k}[{len(v)}]" for k, v in self.jumps.items())
        parts = [f"subsystems=({subs})", f"terms=({gs})"]
        if js:
            parts.append(f"jumps=({js})")
        return f"Model({', '.join(parts)})"


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
         frame: str | None = None, dims: dict | None = None) -> Model:
    """Build a one-term Model -- the atom everything composes from.

    op:    a matrix (with `on=` naming its subsystem), or a dict
           {name: matrix} for a product across several subsystems
           (e.g. {"spin": sigma_plus, "mode": a}), or {(n1, n2): matrix} for
           a joint operator that doesn't factor (needs `dims=`).
    coeff: scalar, or callable f(t) for time-dependent control.
    name:  the term-group name -- the handle `replace()` swaps by. Unnamed
           terms get a unique auto-name (composable, but not swappable).
    frame: optional tag ("lab", "rotating@w0", ...); mixing distinct tags in
           one Model warns at materialization.
    """
    ops, term_dims = _build_ops_and_dims(op, on, dims)
    key = name if name is not None else f"term{next(_anon_counter)}"
    t = Term(coeff, ops, term_dims, frame)
    return Model(term_dims, groups={key: [t]})


def jump(op, on=None, coeff: Coefficient = 1.0, name: str | None = None,
         dims: dict | None = None) -> Model:
    """Build a Model carrying one Lindblad jump operator (and no
    coherent term). The materialized L is coeff * (embedded op) -- keep the
    sqrt(rate) convention: pass coeff=np.sqrt(gamma).

    Composes with `+` exactly like coherent terms, so a dissipative component
    is just another named group you can `replace()` or `without()`."""
    ops, term_dims = _build_ops_and_dims(op, on, dims)
    key = name if name is not None else f"jump{next(_anon_counter)}"
    t = Term(coeff, ops, term_dims, None)
    return Model(term_dims, jumps={key: [t]})


def hconj(h: Model) -> Model:
    """h + h.dag() -- the ubiquitous `X + h.c.` pattern in one call.

    Any jump operators on `h` ride through exactly once (see `dag`): only the
    coherent terms are conjugated and added."""
    return h + h.dag()
