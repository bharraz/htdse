"""QuTiP bridge: compose here, solve there (or vice versa).

qutip is NOT a dependency. It is imported the first time you call something in
this module, and the error if it is missing says so plainly.

The whole bridge is small because both sides are matrices underneath. The one
real difference is how the tensor structure is recorded:

    htdse   Model.subsystems   {"spin": 2, "mode": 13}    ordered dict, NAMED
    qutip   Qobj.dims          [[2, 13], [2, 13]]         nested list, POSITIONAL

Same information; htdse's is strictly richer, since the names survive. The
conversion is therefore trivial in one direction and lossy in the other -- and
ORDER IS THE WHOLE CONTRACT. htdse's registry is ordered by first appearance,
qutip's dims is positional, so a registry that disagrees with how the state was
actually built produces a plausible-looking wrong answer with no error at all.
`to_qobj` checks the dimensions multiply out; it cannot check the order for you.

WHY YOU WOULD USE THIS

Compose in htdse (named groups, `replace()`, the guards), then hand the result
to qutip for the things htdse deliberately does not implement: `mcsolve`
(quantum-trajectory Monte Carlo), `steadystate`, `floquet`. Or go the other way
and pull qutip's state-prep and entanglement measures (`coherent`, `thermal_dm`,
`concurrence`, `negativity`, `entropy_vn`) onto htdse output -- that direction
needs nothing but `to_qobj`.

    from htdse.interop.qutip import to_qutip, to_qobj
    H_q, c_ops = to_qutip(model)                  # -> qutip's own [H0,[H1,f1]] form
    result = qutip.mcsolve(H_q, to_qobj(psi0, model.subsystems), ts, c_ops)
"""
import numpy as np


def _qt():
    """Import qutip on demand, with an error that explains itself."""
    try:
        import qutip
    except ImportError as e:
        raise ImportError(
            "htdse.interop.qutip needs qutip installed (`pip install qutip`). "
            "It is deliberately not an htdse dependency -- nothing else in the "
            "package imports it."
        ) from e
    return qutip


def _dims_for(array, subsystems):
    """qutip `dims` for an array over this registry, ket/operator aware."""
    dims = list(subsystems.values())
    total = int(np.prod(dims)) if dims else 1
    a = np.asarray(array) if not hasattr(array, "toarray") else array
    shape = a.shape
    if shape[-1] != total and shape[0] != total:
        raise ValueError(
            f"registry {subsystems} multiplies to {total}, which matches no axis "
            f"of an array with shape {shape}. The registry and the state disagree "
            f"about the space -- check you passed the right `subsystems`.")
    if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):   # ket
        return [dims, [1] * len(dims)]
    return [dims, dims]                                          # operator / density matrix


def to_qobj(array, subsystems=None):
    """ndarray (or scipy sparse) -> `qutip.Qobj`, with tensor structure attached.

    subsystems: an htdse registry ({name: dim}, e.g. `model.subsystems`). Its
    ORDER must match how the state was built -- see the module docstring. Omit
    it for a plain unstructured operator (qutip will treat it as one factor).
    """
    qutip = _qt()
    if subsystems:
        return qutip.Qobj(array, dims=_dims_for(array, subsystems))
    return qutip.Qobj(array)


def from_qobj(qobj, names=None):
    """`qutip.Qobj` -> (ndarray, registry).

    names: subsystem names to attach, one per factor of `qobj.dims[0]`. qutip
    does not carry names, so without this you get positional placeholders
    ("s0", "s1", ...) -- the information genuinely is not in the Qobj.
    """
    dims = list(qobj.dims[0])
    if names is None:
        names = [f"s{i}" for i in range(len(dims))]
    if len(names) != len(dims):
        raise ValueError(f"got {len(names)} names for {len(dims)} qutip factors {dims}")
    return np.asarray(qobj.full()), dict(zip(names, dims))


def to_qutip(model, include_jumps=True):
    """htdse `Model` -> (H, c_ops) in qutip's own time-dependent format.

    Returns H as `[H0, [H1, f1], [H2, f2], ...]` -- qutip's NATIVE list form,
    not a Python callback returning a Qobj. That matters: a callable-returning-
    Qobj is qutip's slow path (it re-enters Python for the whole operator on
    every step), while the list form lets it keep the constant matrices and
    evaluate only the scalar coefficients. htdse's term layer already stores
    exactly this decomposition, so the fast form is what falls out naturally.

    c_ops are the jump operators, same convention as htdse (each pre-scaled by
    sqrt(rate)) which is also qutip's. Pass `include_jumps=False` for the
    coherent part alone.

    Feed the result to `qutip.mesolve` / `mcsolve` / `steadystate`.
    """
    _qt()  # fail early and clearly if qutip is missing
    subs = model.subsystems
    static, dynamic, jump_static, jump_dynamic = model._materialize()

    def wrap(fn):
        # qutip may call a coefficient as f(t) or f(t, args) depending on
        # version/solver; htdse coefficients take t alone.
        return lambda t, *args, _f=fn, **kw: _f(t)

    H = [to_qobj(static, subs)]
    H += [[to_qobj(mat, subs), wrap(fn)] for fn, mat in dynamic]
    if len(H) == 1:
        H = H[0]        # purely static: hand back a bare Qobj, not a 1-list

    if not include_jumps:
        return H, []
    c_ops = [to_qobj(L, subs) for L in jump_static]
    c_ops += [[to_qobj(mat, subs), wrap(fn)] for fn, mat in jump_dynamic]
    return H, c_ops


def as_mechanism(source, subsystems=None, jumps=None):
    """A qutip `Qobj` / `QobjEvo` -> an htdse `Mechanism`, so the htdse evolution
    classes (and their guards, and `compare_over`) can consume it.

    source:      a Qobj (constant H) or QobjEvo (time-dependent).
    subsystems:  registry to attach; defaults to positional names read off the
                 Qobj's own dims.
    jumps:       optional list of Qobj collapse operators, htdse's sqrt(rate)
                 convention (== qutip's).
    """
    qutip = _qt()
    from ..core.mechanism import Mechanism

    dims = list(source.dims[0])
    if subsystems is None:
        subsystems = {f"s{i}": d for i, d in enumerate(dims)}
    jump_arrays = [np.asarray(L.full()) for L in (jumps or [])]
    # NOT `callable(source)`: a plain Qobj is itself callable in qutip 5 (it
    # applies the operator to a state), so duck-typing here would call the
    # constant operator as if it were a time-dependent one and raise.
    time_dependent = isinstance(source, qutip.QobjEvo)

    class _QutipMechanism(Mechanism):
        def __init__(self):
            self.subsystems = dict(subsystems)
            self.source = source

        def hamiltonian(self, t):
            H = self.source(t) if time_dependent else self.source
            return np.asarray(H.full())

        def jump_operators(self, t):
            return list(jump_arrays)

        def __repr__(self):
            return f"QutipMechanism({type(self.source).__name__}, dims={dims})"

    return _QutipMechanism()
