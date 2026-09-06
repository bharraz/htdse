"""The READ layer: turning a solved answer back into physics you can read.

BUILD (`term`, `driven_spins`, ...) and RUN (the evolution classes) both have a
complete, composable, functional surface. This module is the same idiom
applied to the third step, which used to not exist as package code: given a
propagator or a solved state, what Hamiltonian did it actually implement, and
how much do you trust the answer?

Every function here takes plain data (an array, a dict) and returns plain
data -- no analysis object, no method chaining. `magnus_pauli` already does
this job perturbatively for a Pauli-only H(t); this module is the
non-perturbative, post-solve counterpart, for when the propagator you have is
the honest numerical answer (from `UnitaryEvolution`, or a closed form like
`ms_closed_form`) rather than a series expansion, and/or the system carries a
mode `magnus_pauli` can't touch (it needs dim = 2^n).

Typical use, projecting a spin-dependent-force gate down to its spin-only
effective Hamiltonian once the motion has (approximately) returned:

    M = project_block(U, subsystems, on="mode", state=0)   # <0|U|0> on the mode
    c = closure(M)                                    # 1.0 = motion returned cleanly
    H_eff = generator(M, T)                           # M ~= exp(-i H_eff T)
    print(paulis(H_eff))                              # {"XX": ..., "ZI": ...}
    print(max_eigenphase(H_eff, T))                   # trust the row above this?
"""
import numpy as np
from numbers import Integral
from scipy.linalg import logm

from .subsystems import _total_dim
from ..magnus import pauli_decompose
from ..util import fidelity

paulis = pauli_decompose  # same function, read-layer name for discoverability


def project_block(U, subsystems: dict, on: str, state: int = 0) -> np.ndarray:
    """<state|_on U |state>_on -- project OUT one named subsystem of an
    operator by fixing it to one basis state on both sides, leaving the
    operator on the remaining subsystems.

    Typical use: `on="mode"`, `state=0` reads off the spin-only gate a
    spin-dependent-force Hamiltonian implements, GIVEN the motion returns to
    its starting Fock state -- which is an assumption, not a guarantee.
    Check it with `closure(project_block(...))` before trusting the result: if the
    motion doesn't fully decouple, `project_block` silently discards the leaked
    amplitude and hands back a well-formed but non-unitary matrix.

    U: (D, D) operator over `subsystems` (same registry-order convention as
       `partial_trace`/`embed`).
    on: the subsystem name to fix.
    state: basis index within that subsystem (default 0, e.g. the Fock
       vacuum for a motional mode).
    """
    U = np.asarray(U, dtype=complex)
    names = list(subsystems.keys())
    shape = list(subsystems.values())
    if on not in subsystems:
        raise KeyError(f"unknown subsystem {on!r}; registry has {names}")
    if U.shape != (_total_dim(subsystems),) * 2:
        raise ValueError(f"subsystems {subsystems} give dim {_total_dim(subsystems)}, "
                         f"but U has shape {U.shape}")
    i = names.index(on)
    n = len(names)
    if not (0 <= state < shape[i]):
        raise ValueError(f"state={state} out of range for subsystem {on!r} (dim {shape[i]})")
    tensor = U.reshape(*shape, *shape)
    idx = [slice(None)] * (2 * n)
    idx[i] = state
    idx[i + n] = state
    reduced = tensor[tuple(idx)]
    rest_dim = int(np.prod([d for j, d in enumerate(shape) if j != i])) if n > 1 else 1
    return np.asarray(reduced.reshape(rest_dim, rest_dim))


def closure(M) -> float:
    """How close a projected block `M` (from `project_block`, typically) is to
    unitary -- the physical closure diagnostic.

    1.0: perfect closure -- the projected-out subsystem returned to its
    starting state independent of everything else, so M really is (up to
    numerical error) the unitary it looks like.
    Less than 1: some state-dependent amplitude leaked into a DIFFERENT
    state of the projected-out subsystem and was silently discarded by
    `project_block`'s fixed-index slice -- real information loss `project_block` cannot
    see by itself. Returns the smallest singular value of M, i.e. its
    worst-case shrinkage as an operator (the closure fidelity for the worst
    input state)."""
    M = np.asarray(M, dtype=complex)
    return float(np.linalg.svd(M, compute_uv=False).min())


def generator(M, T: float) -> np.ndarray:
    """H_eff such that M ~= exp(-i H_eff T) -- the non-perturbative inverse
    of time evolution, via matrix logarithm.

    Branch-ambiguous once the gate's eigenphases spread past ~pi (`logm`
    picks a principal branch per eigenvalue) -- check with `max_eigenphase`
    before trusting the result. Hermitizes the raw log (a not-quite-unitary
    M -- see `closure` -- gives a not-quite-anti-Hermitian log) and drops the
    trace: an overall global phase is physically meaningless and doubly
    branch-ambiguous if read off directly, so it is removed here rather than
    left for `paulis`'s `II...I` coefficient to misreport.
    """
    M = np.asarray(M, dtype=complex)
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"generator needs a square operator, got {M.shape}")
    H = 1j * logm(M) / T
    H = (H + H.conj().T) / 2
    H = H - (np.trace(H) / H.shape[0]) * np.eye(H.shape[0], dtype=complex)
    return H


def max_eigenphase(H_eff, T: float) -> float:
    """Largest |eigenvalue(H_eff)| * T / pi -- how close `generator`'s logm
    extraction sat to its branch cut. Approaching or exceeding 1 means the
    gate's eigenphases wrapped past pi during the solve, and the extracted
    H_eff (and its `paulis`) may not mean what you think: recompute at a
    smaller T, or distrust the row. A heuristic threshold to watch, not a
    certificate -- a scan that looks discontinuous should be distrusted
    regardless of this number."""
    H_eff = np.asarray(H_eff, dtype=complex)
    hermitian = np.allclose(H_eff, H_eff.conj().T, atol=1e-8)
    evals = np.linalg.eigvalsh(H_eff) if hermitian else np.linalg.eigvals(H_eff)
    return float(np.max(np.abs(evals)) * T / np.pi)


def expect(operator, state) -> complex | np.ndarray:
    """<psi|operator|psi> for a ket, Tr(operator rho) for a density matrix --
    the one call that reads out an expectation value regardless of which
    Evolution class produced `state`. `bra(s) @ psi` still reads best for a
    literal amplitude <s|psi>; `expect` is for "what does this operator read
    on this state," ket or mixed, without branching on state.ndim yourself."""
    operator = np.asarray(operator)
    state = np.asarray(state)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError(f"expect needs a square operator, got shape {operator.shape}")
    if state.ndim == 1:
        return complex(np.vdot(state, operator @ state))
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return complex(np.trace(operator @ state))
    if state.ndim == 2:
        if state.shape[1] != operator.shape[0]:
            raise ValueError(f"expect ket batch dimension {state.shape[1]} does not match operator dimension {operator.shape[0]}")
        return np.einsum("ni,ij,nj->n", state.conj(), operator, state)
    if state.ndim == 3:
        if state.shape[-2:] != operator.shape:
            raise ValueError(f"expect density batch shape {state.shape[-2:]} does not match operator shape {operator.shape}")
        return np.einsum("ij,nji->n", operator, state)
    raise ValueError(f"expect needs a ket, density matrix, or batch thereof, got shape {state.shape}")


def _basis_index(selector, dim: int) -> int:
    if isinstance(selector, str):
        if not selector or any(bit not in "01" for bit in selector):
            raise ValueError(f"basis label must be a qubit bitstring, got {selector!r}")
        if 2 ** len(selector) != dim:
            raise ValueError(f"basis label {selector!r} describes dimension {2 ** len(selector)}, "
                             f"but the state dimension is {dim}")
        return int(selector, 2)
    if isinstance(selector, Integral) and not isinstance(selector, (bool, np.bool_)):
        index = int(selector)
        if 0 <= index < dim:
            return index
        raise IndexError(f"basis index {index} outside dimension {dim}")
    raise TypeError("basis selector must be an integer index or qubit bitstring")


def element(state, row, col=None, *, on=None, subsystems=None):
    """A ket coefficient or density-matrix element, optionally on a subsystem.

    `element(psi, "01")` is <01|psi>; `element(rho, "01", "10")` is
    <01|rho|10>. One selector on a density matrix means its diagonal element.
    With `on=`, all other named subsystems are traced out first, so the result
    is always a reduced-density-matrix element. Trajectories return one value
    per time.
    """
    state = np.asarray(state, dtype=complex)
    if on is not None:
        if subsystems is None:
            raise ValueError("`on=` requires a `subsystems=` registry")
        from .subsystems import partial_trace
        kept = (on,) if isinstance(on, str) else tuple(on)
        unknown = [name for name in kept if name not in subsystems]
        if unknown:
            raise KeyError(f"unknown subsystem(s) {unknown}; registry has {list(subsystems)}")
        trace_out = tuple(name for name in subsystems if name not in kept)
        state = partial_trace(state, subsystems, trace_out)
    elif subsystems is not None:
        raise ValueError("`subsystems=` is only needed with `on=`")

    if state.ndim == 1:
        i = _basis_index(row, state.shape[0])
        if col is None:
            return complex(state[i])
        j = _basis_index(col, state.shape[0])
        return complex(state[i] * state[j].conj())
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        i = _basis_index(row, state.shape[0])
        j = i if col is None else _basis_index(col, state.shape[1])
        return complex(state[i, j])
    if state.ndim == 2:
        i = _basis_index(row, state.shape[1])
        if col is None:
            return np.asarray(state[:, i])
        j = _basis_index(col, state.shape[1])
        return np.asarray(state[:, i] * state[:, j].conj())
    if state.ndim == 3 and state.shape[-2] == state.shape[-1]:
        i = _basis_index(row, state.shape[-1])
        j = i if col is None else _basis_index(col, state.shape[-1])
        return np.asarray(state[:, i, j])
    raise ValueError(f"element needs a ket, density matrix, or trajectory, got {state.shape}")


def population(target, state, *, on=None, subsystems=None):
    """Population of a basis label or normalized target ket in `state`."""
    state = np.asarray(state, dtype=complex)
    if on is not None:
        if subsystems is None:
            raise ValueError("`on=` requires a `subsystems=` registry")
        from .subsystems import partial_trace
        kept = (on,) if isinstance(on, str) else tuple(on)
        unknown = [name for name in kept if name not in subsystems]
        if unknown:
            raise KeyError(f"unknown subsystem(s) {unknown}; registry has {list(subsystems)}")
        trace_out = tuple(name for name in subsystems if name not in kept)
        state = partial_trace(state, subsystems, trace_out)
    if isinstance(target, (str, Integral)) and not isinstance(target, (bool, np.bool_)):
        value = element(state, target)
        is_ket = state.ndim == 1 or (state.ndim == 2 and state.shape[0] != state.shape[1])
        result = np.abs(value) ** 2 if is_ket else np.real(value)
        return float(result) if np.ndim(result) == 0 else np.asarray(result)

    target = np.asarray(target, dtype=complex)
    if target.ndim != 1:
        raise ValueError("population target must be a basis label, index, or ket")
    if not np.isclose(np.linalg.norm(target), 1.0, atol=1e-8):
        raise ValueError("population target ket must be normalized")
    if state.shape[-1] != target.shape[0]:
        raise ValueError("target ket and state dimensions do not match")
    if state.ndim == 1:
        return float(np.abs(np.vdot(target, state)) ** 2)
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return float(np.real(np.vdot(target, state @ target)))
    if state.ndim == 2:
        return np.abs(state @ target.conj()) ** 2
    if state.ndim == 3:
        return np.real(np.einsum("i,nij,j->n", target.conj(), state, target))
    raise ValueError(f"population needs a ket, density matrix, or trajectory, got {state.shape}")


def overlap(state1, state2):
    """<state1|state2> for kets; Tr(state1^dag state2) for matrices."""
    a, b = np.asarray(state1, dtype=complex), np.asarray(state2, dtype=complex)
    if a.shape != b.shape:
        raise ValueError(f"overlap needs matching shapes, got {a.shape} and {b.shape}")
    if a.ndim == 1:
        return complex(np.vdot(a, b))
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        return complex(np.vdot(a, b))
    if a.ndim == 2:
        return np.einsum("ni,ni->n", a.conj(), b)
    if a.ndim == 3:
        return np.einsum("nij,nij->n", a.conj(), b)
    raise ValueError("overlap needs kets, matrices, or matching trajectories")


def distance(state1, state2) -> float:
    """Trace distance, reducing to sqrt(1-F) for two pure states."""
    a, b = np.asarray(state1, dtype=complex), np.asarray(state2, dtype=complex)
    if a.ndim == b.ndim == 1:
        return float(np.sqrt(max(0.0, 1.0 - fidelity(a, b))))
    if a.ndim == 1:
        a = np.outer(a, a.conj())
    if b.ndim == 1:
        b = np.outer(b, b.conj())
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("distance needs two kets or density matrices of equal dimension")
    return float(0.5 * np.linalg.svd(a - b, compute_uv=False).sum())


def show(H, t: float = 0.0, tol: float = 1e-10):
    """Print H(t) readably instead of a raw ndarray repr: a Pauli-coefficient
    table when dim = 2^n (via `paulis`), otherwise the rounded matrix with
    near-zero entries cleaned to exactly 0.

    H: a System (its `.hamiltonian(t)` is called) or a plain array --
    either way you get a look at the actual numbers without materializing
    and formatting it by hand first."""
    M = H.hamiltonian(t) if hasattr(H, "hamiltonian") else H
    M = np.asarray(M, dtype=complex)
    d = M.shape[0]
    n = round(np.log2(d)) if d > 0 else 0
    if d > 0 and 2 ** n == d:
        coeffs = paulis(M, tol=tol)
        if not coeffs:
            print(f"H(t={t}) = 0   ({d}x{d})")
            return
        print(f"H(t={t})   ({d}x{d}, Pauli basis)")
        for p, c in sorted(coeffs.items(), key=lambda kv: -abs(kv[1])):
            line = f"{c.real:+.5f}" if abs(c.imag) < tol else f"{c:+.5f}"
            print(f"  {p:<8s} {line}")
        return
    print(f"H(t={t})   ({d}x{d})")
    Mr = np.round(M, 6)
    Mr[np.abs(Mr) < tol] = 0
    with np.printoptions(suppress=True, linewidth=120):
        print(Mr)
