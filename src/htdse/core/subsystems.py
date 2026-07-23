import numpy as np
from scipy import sparse as _sp

from .operator import Operator


def _total_dim(dims: dict) -> int:
    total = 1
    for d in dims.values():
        total *= d
    return total


def _check_dims(rho, dims: dict) -> None:
    total = _total_dim(dims)
    if rho.shape[-1] != total:
        raise ValueError(f"subsystem dims {dims} multiply to {total}, "
                         f"but the operator has dimension {rho.shape[-1]}")


def partial_trace(rho: Operator, dims: dict, trace_out: tuple) -> Operator:
    """Partial trace of a density matrix over named subsystems.

    H = H_1 (x) ... (x) H_N, `dims` = {name: dim(H_i)} in tensor-product order.
    Reshapes rho into a 2N-index tensor (one row + one column index per
    subsystem) and traces the row/column pair for each name in `trace_out`:

        rho_kept[m,n] = sum_a rho[(...,m,...,a,...), (...,n,...,a,...)]

    Accepts a single density matrix (D, D) or a batched trajectory
    (..., D, D) -- e.g. the (n_times, D, D) stack an evolution's `state_at(ts)`
    produces -- tracing every entry in one shot.

    `trace_out` is a name or an iterable of names.
    """
    rho = np.asarray(rho, dtype=complex)
    _check_dims(rho, dims)
    # a bare "mode" would otherwise iterate into 'm','o','d','e'
    trace_out = (trace_out,) if isinstance(trace_out, str) else tuple(trace_out)
    unknown = [n for n in trace_out if n not in dims]
    if unknown:
        raise KeyError(f"unknown subsystem(s) {unknown}; registry has {list(dims)}")
    if len(set(trace_out)) != len(trace_out):
        raise ValueError(f"repeated subsystem in {trace_out}")
    names = list(dims.keys())
    shape = list(dims.values())
    N = len(names)
    batch = rho.shape[:-2]
    nb = len(batch)

    tensor = rho.reshape(*batch, *shape, *shape)  # split flat indices per subsystem
    for name in trace_out:
        i = names.index(name)
        # axis nb+i = this subsystem's row index, axis nb+i+N its column index
        # (in the *current* tensor) -- trace them together, then drop both.
        tensor = np.trace(tensor, axis1=nb + i, axis2=nb + i + N)
        names.pop(i)
        N -= 1

    kept_dim = int(np.prod([dims[n] for n in names])) if names else 1
    return Operator(tensor.reshape(*batch, kept_dim, kept_dim))  # back to flat matrices


def embed(op: Operator, dims: dict, subsystem) -> Operator:
    """Lift `op` into the full joint space defined by `dims` ({name: dim},
    in tensor-product order), acting as identity everywhere it isn't defined.

    `subsystem` is one name or a tuple of names:

    - embed(H_A, dims, "A")            -- op on one factor: H_A (x) I (x) ...
    - embed(M, dims, ("A", "C"))       -- op on several, possibly NON-ADJACENT
      factors. M lives on H_A (x) H_C *in the order given*; the identity on
      everything else and the permutation into `dims` order are handled here,
      so interaction terms between arbitrary factors never need hand-rolled
      Kronecker bookkeeping.

    A scipy.sparse `op` returns a sparse (CSR) result -- the whole computation
    stays sparse (kron with a sparse identity, permutation as an O(nnz) index
    remap), so the dense joint matrix is never formed. Dense in -> `Operator`
    out; sparse in -> CSR out.
    """
    names = list(dims.keys())
    involved = (subsystem,) if isinstance(subsystem, str) else tuple(subsystem)
    for nm in involved:
        if nm not in dims:
            raise KeyError(f"unknown subsystem {nm!r}; registry has {names}")
    if len(set(involved)) != len(involved):
        raise ValueError(f"repeated subsystem in {involved}")

    is_sparse = _sp.issparse(op)
    if not is_sparse:
        op = np.asarray(op, dtype=complex)
    d_inv = int(np.prod([dims[n] for n in involved]))
    if op.shape != (d_inv, d_inv):
        raise ValueError(f"op has shape {op.shape}, but subsystems {involved} "
                         f"give dimension {d_inv}")

    rest = [n for n in names if n not in involved]
    d_rest = int(np.prod([dims[n] for n in rest])) if rest else 1
    order_now = list(involved) + rest

    if is_sparse:
        # kron with the identity on the rest, all sparse: ordered (involved..., rest...)
        big = _sp.kron(op.astype(complex), _sp.identity(d_rest, dtype=complex),
                       format="coo")
        if order_now == names:
            return big.tocsr()
        return _permute_factors_sparse(big, dims, order_now)

    big = np.kron(op, np.eye(d_rest, dtype=complex))  # ordered: involved..., rest...

    if order_now == names:
        return Operator(big)  # already in canonical order, no permutation needed

    # permute tensor factors from (involved..., rest...) into `dims` order
    shape_now = [dims[n] for n in order_now]
    n = len(names)
    perm = [order_now.index(nm) for nm in names]  # output axis j reads source axis perm[j]
    tensor = big.reshape(shape_now + shape_now)
    tensor = tensor.transpose(perm + [p + n for p in perm])  # rows and columns together
    D = _total_dim(dims)
    return Operator(tensor.reshape(D, D))


def _permute_factors_sparse(big, dims: dict, order_now: list):
    """Sparse counterpart of the dense reshape/transpose factor permutation.

    `big` (COO, D x D) lives on the tensor factors in `order_now` order; the
    result lives on them in `dims` (canonical) order. A sparse matrix cannot be
    reshaped into a 2N-index tensor, so instead each nonzero's flat row/column
    index is decomposed into per-factor digits, the digits are reordered, and
    the index is re-flattened -- an O(nnz) remap, exactly equivalent to the
    dense `tensor.transpose(perm + [p + n for p in perm])`.
    """
    names = list(dims.keys())
    shape_now = [dims[n] for n in order_now]
    perm = [order_now.index(nm) for nm in names]  # output axis j reads source axis perm[j]
    out_shape = [shape_now[p] for p in perm]      # == [dims[n] for n in names]

    row_digits = np.unravel_index(big.row, shape_now)  # one digit array per factor
    col_digits = np.unravel_index(big.col, shape_now)
    new_rows = np.ravel_multi_index([row_digits[p] for p in perm], out_shape)
    new_cols = np.ravel_multi_index([col_digits[p] for p in perm], out_shape)

    D = _total_dim(dims)
    return _sp.coo_matrix((big.data, (new_rows, new_cols)), shape=(D, D)).tocsr()


def apply(state, op, dims: dict, on) -> Operator:
    """Apply a local operator `op` on named subsystem(s) `on` to a ket or a
    density matrix, leaving the other subsystems alone.

    `on` is a name or a tuple of names; `dims` is the {name: dim} registry
    (e.g. an evolution's `.subsystems`). `op` acts on just the `on` factor(s)
    and is lifted with `embed`, so you never write the identity padding:

        ket:            |psi>  ->  U|psi>
        density matrix: rho    ->  U rho U^dagger,     U = embed(op, dims, on)

    Dispatched on shape (1-D -> ket, 2-D -> density matrix). Example: a
    Hadamard on one ancilla is `apply(rho, H, dims, "a1")`; on both at once,
    `apply(rho, otimes(H, H), dims, ("a1", "a2"))`.
    """
    U = embed(op, dims, on)
    arr = np.asarray(state)
    if arr.ndim == 1:
        return Operator(U @ arr)
    return Operator(U @ arr @ U.conj().T)


def project(state, dims: dict, on, onto):
    """Projective measurement of subsystem(s) `on` onto the pure state `onto`,
    reduced onto the remaining subsystems. Returns (reduced_rho, probability).

    Accepts a ket OR a density matrix, and ALWAYS returns a density matrix --
    a measured-and-reduced state is generically mixed. `onto` is a state on
    the `on` factor(s), so measuring in the +/- basis is just
    `onto = otimes(|+>, |+>)` (no basis change needed). The probability is the
    Born rule Tr(P rho), P = embed(|onto><onto|, dims, on).

        ket:            phi = P|psi>;  p = <phi|phi>;  reduce |phi><phi|
        density matrix: rho -> P rho P;  p = Tr(P rho);  reduce
    then partial-trace out `on`, leaving the conditional state on the rest.
    """
    onto = np.asarray(onto, dtype=complex)
    onto = onto / np.linalg.norm(onto)
    P = embed(np.outer(onto, onto.conj()), dims, on)   # |onto><onto| embedded
    arr = np.asarray(state, dtype=complex)
    if arr.ndim == 1:                       # ket
        phi = P @ arr
        p = float(np.real(np.vdot(phi, phi)))
        collapsed = np.outer(phi, phi.conj())
    else:                                   # density matrix
        collapsed = P @ arr @ P
        p = float(np.real(np.trace(collapsed)))
    if p < 1e-12:
        raise ValueError(f"measurement outcome has ~zero probability ({p:.3g}); "
                         "cannot condition on it")
    reduced = partial_trace(Operator(collapsed / p), dims, on)
    return reduced, p
