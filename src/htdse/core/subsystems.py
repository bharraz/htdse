import numpy as np
from scipy import sparse as _sp



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


def partial_trace(rho, subsystems: dict, trace_out: tuple) -> np.ndarray:
    """Partial trace of a ket or density matrix over named subsystems.

    H = H_1 (x) ... (x) H_N, `subsystems` = {name: dim(H_i)} in tensor-product order.
    Reshapes rho into a 2N-index tensor (one row + one column index per
    subsystem) and traces the row/column pair for each name in `trace_out`:

        rho_kept[m,n] = sum_a rho[(...,m,...,a,...), (...,n,...,a,...)]

    Accepts a ket (D,), density matrix (D, D), or a batched trajectory of
    either. Kets are converted to |psi><psi|, since a reduced state is
    generically mixed even when the joint state is pure.

    `trace_out` is a name or an iterable of names.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    elif rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        rho = np.einsum("ni,nj->nij", rho, rho.conj())
    _check_dims(rho, subsystems)
    # a bare "mode" would otherwise iterate into 'm','o','d','e'
    trace_out = (trace_out,) if isinstance(trace_out, str) else tuple(trace_out)
    unknown = [n for n in trace_out if n not in subsystems]
    if unknown:
        raise KeyError(f"unknown subsystem(s) {unknown}; registry has {list(subsystems)}")
    if len(set(trace_out)) != len(trace_out):
        raise ValueError(f"repeated subsystem in {trace_out}")
    names = list(subsystems.keys())
    shape = list(subsystems.values())
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

    kept_dim = int(np.prod([subsystems[n] for n in names])) if names else 1
    return np.asarray(tensor.reshape(*batch, kept_dim, kept_dim))  # back to flat matrices


def embed(op, subsystems: dict, on) -> np.ndarray:
    """Lift `op` into the full joint space defined by `subsystems` ({name: dim},
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
    remap), so the dense joint matrix is never formed. Dense in -> dense ndarray
    out; sparse in -> CSR out.
    """
    names = list(subsystems.keys())
    involved = (on,) if isinstance(on, str) else tuple(on)
    for nm in involved:
        if nm not in subsystems:
            raise KeyError(f"unknown subsystem {nm!r}; registry has {names}")
    if len(set(involved)) != len(involved):
        raise ValueError(f"repeated subsystem in {involved}")

    is_sparse = _sp.issparse(op)
    if not is_sparse:
        op = np.asarray(op, dtype=complex)
    d_inv = int(np.prod([subsystems[n] for n in involved]))
    if op.shape != (d_inv, d_inv):
        raise ValueError(f"op has shape {op.shape}, but subsystems {involved} "
                         f"give dimension {d_inv}")

    rest = [n for n in names if n not in involved]
    d_rest = int(np.prod([subsystems[n] for n in rest])) if rest else 1
    order_now = list(involved) + rest

    if is_sparse:
        # kron with the identity on the rest, all sparse: ordered (involved..., rest...)
        big = _sp.kron(op.astype(complex), _sp.identity(d_rest, dtype=complex),
                       format="coo")
        if order_now == names:
            return big.tocsr()
        return _permute_factors_sparse(big, subsystems, order_now)

    big = np.kron(op, np.eye(d_rest, dtype=complex))  # ordered: involved..., rest...

    if order_now == names:
        return np.asarray(big)  # already in canonical order, no permutation needed

    # permute tensor factors from (involved..., rest...) into `dims` order
    shape_now = [subsystems[n] for n in order_now]
    n = len(names)
    perm = [order_now.index(nm) for nm in names]  # output axis j reads source axis perm[j]
    tensor = big.reshape(shape_now + shape_now)
    tensor = tensor.transpose(perm + [p + n for p in perm])  # rows and columns together
    D = _total_dim(subsystems)
    return np.asarray(tensor.reshape(D, D))


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


def apply_unitary(operator, state, subsystems=None, on=None) -> np.ndarray:
    """Apply U to a ket, density matrix, or operator.

    `on` is a name or a tuple of names; `subsystems` is the {name: dim} registry
    (e.g. an evolution's `.subsystems`). A local operator is lifted with
    `embed`, so you never write the identity padding. For a full-space
    operator, omit `subsystems` and `on`:

        ket:       |psi> -> U|psi>
        matrix:    A     -> U A U^dagger,     U = embed(op, dims, on)

    Dispatched on shape (1-D -> ket, square 2-D -> conjugation). Example: a
    Hadamard on one ancilla is
    `apply_unitary(H, rho, subsystems, "a1")`; on both at once, use
    `apply_unitary(otimes(H, H), rho, subsystems, ("a1", "a2"))`.
    """
    U = np.asarray(operator)
    arr = np.asarray(state)
    if subsystems is not None:
        if on is None:
            raise ValueError("local operator application needs `on=` naming its subsystem(s)")
        U = embed(U, subsystems, on)
    elif on is not None:
        raise ValueError("`on=` requires a `subsystems=` registry")
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"unitary must be square, got shape {U.shape}")
    if arr.ndim == 1:
        if arr.shape[0] != U.shape[0]:
            raise ValueError(f"state dimension {arr.shape[0]} does not match unitary dimension {U.shape[0]}")
        return np.asarray(U @ arr)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        if arr.shape[0] != U.shape[0]:
            raise ValueError(f"matrix dimension {arr.shape[0]} does not match unitary dimension {U.shape[0]}")
        return np.asarray(U @ arr @ U.conj().T)
    if arr.ndim == 2:
        if arr.shape[1] != U.shape[0]:
            raise ValueError(f"ket batch dimension {arr.shape[1]} does not match unitary dimension {U.shape[0]}")
        return np.asarray(arr @ U.T)
    if arr.ndim == 3:
        if arr.shape[-2:] != U.shape:
            raise ValueError(f"matrix batch shape {arr.shape[-2:]} does not match unitary shape {U.shape}")
        return np.asarray(U @ arr @ U.conj().T)
    raise ValueError(f"target must be a ket, square matrix, or batch thereof; got shape {arr.shape}")


def change_basis(basis, state) -> np.ndarray:
    """Coordinates of a ket or matrix in a new orthonormal basis.

    The columns of `basis` are the new basis vectors in the old coordinates:
    |psi> -> V^dagger |psi>, and A -> V^dagger A V.
    """
    V = np.asarray(basis, dtype=complex)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError(f"basis must be a square matrix, got shape {V.shape}")
    if not np.allclose(V.conj().T @ V, np.eye(V.shape[0]), atol=1e-8):
        raise ValueError("basis columns must be orthonormal")
    return apply_unitary(V.conj().T, state)


def measure(operator, state, subsystems=None, on=None):
    """Condition on one projective or general measurement outcome.

    `operator` is the selected measurement/Kraus operator M. Returns
    `(post_state, probability)` using p=<psi|M^dag M|psi> for a ket or
    p=Tr(M^dag M rho) for a density matrix. A projector is the special case
    M=P. For a local measurement, pass `on=` and the subsystem registry; the
    returned state remains on the full Hilbert space and can be reduced with
    `partial_trace` if desired.
    """
    M = np.asarray(operator, dtype=complex)
    if subsystems is not None:
        if on is None:
            raise ValueError("local measurement needs `on=` naming its subsystem(s)")
        M = embed(M, subsystems, on)
    elif on is not None:
        raise ValueError("`on=` requires a `subsystems=` registry")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"measurement operator must be square, got shape {M.shape}")
    arr = np.asarray(state, dtype=complex)
    if arr.ndim == 1:
        collapsed = M @ arr
        p = float(np.real(np.vdot(collapsed, collapsed)))
        normalizer = np.sqrt(p)
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        collapsed = M @ arr @ M.conj().T
        p = float(np.real(np.trace(collapsed)))
        normalizer = p
    else:
        raise ValueError("measure needs one ket or density matrix, not a trajectory")
    if p < 1e-12:
        raise ValueError(f"measurement outcome has ~zero probability ({p:.3g}); "
                         "cannot condition on it")
    return np.asarray(collapsed / normalizer), p
