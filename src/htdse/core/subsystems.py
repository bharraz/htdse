import numpy as np

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
    """
    names = list(dims.keys())
    involved = (subsystem,) if isinstance(subsystem, str) else tuple(subsystem)
    for nm in involved:
        if nm not in dims:
            raise KeyError(f"unknown subsystem {nm!r}; registry has {names}")
    if len(set(involved)) != len(involved):
        raise ValueError(f"repeated subsystem in {involved}")

    op = np.asarray(op, dtype=complex)
    d_inv = int(np.prod([dims[n] for n in involved]))
    if op.shape != (d_inv, d_inv):
        raise ValueError(f"op has shape {op.shape}, but subsystems {involved} "
                         f"give dimension {d_inv}")

    rest = [n for n in names if n not in involved]
    d_rest = int(np.prod([dims[n] for n in rest])) if rest else 1
    big = np.kron(op, np.eye(d_rest, dtype=complex))  # ordered: involved..., rest...

    order_now = list(involved) + rest
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
