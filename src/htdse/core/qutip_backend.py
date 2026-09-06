"""Private QuTiP compiler and numerical evolution backend.

Public htdse values remain NumPy arrays.  This module is the single boundary
where they acquire QuTiP's tensor metadata and enter its solvers.
"""
from __future__ import annotations

import numpy as np
import qutip


def _dims(subsystems, dim, *, ket=False):
    factors = list((subsystems or {}).values())
    if not factors or int(np.prod(factors)) != dim:
        factors = [dim]
    return [factors, [1] * len(factors)] if ket else [factors, factors]


def qobj(array, subsystems=None, *, ket=False):
    """Copy an htdse array into a Qobj with the named registry's dimensions."""
    shape = np.shape(array)
    dim = int(shape[0])
    return qutip.Qobj(array.copy() if hasattr(array, "copy") else np.array(array),
                      dims=_dims(subsystems, dim, ket=ket))


def _coefficient(fn):
    # QuTiP versions differ on whether ``args`` is supplied positionally.
    return lambda t, *args, _fn=fn, **kwargs: _fn(t)


def compile_system(system, include_jumps=True):
    """Snapshot a System into QuTiP's native constant-matrix/coefficient form.

    Term-built Systems expose their already embedded decomposition, so QuTiP
    only calls scalar coefficients during integration.  Private analytic or
    legacy providers fall back to an operator callback.
    """
    subsystems = dict(getattr(system, "subsystems", {}) or {})
    materialize = getattr(system, "_materialize", None)
    if not callable(materialize):
        hamiltonian = getattr(system, "hamiltonian")

        def H(t, *args, **kwargs):
            return qobj(np.asarray(hamiltonian(t)), subsystems)

        jumps = []
        if include_jumps:
            for L in getattr(system, "jump_operators", lambda t: [])(0.0):
                jumps.append(qobj(L, subsystems))
        return qutip.QobjEvo(H), jumps

    static, dynamic, jump_static, jump_dynamic = materialize()
    static_qobj = qobj(static, subsystems)
    contributions = []
    for kind, *payload in dynamic:
        if kind == "operator":
            contribution = payload[0]

            def operator(t, *args, _term=contribution, **kwargs):
                return qobj(_term.at(t, target_dims=subsystems), subsystems)

            contributions.append(qutip.QobjEvo(operator))
        else:
            fn, matrix = payload
            contributions.append(qutip.QobjEvo(
                [qobj(matrix, subsystems), _coefficient(fn)]))
    if contributions:
        compiled_h = qutip.QobjEvo(static_qobj)
        for contribution in contributions:
            compiled_h += contribution
    else:
        compiled_h = static_qobj

    if not include_jumps:
        return compiled_h, []
    c_ops = [qobj(L, subsystems) for L in jump_static]
    for kind, *payload in jump_dynamic:
        if kind == "operator":
            contribution = payload[0]

            def operator(t, *args, _term=contribution, **kwargs):
                return qobj(_term.at(t, target_dims=subsystems), subsystems)

            c_ops.append(qutip.QobjEvo(operator))
        else:
            fn, matrix = payload
            c_ops.append([qobj(matrix, subsystems), _coefficient(fn)])
    return compiled_h, c_ops


def solver_options(method="dop853", rtol=1e-8, atol=1e-10):
    methods = {
        "RK45": "vern7",
        "RK23": "vern7",
        "DOP853": "dop853",
        "BDF": "bdf",
        "LSODA": "lsoda",
        "adams": "adams",
        "vern7": "vern7",
        "vern9": "vern9",
        "diag": "diag",
        "krylov": "krylov",
    }
    selected = methods.get(method, method)
    options = {
        "method": selected,
        "progress_bar": False,
        "normalize_output": False,
        "store_states": True,
    }
    if selected not in {"diag", "krylov"}:
        options.update(rtol=float(rtol), atol=float(atol))
    return options


def solve(H, initial, times, *, subsystems=None, c_ops=(), method="dop853",
          rtol=1e-8, atol=1e-10):
    """Solve at an ordered time grid and return a stack of NumPy states."""
    initial = np.asarray(initial)
    options = solver_options(method, rtol, atol)
    if initial.ndim == 1:
        result = qutip.sesolve(
            H, qobj(initial, subsystems, ket=True), times, options=options)
        return np.asarray([state.full().ravel() for state in result.states])
    if c_ops:
        result = qutip.mesolve(
            H, qobj(initial, subsystems), times, c_ops=c_ops, options=options)
        return np.asarray([state.full() for state in result.states])
    propagators = qutip.propagator(H, times, options=options)
    if isinstance(propagators, qutip.Qobj):
        propagators = [propagators]
    return np.asarray([U.full() @ initial for U in propagators])
