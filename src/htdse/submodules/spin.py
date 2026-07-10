"""Spin-1/2 sector: Pauli matrices, ladder operators, and human-readable
Pauli-string construction of composable multi-qubit Hamiltonians.

Convention: ket("0") = [1, 0] is the sigma_z = +1 eigenstate. sigma_plus
raises TOWARD |0> (sigma_plus |1> = |0>), i.e. |0> plays the excited state
when the splitting is written +(w0/2) sigma_z.
"""
import re

import numpy as np

from ..core.terms import Hamiltonian, term

# Pauli matrices and identity for the spin-1/2 sector.
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1| = (sx + i sy)/2
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)  # |1><0| = (sx - i sy)/2

PAULIS = {"X": sigma_x, "Y": sigma_y, "Z": sigma_z, "I": I2,
          "+": sigma_plus, "-": sigma_minus}

_TOKEN = re.compile(r"([XYZI+\-])(\d+)")


def pauli_term(spec: str, coeff=1.0, name=None, n_qubits=None,
               frame=None, prefix="q") -> Hamiltonian:
    """One product of single-qubit Paulis as a composable term-layer Hamiltonian.

    spec: e.g. "X0X1" (sigma_x on qubits 0 and 1), "Z2", "+0-1". Qubit i
    becomes subsystem "q{i}", so pauli terms compose with anything else by
    name. Repeated indices multiply on that qubit ("X0Y0" -> sigma_x sigma_y).
    coeff may be a scalar or f(t). `n_qubits` pre-registers q0..q{n-1} so the
    materialized matrix covers the full register even for qubits this term
    doesn't touch.
    """
    tokens = _TOKEN.findall(spec)
    if not tokens or "".join(p + i for p, i in tokens) != spec.replace(" ", ""):
        raise ValueError(f"could not parse Pauli spec {spec!r} "
                         "(expected e.g. 'X0X1', 'Z2', '+0-1')")
    ops: dict = {}
    for p, idx in tokens:
        key = f"{prefix}{int(idx)}"
        ops[key] = ops[key] @ PAULIS[p] if key in ops else PAULIS[p]
    h = term(ops, coeff=coeff, name=name, frame=frame)
    if n_qubits is not None:
        # widen the registry with untouched qubits (identity there)
        h = h + Hamiltonian({f"{prefix}{i}": 2 for i in range(n_qubits)})
    return h


def pauli_sum(spec: str, n_qubits=None, frame=None, prefix="q") -> Hamiltonian:
    """A sum of Pauli terms from one human-readable string:

        pauli_sum("0.5 X0X1 + 0.3 Z0 - Z1")

    Each summand is "[coefficient] SPEC" (coefficient defaults to 1); the
    result is an ordinary composable Hamiltonian (each summand its own
    auto-named group). For swappable groups build the summands individually
    with pauli_term(..., name=...) and `+` them.
    """
    total = Hamiltonian({f"{prefix}{i}": 2 for i in range(n_qubits)}) if n_qubits \
        else Hamiltonian()
    # Normalize "a - b" and "a -b" into "a + -b", then split on "+". The minus
    # must be preceded by whitespace to be a subtraction: a '-' with no space
    # before it is the sigma_minus token ("+0-1").
    normalized = re.sub(r"\s+-\s*", " + -", spec.strip())
    for piece in normalized.split("+"):
        piece = piece.strip()
        if not piece:
            continue
        neg = piece.startswith("-")
        if neg:
            piece = piece[1:].strip()
        parts = piece.split()
        if len(parts) == 1:
            coeff, body = 1.0, parts[0]
        elif len(parts) == 2:
            try:
                coeff = float(parts[0])
            except ValueError:
                raise ValueError(
                    f"could not parse summand {piece!r}: expected "
                    f"'[coefficient] SPEC' (e.g. '0.5 X0X1'), but {parts[0]!r} "
                    f"is not a number") from None
            body = parts[1]
        else:
            raise ValueError(f"could not parse summand {piece!r}")
        total = total + pauli_term(body, coeff=-coeff if neg else coeff,
                                   frame=frame, prefix=prefix)
    return total
