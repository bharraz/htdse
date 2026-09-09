"""Spin-1/2 sector: Pauli matrices, ladder operators, and human-readable
Pauli-string construction of composable multi-qubit Hamiltonians.

Convention: ket("0") = [1, 0] is the sigma_z = +1 eigenstate. sigma_plus
raises TOWARD |0> (sigma_plus |1> = |0>), i.e. |0> plays the excited state
when the splitting is written +(w0/2) sigma_z.
"""
import re

import numpy as np

from ..core.terms import System, term

# Pauli matrices and identity for the spin-1/2 sector.
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1| = (sx + i sy)/2
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)  # |1><0| = (sx - i sy)/2

PAULIS = {"X": sigma_x, "Y": sigma_y, "Z": sigma_z, "I": I2,
          "+": sigma_plus, "-": sigma_minus}

_NUMBERED_TOKEN = re.compile(r"([XYZI+\-])(\d+)")
_NAMED_TOKEN = re.compile(r"([XYZI+\-])([^\s]+)")


def _pauli_tokens(spec):
    """Parse compact numbered or whitespace-delimited named Pauli tokens."""
    compact = spec.strip()
    if not compact:
        return []

    # Arbitrary subsystem names need an explicit boundary: ``Xr1 Xq2``.
    # Preserve the convenient historical spelling ``X0X1`` for numbered
    # registers, and allow a single named token such as ``Xspin``.
    if any(char.isspace() for char in compact):
        tokens = []
        for token in compact.split():
            match = _NAMED_TOKEN.fullmatch(token)
            if match is None:
                return []
            tokens.append(match.groups())
        return tokens

    numbered = _NUMBERED_TOKEN.findall(compact)
    if numbered and "".join(p + label for p, label in numbered) == compact:
        return numbered
    match = _NAMED_TOKEN.fullmatch(compact)
    return [match.groups()] if match is not None else []


def pauli_term(spec: str, coeff=1.0, name=None, n_qubits=None,
               frame=None, prefix="q") -> System:
    """One product of single-qubit Paulis as a composable term-layer `System`.

    spec: e.g. "Xr1 Xq2" (sigma_x on named subsystems r1 and q2), or the
    compact numbered form "X0X1". A numbered label i becomes subsystem
    "q{i}"; a named label is used verbatim. Spaces delimit arbitrary names.
    Repeated labels multiply on that subsystem ("Xspin Yspin" -> sigma_x
    sigma_y).
    coeff may be a scalar or f(t). `n_qubits` pre-registers q0..q{n-1} so the
    materialized matrix covers the full register even for qubits this term
    doesn't touch.
    """
    tokens = _pauli_tokens(spec)
    if not tokens:
        raise ValueError(f"could not parse Pauli spec {spec!r} "
                         "(expected e.g. 'Xr1 Xq2', 'X0X1', or 'Zspin')")
    ops: dict = {}
    for p, label in tokens:
        key = f"{prefix}{int(label)}" if label.isdigit() else label
        ops[key] = ops[key] @ PAULIS[p] if key in ops else PAULIS[p]
    h = term(ops, coeff=coeff, name=name, frame=frame)
    if n_qubits is not None:
        # Seed the registry with the FULL q0..q{n-1} order first, then add h
        # onto it -- `+` preserves the LEFT operand's registry order and only
        # appends keys the left side doesn't already have, so widening by
        # `h + seed` (seed on the right) puts h's own touched qubits first and
        # the untouched ones wherever they land after that -- e.g. "Z2" would
        # register (q2, q0, q1) instead of (q0, q1, q2), silently swapping
        # which physical qubit occupies which tensor slot. Seeding first keeps
        # the natural q0..q{n-1} order regardless of which qubits `spec` touches.
        h = System({f"{prefix}{i}": 2 for i in range(n_qubits)}) + h
    return h


def pauli_sum(spec: str, n_qubits=None, frame=None, prefix="q") -> System:
    """A sum of Pauli terms from one human-readable string:

        pauli_sum("0.5 Xr1 Xq2 + 0.3 Zr1 - Zq2")

    Each summand is "[coefficient] SPEC" (coefficient defaults to 1); the
    result is an ordinary composable `System` (each summand its own
    auto-named group). For swappable groups build the summands individually
    with pauli_term(..., name=...) and `+` them.
    """
    total = System({f"{prefix}{i}": 2 for i in range(n_qubits)}) if n_qubits \
        else System()
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
        try:
            coeff = float(parts[0])
        except ValueError:
            coeff, body = 1.0, piece
        else:
            if len(parts) == 1:
                raise ValueError(f"summand {piece!r} has a coefficient but no Pauli term")
            body = " ".join(parts[1:])
        total = total + pauli_term(body, coeff=-coeff if neg else coeff,
                                   frame=frame, prefix=prefix)
    return total
