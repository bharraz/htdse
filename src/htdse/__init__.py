"""htdse -- target-vs-reality Hamiltonian simulation.

Everything a physicist reaches for interactively is importable from the top:

    import htdse as ht
    H = ht.term(...) + ht.term(...)
    ev = ht.HamiltonianEvolution(H, psi0)
"""
from .core.config import quiet, no_truncation_check
from .core.operator import Operator
from .core.truncation import TruncationWarning, truncation_populations
from .core.mechanism import Mechanism
from .core.terms import Model, term, jump, hconj
from .core.evolution import (HamiltonianEvolution, UnitaryEvolution,
                             DensityMatrixEvolution, LindbladEvolution)
from .core.subsystems import embed, partial_trace
from .core.compare import compare_over
from .magnus import magnus, magnus_pauli, pauli_decompose
from .util import (MAG_THRESHOLD, otimes, ket, projector, fidelity,
                   process_fidelity, density_fidelity, relative_phase,
                   binary_to_index, index_to_binary, sampled_pulse)

__all__ = [
    "quiet", "no_truncation_check", "TruncationWarning", "truncation_populations",
    "Operator", "Mechanism", "Model", "term", "jump", "hconj",
    "HamiltonianEvolution", "UnitaryEvolution", "DensityMatrixEvolution",
    "LindbladEvolution", "embed", "partial_trace", "compare_over",
    "magnus", "magnus_pauli", "pauli_decompose",
    "MAG_THRESHOLD", "otimes", "ket", "projector", "fidelity",
    "process_fidelity", "density_fidelity", "relative_phase",
    "binary_to_index", "index_to_binary", "sampled_pulse",
]
