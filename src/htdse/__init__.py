"""htdse -- target-vs-reality Hamiltonian simulation.

Everything a physicist reaches for interactively is importable from the top --
the generic engine (Model/term/the Evolution classes) AND the physics vocabulary
(Paulis, ladder operators, the spin-boson/Mølmer–Sørensen builders):

    import htdse as ht
    H = ht.term(0.5 * ht.sigma_z, on="q") + ht.term(0.5 * ht.sigma_x, on="q")
    psi = ht.evolve(H, ht.ket("0"), t=np.pi)

Each physics submodule is also reachable by name (`ht.spin`, `ht.spin_boson`,
`ht.molmer_sorensen`, `ht.harmonic_oscillator`, `ht.trapped_ion`, `ht.trotter`,
`ht.wigner`) for anything not flattened here -- `ht.wigner.wigner(...)` in
particular, since a bare `wigner` name would collide with its own module.
"""
from .core.config import quiet, no_truncation_check
from .core.truncation import TruncationWarning, truncation_populations
from .core.system import System
from .core.terms import Model, term, jump, plus_hc, hc, SparseSuggestion
from .core.evolution import (HamiltonianEvolution, UnitaryEvolution,
                             DensityMatrixEvolution, LindbladEvolution,
                             evolve, propagator)
from .core.subsystems import embed, partial_trace
from .core.compare import compare_over
from .core.convergence import converged
from .core.read import show, project, closure, generator, paulis, max_eigenphase, expect
from .core.plotting import (plot_populations, plot_eigenspectrum, plot_matrix,
                            plot_phases, plot_adiabatic_populations)
from .magnus import magnus, magnus_pauli, pauli_decompose
from .util import (MAG_THRESHOLD, dag, otimes, ket, bra, projector, fidelity,
                   process_fidelity, density_fidelity, relative_phase,
                   binary_to_index, index_to_binary, sampled_pulse)

# The physics vocabulary. Each submodule is reachable by name (`ht.spin`, ...)
# via these very imports; the specific names below are flattened on top of
# that because they are what gets typed constantly. `submodules/` has no
# privileged access to `core/` -- this is just importing its public API.
from .submodules import (spin, harmonic_oscillator, spin_boson, trapped_ion,
                         molmer_sorensen, trotter, wigner)
from .submodules.spin import (sigma_x, sigma_y, sigma_z, I2, sigma_plus,
                              sigma_minus, PAULIS, pauli_term, pauli_sum)
from .submodules.harmonic_oscillator import (annihilation, creation, number_operator,
                                             ladder_operators, fock, thermal,
                                             ThermalMotionalDecoherence)
from .submodules.spin_boson import (Tone, Mode, driven_spins, jaynes_cummings,
                                    exact_drive, rabi, explain, TONE_TABLE)
from .submodules.trapped_ion import IonChain
from .submodules.molmer_sorensen import (ms_tones, ms_closed_form, ideal_gate,
                                         expectation_alpha, plot_phase_space)
from .submodules.trotter import TrotterizedSystem

__all__ = [
    "quiet", "no_truncation_check", "TruncationWarning", "truncation_populations",
    "System", "Model", "term", "jump", "plus_hc", "hc", "SparseSuggestion",
    "HamiltonianEvolution", "UnitaryEvolution", "DensityMatrixEvolution",
    "LindbladEvolution", "evolve", "propagator",
    "embed", "partial_trace", "compare_over", "converged",
    "magnus", "magnus_pauli", "pauli_decompose",
    "MAG_THRESHOLD", "dag", "otimes", "ket", "bra", "projector", "fidelity",
    "process_fidelity", "density_fidelity", "relative_phase",
    "binary_to_index", "index_to_binary", "sampled_pulse",
    "show", "project", "closure", "generator", "paulis", "max_eigenphase", "expect",
    "plot_populations", "plot_eigenspectrum", "plot_matrix",
    "plot_phases", "plot_adiabatic_populations",
    # submodules, reachable by name
    "spin", "harmonic_oscillator", "spin_boson", "trapped_ion",
    "molmer_sorensen", "trotter", "wigner",
    # the physics vocabulary, flattened
    "sigma_x", "sigma_y", "sigma_z", "I2", "sigma_plus", "sigma_minus",
    "PAULIS", "pauli_term", "pauli_sum",
    "annihilation", "creation", "number_operator", "ladder_operators", "fock",
    "thermal", "ThermalMotionalDecoherence",
    "Tone", "Mode", "driven_spins", "jaynes_cummings", "exact_drive", "rabi",
    "explain", "TONE_TABLE", "IonChain",
    "ms_tones", "ms_closed_form", "ideal_gate", "expectation_alpha", "plot_phase_space",
    "TrotterizedSystem",
]
