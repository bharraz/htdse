"""Private helpers for objects that provide dynamics to the solver.

The public ``System`` value lives in :mod:`htdse.core.terms`. Other physics
providers are intentionally duck-typed and use this private capability module
without exposing a framework protocol or encouraging subclassing.
"""


def provides_unitary(system) -> bool:
    """Whether an internal dynamics provider supplies ``unitary``."""
    return callable(getattr(system, "unitary", None))


def provides_hamiltonian(system) -> bool:
    """Whether an internal dynamics provider supplies ``hamiltonian``."""
    return callable(getattr(system, "hamiltonian", None))
