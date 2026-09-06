"""Optional escape hatch to the QuTiP objects used by htdse internally.

Ordinary htdse code should use ``evolve`` and ``run`` and will continue to
receive NumPy arrays. These helpers are for QuTiP features outside htdse's
small public vocabulary, such as ``mcsolve`` or ``steadystate``.
"""
import numpy as np

from ..core.qutip_backend import compile_system, qobj


def to_qobj(array, subsystems=None):
    """Copy an array into a Qobj with the registry's tensor dimensions."""
    arr = np.asarray(array) if not hasattr(array, "shape") else array
    shape = arr.shape
    if subsystems:
        total = int(np.prod(list(subsystems.values())))
        if not shape or (shape[0] != total and shape[-1] != total):
            raise ValueError(
                f"registry {subsystems} multiplies to {total}, which matches no "
                f"axis of an array with shape {shape}")
    ket = len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)
    return qobj(arr, subsystems, ket=ket)


def to_qutip(system, include_jumps=True):
    """Compile a System through the production backend and expose QuTiP form."""
    return compile_system(system, include_jumps=include_jumps)
