import numpy as np


class Operator(np.ndarray):
    """ndarray subclass carrying a free-form `.params` metadata dict.

    No shape enforcement -- an Operator may be a Hamiltonian, unitary, density
    matrix, or state vector. Domain-specific subclasses (spin, motional, ...)
    extend this with their own reserved params keys and constructors.

    Metadata propagation rule: any array derived from an Operator (a view, a
    slice, or the result of arithmetic like `H1 + H2`) gets a *shallow copy*
    of the source's params -- for arithmetic with two Operators, numpy hands us
    the first operand, so the result carries a copy of `H1.params`. The copy
    matters: without it every derived array would share (and silently mutate)
    one dict. Treat params on derived arrays as provenance hints, not truth --
    anything semantically important (subsystem structure, term decompositions)
    belongs in the term layer (core/terms.py), which propagates it correctly.
    """

    def __new__(cls, input_array, params=None):
        obj = np.asarray(input_array).view(cls)  # reinterpret, no copy
        obj.params = params if params is not None else {}
        return obj

    def __array_finalize__(self, obj):
        # called on view/slice/ufunc result; carry a COPY of params forward so
        # derived arrays never share (and mutate) the source's dict
        if obj is None:
            return
        self.params = dict(getattr(obj, "params", {}))

    def __repr__(self):
        return f"{super().__repr__()}\nwith params={self.params}"
