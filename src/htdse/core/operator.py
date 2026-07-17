import numpy as np


class Operator(np.ndarray):
    """The package's universal currency: any matrix or vector of the quantum
    mechanics. A Hamiltonian, a propagator, a density matrix, a ket are all
    Operators -- distinguished by *role*, not by type, with no shape
    enforcement. It IS an ndarray (a subclass), so numpy operations always
    just work on it.

    Beyond the array it carries a free-form `.params` metadata dict. Any
    derived array (view, slice, arithmetic like `H1 + H2`) gets a *shallow
    copy* of the source's params (numpy hands us the first operand), so
    derived arrays never share-and-mutate one dict. Treat params on derived
    arrays as provenance hints, not truth -- anything semantically important
    (subsystem structure, term decompositions) belongs in the term layer
    (core/terms.py), which propagates it correctly.
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
