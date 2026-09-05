"""Immutable direct-unitary values."""
from types import MappingProxyType
from scipy import sparse as _sp
import numpy as np


class Unitary:
    """A callable immutable value representing a direct ``U(t)`` provider."""

    __slots__ = ("_fn", "_dim", "_subsystems", "_name", "_helpers")

    def __init__(self, fn, dim, subsystems=None, name="Unitary", **helpers):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_dim", int(dim))
        object.__setattr__(self, "_subsystems", MappingProxyType(dict(subsystems or {})))
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_helpers", MappingProxyType(dict(helpers)))

    def __setattr__(self, name, value):
        raise AttributeError("Unitary values are immutable")

    def __call__(self, t):
        return self._copy(self._fn(t))

    def unitary(self, t=None):
        if t is None:
            raise ValueError(f"{self._name}.unitary needs an explicit time t")
        return self._copy(self._fn(t))

    @staticmethod
    def _copy(value):
        if _sp.issparse(value):
            return value.copy()
        return np.array(value, copy=True)

    @property
    def dim(self):
        return self._dim

    @property
    def subsystems(self):
        return self._subsystems

    def __getattr__(self, name):
        try:
            return self._helpers[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self):
        return self._name
