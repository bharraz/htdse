"""Package-level runtime configuration.

The one knob here is solver verbosity. Every evolution class prints each real
integration it performs by default (nothing happens invisibly); `quiet()` turns
that off for a block of code -- the intended use is optimization loops that
construct thousands of evolutions:

    with htdse.quiet():
        result = scipy.optimize.minimize(cost, x0)

A per-instance `verbose=` argument on any evolution still overrides the global.
"""
from contextlib import contextmanager

VERBOSE = True  # global default; evolutions with verbose=None fall back to this


@contextmanager
def quiet():
    """Context manager: suppress all solver prints inside the block."""
    global VERBOSE
    previous = VERBOSE
    VERBOSE = False
    try:
        yield
    finally:
        VERBOSE = previous
