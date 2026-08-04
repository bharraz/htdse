"""Package-level runtime configuration.

The one knob here is solver verbosity. Every evolution class prints each real
integration it performs by default (nothing happens invisibly); `quiet()` turns
that off for a block of code -- the intended use is optimization loops that
construct thousands of evolutions:

    with htdse.quiet():
        result = scipy.optimize.minimize(cost, x0)

A per-instance `verbose=` argument on any evolution still overrides the global.

The second knob is the Fock-truncation guard (see core/truncation.py): how much
population is allowed to reach the top level of a truncated subsystem before
the evolution warns. A per-instance `truncation=` argument overrides it.
"""
from contextlib import contextmanager

VERBOSE = True  # global default; evolutions with verbose=None fall back to this

# Population at the top level of a truncated ladder that triggers a
# TruncationWarning. 1e-6 is deliberately strict: at the point where the
# ceiling holds a millionth of the population it is already distorting
# high-Fock amplitudes, well before the effect is visible in a fidelity.
TRUNCATION_THRESHOLD = 1e-6  # None disables globally (see no_truncation_check)


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


@contextmanager
def no_truncation_check():
    """Context manager: disable the Fock-truncation guard inside the block.

    For optimizer inner loops where you have already established the
    truncation is adequate and the check is pure overhead. Note that `quiet()`
    does NOT silence truncation warnings -- a wrong answer is not chatter.
    """
    global TRUNCATION_THRESHOLD
    previous = TRUNCATION_THRESHOLD
    TRUNCATION_THRESHOLD = None
    try:
        yield
    finally:
        TRUNCATION_THRESHOLD = previous
