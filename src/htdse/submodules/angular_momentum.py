"""Clebsch-Gordan coefficients and Wigner 6j symbols, hand-rolled from the
standard Racah formula (Wikipedia "Clebsch-Gordan coefficients"/"6-j symbol",
Racah formula sections) -- no sympy/py3nj dependency.

Foundational for `atomic.py`'s angular-momentum recoupling (hyperfine/dipole
matrix elements); has no dependency on anything else in htdse. `atomic.py`
and `trap.py` are ports of AMO.jl's `Atomic`/`Trap` modules
(https://github.com/yuyichao/AMO.jl); this module has no direct Julia
source to port -- AMO.jl gets its Clebsch-Gordan/Wigner-6j from the external
`WignerSymbols.jl` package, so this is a from-scratch replacement for that
dependency, written specifically to support the port above.

Conventions (see MS_REBUILD-style plan note for the running log):
- Angular momenta and projections (j, m) are plain Python floats/ints
  (e.g. 0.5, 1.5) -- no Fraction/HalfInteger type.
- Both functions return 0.0 for physically-forbidden inputs (triangle
  inequality violated, m-conservation violated); they raise ValueError only
  for malformed input (non-half-integer j/m, |m| > j, negative j).
- All factorial arithmetic is done in log-space via scipy.special.gammaln,
  matching wigner.py's stable-ratio precedent -- never raw math.factorial.
"""
import numpy as np
from scipy.special import gammaln


def _half_int_ok(x, tol=1e-8) -> bool:
    return abs(2 * x - round(2 * x)) < tol


def _nonneg_int(x, tol=1e-8) -> bool:
    r = round(x)
    return r >= -tol and abs(x - r) < tol


def _triangle_ok(a, b, c) -> bool:
    return _nonneg_int(a + b - c) and _nonneg_int(a - b + c) and _nonneg_int(-a + b + c)


def _delta_log(a, b, c) -> float:
    """log of Delta(a,b,c) = sqrt((a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!)."""
    return 0.5 * (gammaln(a + b - c + 1) + gammaln(a - b + c + 1) + gammaln(-a + b + c + 1)
                  - gammaln(a + b + c + 2))


def clebsch_gordan(j1, m1, j2, m2, j3, m3) -> float:
    """<j1 m1; j2 m2 | j3 m3>, via the Racah formula. 0.0 if m1+m2 != m3 or
    the triangle inequality |j1-j2| <= j3 <= j1+j2 fails; ValueError for
    malformed (j, m) pairs."""
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        if (j < -1e-8 or not _half_int_ok(j) or not _half_int_ok(m) or abs(m) > j + 1e-8
                or not _nonneg_int(abs(j - m))):
            raise ValueError(f"invalid angular momentum/projection: j={j}, m={m}")
    if abs(m1 + m2 - m3) > 1e-8:
        return 0.0
    if not _triangle_ok(j1, j2, j3):
        return 0.0

    log_pref = (0.5 * np.log(2 * j3 + 1) + _delta_log(j1, j2, j3)
                + 0.5 * (gammaln(j1 + m1 + 1) + gammaln(j1 - m1 + 1)
                        + gammaln(j2 + m2 + 1) + gammaln(j2 - m2 + 1)
                        + gammaln(j3 + m3 + 1) + gammaln(j3 - m3 + 1)))

    kmin = int(round(max(0, j2 - j3 - m1, j1 - j3 + m2)))
    kmax = int(round(min(j1 + j2 - j3, j1 - m1, j2 + m2)))

    total = 0.0
    for k in range(kmin, kmax + 1):
        denom_log = (gammaln(k + 1) + gammaln(j1 + j2 - j3 - k + 1) + gammaln(j1 - m1 - k + 1)
                     + gammaln(j2 + m2 - k + 1) + gammaln(j3 - j2 + m1 + k + 1)
                     + gammaln(j3 - j1 - m2 + k + 1))
        total += (-1) ** k * np.exp(log_pref - denom_log)
    return float(total)


def wigner_6j(j1, j2, j3, J1, J2, J3) -> float:
    """{j1 j2 j3; J1 J2 J3}, via the Racah formula. 0.0 if any of the four
    triads (j1,j2,j3), (j1,J2,J3), (J1,j2,J3), (J1,J2,j3) fails the triangle
    inequality; ValueError for malformed (negative or non-half-integer) j."""
    js = (j1, j2, j3, J1, J2, J3)
    for j in js:
        if j < -1e-8 or not _half_int_ok(j):
            raise ValueError(f"invalid angular momentum: j={j}")
    triads = [(j1, j2, j3), (j1, J2, J3), (J1, j2, J3), (J1, J2, j3)]
    if not all(_triangle_ok(*t) for t in triads):
        return 0.0

    log_pref = sum(_delta_log(*t) for t in triads)

    tmin = int(round(max(a + b + c for a, b, c in triads)))
    tmax = int(round(min(j1 + j2 + J1 + J2, j2 + j3 + J2 + J3, j3 + j1 + J3 + J1)))

    total = 0.0
    for t in range(tmin, tmax + 1):
        num_log = gammaln(t + 2)
        denom_log = (gammaln(t - j1 - j2 - j3 + 1) + gammaln(t - j1 - J2 - J3 + 1)
                     + gammaln(t - J1 - j2 - J3 + 1) + gammaln(t - J1 - J2 - j3 + 1)
                     + gammaln(j1 + j2 + J1 + J2 - t + 1) + gammaln(j2 + j3 + J2 + J3 - t + 1)
                     + gammaln(j3 + j1 + J3 + J1 - t + 1))
        total += (-1) ** t * np.exp(num_log - denom_log)
    return float(np.exp(log_pref) * total)
