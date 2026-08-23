"""Exact Lamb-Dicke sideband physics: the Lamb-Dicke parameter, the exact
sideband matrix element <n1|exp(i eta (a+a^dagger))|n2>, and thermally-averaged
sideband Rabi flopping -- ported from AMO.jl's `Trap` module
(https://github.com/yuyichao/AMO.jl), dropping only its Julia-performance
machinery (the `@generated` unrolled loops and 3-term Laguerre recurrence
become plain loops/scipy calls here -- same results, verified against AMO's
own test fixtures).

    <n1|exp(i eta (a+a^dagger))|n2>
        = exp(-eta^2/2) eta^Δn sqrt(n_-!/n_+!) L^Δn_{n_-}(eta^2)

with Δn = |n1-n2|, n_- = min(n1,n2), n_+ = max(n1,n2); Wineland & Itano,
PRA 20, 1521 (1979). Computed in log-space via scipy.special.gammaln, matching
wigner.py's stable-ratio precedent.

Convention note: `lamb_dicke` is the one function here that works in real SI
units (kg, Hz, m) -- it exists purely to turn real hardware numbers into the
dimensionless eta that everything else in htdse (and htdse's hbar=1
convention, see molmer_sorensen.py) then consumes. Nothing past that boundary
uses SI hbar.
"""
import itertools

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

_c = 299792458.0        # speed of light, m/s (exact, SI)
_h = 6.62607015e-34     # Planck constant, J s (exact, SI 2019)
_hbar = _h / (2 * np.pi)


def lamb_dicke(mass, *, fm=None, omega_m=None, kp=None, lambda_p=None, nu_p=None) -> float:
    """Lamb-Dicke parameter eta = sqrt(hbar/(2*mass*omega_m)) * kp for a
    motional mode of frequency fm (Hz) or omega_m (rad/s), addressed by a
    laser of wavevector kp (rad/m), wavelength lambda_p (m), or frequency
    nu_p (Hz). Exactly one of {fm, omega_m} and exactly one of
    {kp, lambda_p, nu_p} must be given; mass in kg."""
    freq_kwargs = {"fm": fm, "omega_m": omega_m}
    given_freq = [k for k, v in freq_kwargs.items() if v is not None]
    if len(given_freq) != 1:
        raise ValueError(f"lamb_dicke: exactly one of {list(freq_kwargs)} must be given")
    if fm is not None:
        omega_m = 2 * np.pi * fm

    k_kwargs = {"kp": kp, "lambda_p": lambda_p, "nu_p": nu_p}
    given_k = [k for k, v in k_kwargs.items() if v is not None]
    if len(given_k) != 1:
        raise ValueError(f"lamb_dicke: exactly one of {list(k_kwargs)} must be given")
    if lambda_p is not None:
        kp = 2 * np.pi / lambda_p
    elif nu_p is not None:
        kp = 2 * np.pi / _c * nu_p

    return float(np.sqrt(_hbar / (2 * mass * omega_m)) * kp)


def sideband(n1, n2, eta, phase=True):
    """Exact <n1|exp(i eta (a+a^dagger))|n2>. Returns a magnitude-only float
    if phase=False (as used internally by thermal_sideband), else the full
    complex value carrying the i^|n1-n2| phase of the displacement operator.
    0 for negative n1/n2; the n1==n2 Kronecker-delta limit is exact at eta=0."""
    if not (np.isclose(n1, round(n1)) and np.isclose(n2, round(n2))):
        raise ValueError(f"sideband: n1={n1}, n2={n2} must be (near-)integer Fock indices")
    n1, n2 = int(round(n1)), int(round(n2))
    if n1 < 0 or n2 < 0:
        return 0.0
    if eta == 0:
        return complex(n1 == n2) if phase else float(n1 == n2)

    n_minus, n_plus = min(n1, n2), max(n1, n2)
    delta_n = n_plus - n_minus
    eta2 = eta * eta
    if n1 == n2:
        log_pref = -eta2 / 2
    else:
        log_pref = (-eta2 + gammaln(n_minus + 1) - gammaln(n_plus + 1)) / 2 + delta_n * np.log(eta)
    magnitude = float(eval_genlaguerre(n_minus, delta_n, eta2) * np.exp(log_pref))

    if not phase:
        return magnitude
    return magnitude * 1j ** (delta_n % 4)


def sideband_series(delta_n, eta, phase=True, n_max=None):
    """Generator of sideband(n, n+|delta_n|, eta, phase) for n=0,1,2,...,
    up to and including n_max if given, else infinite. Plain loop over
    `sideband` -- AMO.jl's SidebandIter's 3-term Laguerre recurrence exists
    purely for Julia-side performance and isn't needed at htdse's scale."""
    delta_n = abs(delta_n)
    n = 0
    while n_max is None or n <= n_max:
        yield sideband(n, n + delta_n, eta, phase=phase)
        n += 1


def thermal_population_series(nbar, n_max) -> np.ndarray:
    """Thermal (Bose-Einstein) populations P(n) = nbar^n/(1+nbar)^(n+1) for
    n=0..n_max, matching AMO.jl's thermal_population/ThermalPopulationIter
    exactly. NOT the same thing as harmonic_oscillator.thermal(nbar,n_max):
    that function truncates AND renormalizes to build a valid density
    matrix (so its populations sum to 1 by construction, redistributing the
    truncated tail's weight onto the states you kept); this one is the raw,
    un-renormalized formula evaluated at each n -- the two disagree whenever
    n_max isn't large enough for the tail to be negligible, by design."""
    n = np.arange(n_max + 1, dtype=float)
    p0 = 1.0 / (nbar + 1.0)
    alpha = nbar * p0
    return p0 * alpha ** n


def thermal_sideband(nbars, delta_ns, etas, t, thresh=1e-3) -> float:
    """Thermally-averaged excitation probability for a (possibly multi-mode)
    sideband Rabi flopping experiment:

        P(t) = sum_{n1,n2,...} [prod_i P_thermal(nbar_i, n_i)]
                                * sin(t * prod_i sideband(n_i, n_i+delta_n_i, eta_i))^2

    i.e. qubit flopping on the delta_n-th sideband of each mode, averaged
    over an initial thermal distribution of phonons per mode. Scalar
    (nbar, delta_n, eta) is also accepted for the single-mode case.

    Each mode's infinite thermal sum over its starting Fock index n is
    truncated once its own population P(n) has decayed below
    thresh/n_modes -- same early-truncation criterion as AMO.jl's
    `thermal_sideband`, but summed directly over the un-reindexed n (relying
    on `sideband`'s n<0 -> 0 convention for the n+delta_n<0 terms), which is
    exactly the ground-truth form AMO.jl's own test suite checks it against
    -- simpler to verify than porting its negative-delta_n reindexing trick.
    """
    nbars = np.atleast_1d(np.asarray(nbars, dtype=float))
    delta_ns = np.atleast_1d(np.asarray(delta_ns, dtype=int))
    etas = np.atleast_1d(np.asarray(etas, dtype=float))
    if not (len(nbars) == len(delta_ns) == len(etas)):
        raise ValueError("thermal_sideband: nbars, delta_ns, etas must have the same length")
    n_modes = len(nbars)
    thresh_i = thresh / n_modes

    per_mode_pops, per_mode_omegas = [], []
    for nbar, delta_n, eta in zip(nbars, delta_ns, etas):
        if nbar <= 0:
            alpha, p0, n_max = 0.0, 1.0, 0
        else:
            p0 = 1.0 / (nbar + 1.0)
            alpha = nbar * p0
            n_max = max(int(np.ceil(np.log(thresh_i) / np.log(alpha))), 1)

        ns = np.arange(n_max + 1)
        per_mode_pops.append(p0 * alpha ** ns)
        per_mode_omegas.append(np.array([sideband(int(n), int(n) + int(delta_n), eta, phase=False)
                                         for n in ns]))

    total = 0.0
    for pops, omegas in zip(itertools.product(*per_mode_pops), itertools.product(*per_mode_omegas)):
        weight = np.prod(pops)
        omega_total = np.prod(omegas)
        total += weight * np.sin(t * omega_total) ** 2
    return float(total)
