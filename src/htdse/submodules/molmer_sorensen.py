"""Molmer-Sorensen mechanism suite: bichromatic spin-dependent force on N spins
coupled to ANY NUMBER of motional modes, at three levels of approximation.

Conventions follow C. Monroe, "Primer on Molmer-Sorensen Gates in Trapped Ions"
(2021); equation numbers below refer to it. hbar = 1 throughout.

TWO PHASES, and the difference between them is the whole point
---------------------------------------------------------------
The bichromatic drive puts two tones at w0 +- mu with optical phases dphi_B and
dphi_R. Their sum and difference are the two physically distinct knobs:

    spin phase    theta_i = (dphi_B + dphi_R)/2      -- the sigma axis in the XY plane
    motion phase  psi_i   = (dphi_B - dphi_R)/2      -- the phase of the optical force

Writing dphi_B = theta + psi, dphi_R = theta - psi, the two tones combine into

    H_i = Omega_i(t) cos(mu_i t + psi_i) [ sigma_+ e^{-i theta_i} e^{i dk X_i} + h.c. ]

so psi enters ONLY as cos(mu t) -> cos(mu t + psi). Expanding e^{i dk X} to first
order in the Lamb-Dicke parameters, i(sigma_+ e^{-i theta} - sigma_- e^{i theta})
= -sigma_{theta + pi/2}, which is where the leading minus and the pi/2 below come
from. In this module `phases` IS Monroe's theta, and Phi_j = phi_j + pi/2.

WHY psi MATTERS: alpha carries e^{-i psi} (Eq. 26) but the spin-spin coupling
chi does NOT (Eq. 27) -- "the motion phase does not play a role in the effective
spin-spin interactions". So psi rotates the phase-space displacement while
leaving the entangling phase untouched: it is the knob for cancelling residual
motional excitation without disturbing the gate.

ARGUMENTS
    modes         : list of `MSMode` -- the general (multi-mode) form. Each mode
                    carries its own eta_{i,m}, detune delta_{i,m}, n_max, nu.
    participation,
    eta, detune,
    n_max, nu     : the single-mode shorthand (eta_{i} = eta * b_i). Pass these
                    OR `modes=`, never both.
    amplitudes    : per-ion Rabi amplitude Omega_i -- scalar or callable f(t)
    phases        : per-ion SPIN phase theta_i (Monroe's theta)
    motion_phases : per-ion MOTION phase psi_i (Monroe's psi). Default 0.

POST-RWA spin-dependent force (Lamb-Dicke order 1, dropping mu / 2nu terms) --
Monroe Eq. 20:

    H_RWA(t) = sum_{i,m} sigma_{Phi_i}^{(i)} (x) ( f_{i,m}(t) a_m^dag + c.c. a_m )
    f_{i,m}(t) = -(eta_{i,m} Omega_i(t) / 2) e^{-i (delta_{i,m} t + psi_i)}

Because all sigma_{Phi_i} commute and different modes commute, [H(t1), H(t2)] is
a pure (scalar) spin operator and the Magnus series TERMINATES at second order
(Eq. 23) -- `MSMagnus` implements the resulting exact unitary (Eq. 24):

    U(t) = exp( sum_{i,m} sigma_{Phi_i} (alpha_{i,m} a_m^dag - alpha_{i,m}* a_m)
                + i sum_{ij} Theta_ij sigma_{Phi_i} sigma_{Phi_j} )

    alpha_{i,m}(t) = -i int_0^t f_{i,m}                             (Eq. 25/26)
    Theta_ij(t)    = sum_m int_0^t dt1 int_0^t1 dt2 Im[f_{i,m}(t1) f_{j,m}(t2)*]

Theta is this package's geometric phase; it SUMS OVER MODES (Eq. 27). The
entangling angle between ions i != j is Theta_ij + Theta_ji, which is Monroe's
chi_ij up to his sign/factor convention (see `entangling_angle`).

PRE-RWA (what `ms_lamb_dicke1/2` build -- "stop after the Lamb-Dicke expansion",
keeping the off-resonant carrier and counter-rotating terms):

    H(t) = sum_i Omega_i(t) cos(mu_i t + psi_i) [
             sigma_{phi_i}                                              (carrier, eta^0)
             - sum_m eta_{i,m} sigma_{Phi_i} X_m(t)                     (eta^1)
             - (1/2) sum_m eta_{i,m}^2 sigma_{phi_i} X_m(t)^2 ]         (eta^2)

    X_m(t)   = a_m e^{-i nu_m t} + a_m^dag e^{+i nu_m t}
    X_m(t)^2 = a_m^2 e^{-2i nu_m t} + a_m^dag^2 e^{2i nu_m t} + 2 n_m + 1

Applying the mode RWA to the eta^1 line reproduces H_RWA above (cross-validated
in tests against MSMagnus). A single bichromatic beat means mu_i = nu_m +
delta_{i,m} must be the SAME for every mode m -- the builders check this.

NOTE on eta^2 with several modes: the full expansion of prod_m e^{i eta_m X_m}
also contains CROSS-mode terms -eta_{i,m} eta_{i,m'} X_m X_m' (m < m'). Those are
not implemented, so `ms_lamb_dicke2` refuses more than one mode rather than
silently dropping them.

The builders return term-layer `Model`s with named groups per ion and mode
("carrier_q0", "sdf_q0_mode0", "ld2_q0", ...), so error injection is composition:

    H = ms_lamb_dicke1(...) + pauli_term("Z0", coeff=eps_z)     # static sigma_z error
    H_err = H.replace(sdf_q0_mode0=my_miscalibrated_drive)      # swap one ion's drive
"""
import cmath
import math
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm

from ..core.mechanism import Mechanism
from ..core.operator import Operator
from ..core.subsystems import embed
from ..core.terms import Model, hconj, term
from .harmonic_oscillator import annihilation
from .spin import sigma_x, sigma_y


# ---------------------------------------------------------------------------
# argument normalization
# ---------------------------------------------------------------------------

def _real(xi, what):
    """A real scalar, or a clear refusal. Taking `.real` silently would discard
    a phase the user encoded in a complex amplitude."""
    z = complex(xi)
    if z.imag != 0:
        raise ValueError(f"{what}: got complex value {xi!r}, but these are real "
                         f"quantities -- encode the phase via `phases=` instead.")
    return z.real


def _as_funcs(x, n, what):
    """Normalize `x` (scalar | callable | sequence of either, length n) into a
    list of n callables f(t)."""
    if np.isscalar(x) or callable(x):
        x = [x] * n
    if len(x) != n:
        raise ValueError(f"{what}: expected {n} entries (one per ion), got {len(x)}")
    return [xi if callable(xi) else (lambda t, c=_real(xi, what): c) for xi in x]


def _as_consts(x, n, what):
    """Like _as_funcs but requires plain numbers (no callables)."""
    if np.isscalar(x):
        x = [x] * n
    if any(callable(xi) for xi in x):
        raise ValueError(f"{what} must be constants here (see class docstring)")
    if len(x) != n:
        raise ValueError(f"{what}: expected {n} entries (one per ion), got {len(x)}")
    return [float(xi) for xi in x]


def _memo1(fn):
    """One-slot memo on the argument. Every term of one ion is evaluated at the
    SAME t inside a single H(t) call, so the shared drive factor
    Omega(t) cos(mu t + psi(t)) is computed once per call instead of once per
    term. `t` is a plain float here (the solver never passes arrays to a
    coefficient), so `==` is the right test."""
    last_t, last_v = [None], [None]

    def g(t):
        if t != last_t[0]:
            last_t[0], last_v[0] = t, fn(t)
        return last_v[0]
    return g


def _const_of(x):
    """The plain number `x` is, or None if it is a callable (time-dependent)."""
    return None if callable(x) else float(np.real(x))


def _trig(fn, const):
    """t -> (cos, sin) of a phase, constant-folded when the phase is constant.
    math.* not np.* : these run once per term per ODE step, and numpy's scalar
    path is ~10x slower than the C library's."""
    if const is not None:
        c, s = math.cos(const), math.sin(const)
        return lambda t: (c, s)
    return lambda t: (math.cos(fn(t)), math.sin(fn(t)))


def _eval(fn, t):
    """Evaluate a coefficient callable at a scalar or array time.

    Fast path first: most drive parameters are constants or numpy-aware
    expressions, and both evaluate on the whole grid in one call. `np.vectorize`
    is a Python-level loop over every grid point (tens of thousands of them per
    Magnus quadrature), so it is the LAST resort, not the first."""
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return fn(float(t))
    try:
        out = np.asarray(fn(t))
    except Exception:
        return np.vectorize(fn)(t)   # genuinely scalar-only (e.g. branches on t)
    if out.shape == t.shape:
        return out                   # numpy-aware callable
    if out.ndim == 0:
        return np.broadcast_to(out, t.shape)   # constant
    return np.vectorize(fn)(t)       # returned something unexpected: be safe


def _sigma(phi):
    """sigma_phi = cos(phi) sx + sin(phi) sy (Monroe Eq. 13)."""
    return np.cos(phi) * sigma_x + np.sin(phi) * sigma_y


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

class MSMode(NamedTuple):
    """One motional mode, as the MS drive sees it.

    eta    : eta_{i,m} per ion -- the Lamb-Dicke coupling of each ion to THIS
             mode, i.e. b_{i,m} * dk * x_m0 already folded together (Monroe:
             eta_{i,m} = b_{i,m} dk sqrt(hbar/2 M w_m)). Scalar broadcasts.
             SAME for both tones -- the coupling is geometric, not optical.
    detune : delta_{i,m} = mu_i - nu_m, per ion. Scalar broadcasts. The BLUE
             tone's detuning (or the shared detuning, when `detune_red` is
             None). The loop of a constant-amplitude symmetric drive closes at
             T = 2 pi / |delta|.
    n_max  : Fock truncation of this mode.
    nu     : trap frequency of this mode (pre-RWA builders only; mu = nu + delta).
    name   : subsystem name. Must be unique across modes.
    detune_red : RED tone's detuning, per ion, if different from `detune` --
             an ASYMMETRIC bichromatic drive (both tones shifted by different
             amounts from their sidebands). None (default): red uses `detune`
             too, i.e. today's symmetric drive, unchanged. Only `ms_lamb_dicke1`
             /`ms_lamb_dicke2` (real H(t), ODE-solved) support this -- NOT
             `MSMagnus`, whose closed form relies on a single, time-independent
             sigma_Phi axis per ion; an asymmetric eta^1 term has a ROTATING
             axis (see `_tone_group`), so the Magnus-terminates argument does
             not obviously carry over and this module does not claim it does.
    """
    eta: object
    detune: object
    n_max: int
    nu: object = None
    name: str = "mode"
    detune_red: object = None


def _norm_mode(md, n_ions) -> MSMode:
    """Broadcast one mode's eta/detune(/detune_red) to per-ion arrays."""
    eta = np.broadcast_to(np.asarray(md.eta, dtype=float), (n_ions,)).copy()
    detune = np.broadcast_to(np.asarray(md.detune, dtype=float), (n_ions,)).copy()
    detune_red = (None if md.detune_red is None else
                 np.broadcast_to(np.asarray(md.detune_red, dtype=float), (n_ions,)).copy())
    return MSMode(eta, detune, int(md.n_max),
                  None if md.nu is None else float(md.nu), str(md.name), detune_red)


def _mode_list(modes, participation, eta, detune, n_max, nu, mode_name,
               amplitudes, phases, detune_red=None) -> list:
    """Normalize either `modes=` or the single-mode shorthand into a list of
    per-ion-broadcast MSMode. Exactly one of the two forms must be given.

    detune_red: only meaningful for the single-mode shorthand (asymmetric
    bichromatic drive); for `modes=`, set `.detune_red` on each MSMode instead."""
    legacy = [participation, eta, detune, n_max]
    if modes is not None:
        if any(v is not None for v in legacy):
            raise ValueError(
                "pass either `modes=` or the single-mode shorthand "
                "(participation, eta, detune, n_max) -- not both")
        if detune_red is not None:
            raise ValueError("detune_red is only for the single-mode shorthand -- "
                             "set `.detune_red` on each MSMode for `modes=`")
        modes = list(modes)
        if not modes:
            raise ValueError("`modes=` is empty: an MS drive needs at least one mode")
        n_ions = _infer_n_ions(modes, amplitudes, phases)
        out = [_norm_mode(md, n_ions) for md in modes]
    else:
        if any(v is None for v in legacy):
            raise ValueError(
                "single-mode form needs participation, eta, detune and n_max "
                "(or use `modes=[MSMode(...), ...]` for the multi-mode form)")
        b = np.atleast_1d(np.asarray(participation, dtype=float))
        n_ions = len(b)
        # eta_{i} = eta * b_i -- the participation folds into the coupling
        out = [_norm_mode(MSMode(float(eta) * b, float(detune), int(n_max),
                                 nu, mode_name, detune_red), n_ions)]
    names = [md.name for md in out]
    if len(set(names)) != len(names):
        raise ValueError(f"mode names must be unique, got {names}")
    return out, n_ions


def _infer_n_ions(modes, amplitudes, phases) -> int:
    """Number of ions, from whichever argument actually carries it."""
    for md in modes:
        arr = np.atleast_1d(np.asarray(md.eta, dtype=float))
        if arr.size > 1:
            return arr.size
    for x in (phases, amplitudes):
        if not np.isscalar(x) and not callable(x) and x is not None:
            return len(x)
    for md in modes:  # every eta scalar and no per-ion list anywhere: single ion
        return 1
    raise ValueError("cannot infer the number of ions -- give per-ion `eta` "
                     "arrays on the modes, or a per-ion `phases` list")


def _beat(modes, n_ions, tone="detune"):
    """mu_i = nu_m + delta_{i,m} for the named tone ("detune" = blue/shared,
    "detune_red" = red). A single tone drives every mode at once, so this must
    not depend on m -- disagreement means the mode table and the detunings
    describe different lasers, which is a setup error, not physics."""
    if any(md.nu is None for md in modes):
        raise ValueError("pre-RWA builders need the trap frequency `nu` on every mode")
    mus = np.array([md.nu + getattr(md, tone) for md in modes])   # (n_modes, n_ions)
    mu = mus[0]
    if not np.allclose(mus, mu, rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(mu)))):
        raise ValueError(
            f"inconsistent bichromatic beat ({tone}): nu_m + delta_{{i,m}} must be "
            f"the same for every mode (one drive, one mu), got mu per mode =\n{mus}")
    return mu


# ---------------------------------------------------------------------------
# 1) ideal / closed-form: the terminated Magnus expansion
# ---------------------------------------------------------------------------

class MSMagnus(Mechanism):
    """Exact (terminated-Magnus) unitary of the post-RWA MS spin-dependent
    force. Defined as a GATE (`.unitary(t)` only) -- `UnitaryEvolution` and
    `DensityMatrixEvolution` consume it directly with no ODE solve.

    Restrictions inherent to the closed form: `phases` must be CONSTANT per
    ion (time-dependent spin phase breaks the commutator structure that
    terminates the Magnus series -- use the ms_lamb_dicke builders for that),
    and t >= 0. Time-dependent amplitudes are fine.

    Helpers: `alpha(t)` / `alpha_trajectory(ts)` (per-ion, per-mode phase-space
    trajectory), `geometric_phase(t)` (Theta_jk, summed over modes) and
    `entangling_angle(t)`. Integrals are computed by (cumulative) trapezoid on a
    dense grid with `points_per_period` points per period of the FASTEST detuning
    (default 400; refine for very fast amplitude modulation).

    Truncation caveat: the closed form is exact for the INFINITE-dimensional
    oscillator ([a, a^dag] = 1); here it is evaluated with truncated
    operators. An ODE solve of the truncated H(t) instead propagates the
    truncated model exactly. The two agree on any state far from the Fock
    edge (choose n_max well above max |alpha|^2 + initial occupation),
    but differ on states near |n_max> -- so compare via low-Fock states, not
    full-space process fidelity.
    """

    def __init__(self, participation=None, eta=None, detune=None, amplitudes=1.0,
                 phases=0.0, n_max=None, points_per_period=400, *,
                 modes=None, motion_phases=0.0):
        self.modes, self.n_ions = _mode_list(modes, participation, eta, detune,
                                             n_max, None, "mode", amplitudes, phases)
        self.n_modes = len(self.modes)
        self.amplitudes = _as_funcs(amplitudes, self.n_ions, "amplitudes")
        # SPIN phase must be constant: it sets sigma_{Phi_j}, and a time-dependent
        # sigma breaks the commutator structure that terminates the Magnus series.
        self.phases = _as_consts(phases, self.n_ions, "phases (MSMagnus)")
        # MOTION phase may be time-dependent: it enters only through f_{j,m}(t),
        # never through the spin operators, so [H(t1), H(t2)] stays a pure spin
        # operator and the series still terminates.
        self.motion_phases = _as_funcs(motion_phases, self.n_ions, "motion_phases")
        self.points_per_period = int(points_per_period)
        self.subsystems = {f"q{j}": 2 for j in range(self.n_ions)}
        for md in self.modes:
            self.subsystems[md.name] = md.n_max + 1
        self._a = [annihilation(md.n_max) for md in self.modes]
        # sigma_{Phi_j} with Phi = phi + pi/2, embedded on the joint space -- built
        # LAZILY (see the `_S` property below), not here: this embed is (dim, dim)
        # per ion, and only `.unitary()` ever needs it. `alpha`/`geometric_phase`/
        # `entangling_angle` do not, so a many-ion many-mode chain used only for
        # those must not pay for (or fail to allocate) an embed it never asked for.
        self._S_cache = None
        self._frozen = True  # no attribute may change from here on; see __setattr__

    @property
    def _S(self):
        if self._S_cache is None:
            S = [np.asarray(embed(_sigma(phi + np.pi / 2), self.subsystems, f"q{j}"))
                for j, phi in enumerate(self.phases)]
            # bypass the frozen-after-init guard: this fills a cache, it does not
            # change any physics parameter the guard is protecting.
            object.__setattr__(self, "_S_cache", S)
        return self._S_cache

    def __setattr__(self, name, value):
        """Frozen after construction. `_a`, `subsystems` and `_S` are derived
        from the mode table and `phases` at __init__, while `f()` reads
        `amplitudes` live -- so a post-hoc attribute assignment would leave the
        two halves of the same gate disagreeing about the parameters. Keying
        each derived cache on its inputs is whack-a-mole; this is the rule the
        class docstring already claims."""
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"MSMagnus is frozen after construction (cannot set {name!r}). "
                "Its operators are derived from the constructor arguments; "
                "build a new MSMagnus instead of mutating this one.")
        object.__setattr__(self, name, value)

    # -- drive kernel -------------------------------------------------------

    def f(self, j, m, t):
        """f_{j,m}(t) = -(eta_{j,m} Omega_j(t)/2) e^{-i(delta_{j,m} t + psi_j)},
        the coefficient of a_m^dag for ion j (Monroe Eq. 20). The motion phase
        psi_j enters here and ONLY here."""
        md = self.modes[m]
        t = np.asarray(t, dtype=float)
        Om = _eval(self.amplitudes[j], t)
        psi = _eval(self.motion_phases[j], t)
        return -(md.eta[j] * Om / 2) * np.exp(-1j * (md.detune[j] * t + psi))

    def _grid(self, t):
        """Dense grid resolving the FASTEST detuning present (multi-mode: the
        modes far from the beat oscillate quickest and set the quadrature cost)."""
        fastest = max((abs(d) for md in self.modes for d in md.detune), default=0.0)
        period = 2 * np.pi / max(fastest, 1e-12)
        n = int(max(2001, self.points_per_period * (t / period + 1)))
        return np.linspace(0.0, t, n)

    # -- Magnus ingredients ---------------------------------------------------

    def alpha(self, t) -> np.ndarray:
        """Phase-space displacement at time t: alpha_{j,m}(t) = -i int_0^t f_{j,m}.
        Complex array (n_ions, n_modes). The displacement of mode m conditioned
        on the sigma_{Phi_j} = +1 branch is +alpha_{j,m} (and -alpha on the -1
        branch). Carries e^{-i psi_j} -- Monroe Eq. 26."""
        return self._magnus(float(t))[0]

    def alpha_trajectory(self, ts) -> np.ndarray:
        """alpha_{j,m} at every time in `ts` (ascending, ts[0] >= 0): complex
        array (len(ts), n_ions, n_modes). Feed straight into `plot_phase_space`."""
        ts = np.asarray(ts, dtype=float)
        if ts[0] < 0:
            raise ValueError("MSMagnus is defined for t >= 0")
        # `ts` may serve as the quadrature grid only if it already resolves the
        # integral from its lower limit 0 -- otherwise int_0^t f would silently
        # start at ts[0] and every alpha would be off by the missing int_0^ts[0].
        reuse_ts = (len(ts) > 1000 and ts[0] == 0.0
                    and np.allclose(np.diff(ts), ts[1] - ts[0]))
        grid = ts if reuse_ts else np.union1d(self._grid(ts.max()), ts)
        F = self._kernel(grid)                                  # (N, M, n)
        alpha = -1j * _cumtrapz(F, grid)                        # (N, M, n)
        idx = np.searchsorted(grid, ts)
        return np.moveaxis(alpha[:, :, idx], -1, 0)             # (n_times, N, M)

    def geometric_phase(self, t) -> np.ndarray:
        """Theta_jk(t) = sum_m int_0^t dt1 int_0^t1 dt2 Im[f_{j,m}(t1) f_{k,m}(t2)*]
        -- the accumulated two-spin (and j = k: global) phase matrix
        (n_ions, n_ions), SUMMED OVER MODES (Monroe Eq. 27)."""
        return self._magnus(float(t))[1]

    def entangling_angle(self, t) -> np.ndarray:
        """Theta_jk + Theta_kj -- the two-spin angle actually multiplying
        sigma_{Phi_j} sigma_{Phi_k} in U (the j != k entries). This is Monroe's
        chi_jk up to his sign/factor convention; a maximally entangling gate is
        |angle| = pi/4."""
        Th = self.geometric_phase(t)
        return Th + Th.T

    def _kernel(self, grid) -> np.ndarray:
        """f_{j,m} on `grid`: complex array (n_ions, n_modes, len(grid))."""
        return np.array([[self.f(j, m, grid) for m in range(self.n_modes)]
                         for j in range(self.n_ions)])

    def _magnus(self, t):
        """(alpha, Theta) at time t from ONE quadrature grid -- both Magnus
        ingredients are built from the same int_0^t1 f, so `unitary()` does
        not grid and integrate f twice.

        alpha: (N, M). Theta: (N, N), summed over modes."""
        grid = self._grid(t)
        F = self._kernel(grid)                                   # (N, M, n)
        Fint = _cumtrapz(F, grid)                                # int_0^t1 f  (N, M, n)
        alpha = -1j * Fint[:, :, -1]                             # (N, M)
        # Im[f_{j,m}(t1) F_{k,m}(t1)*], contracted over the mode axis m
        integrand = np.einsum("jmn,kmn->jkn", F, Fint.conj()).imag
        Theta = np.trapezoid(integrand, grid, axis=-1)           # (N, N)
        return alpha, Theta

    # -- the gate -------------------------------------------------------------

    def unitary(self, t=None) -> Operator:
        """U(t) = exp(Omega1 + Omega2): spin-dependent displacement (summed over
        modes) times the geometric-phase gate. Different modes' displacements and
        all the spin operators commute, so one expm of the sum is exact."""
        if t is None:
            raise ValueError("MSMagnus.unitary needs an explicit time t")
        t = float(t)
        if t < 0:
            raise ValueError("MSMagnus is defined for t >= 0")
        alpha, Theta = self._magnus(t)
        D = np.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.n_ions):
            for m, md in enumerate(self.modes):
                a = self._a[m]
                Mjm = np.asarray(embed(alpha[j, m] * a.conj().T
                                       - np.conj(alpha[j, m]) * a,
                                       self.subsystems, md.name))
                D = D + self._S[j] @ Mjm      # sigma_Phi_j (alpha a_m^dag - alpha* a_m)
        G = np.zeros_like(D)
        for j in range(self.n_ions):
            for k in range(self.n_ions):
                G = G + 1j * Theta[j, k] * (self._S[j] @ self._S[k])
        return Operator(expm(D + G))

    @property
    def dim(self):
        d = 2 ** self.n_ions
        for md in self.modes:
            d *= md.n_max + 1
        return d

    def __repr__(self):
        modes = ", ".join(f"{md.name}(n_max={md.n_max})" for md in self.modes)
        return (f"MSMagnus(N={self.n_ions}, modes=[{modes}], "
                f"phases={self.phases}, motion_phases={self.motion_phases})")


def _cumtrapz(y, x):
    """Cumulative trapezoid along the last axis, with y[..., 0] -> 0."""
    dx = np.diff(x)
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    out = np.zeros_like(y)
    out[..., 1:] = np.cumsum(seg, axis=-1)
    return out


# ---------------------------------------------------------------------------
# one off-resonant TONE's full pre-RWA contribution -- the atomic building
# block an asymmetric (blue != red) bichromatic drive is assembled from
# ---------------------------------------------------------------------------

def _tone_group(H, q, j, modes, ops, mu_j, Om_fn, phi_fn, order, tag, sign=1):
    """Add ONE tone's carrier (+ eta^1, + eta^2) contribution for ion `q` to
    Model H. Derived directly from a single Monroe Eq. 7 off-resonant tone,
    Lamb-Dicke expanded:

        carrier ~  Omega(t) * sigma_{sign*mu_j*t + phi(t)}
        eta^1   ~ -(Omega(t) eta_m/2) e^{i nu_m t} sigma_{sign*mu_j*t + phi(t) + pi/2}
                  (coefficient of a_m^dag; +h.c. gives a_m)
        eta^2   ~ -(Omega(t) eta_m^2/4) sigma_{sign*mu_j*t + phi(t)}
                  * (a_m^2 e^{-2i nu_m t} + a_m^dag^2 e^{2i nu_m t} + 2 n_m + 1)

    `sign`: BLUE (Monroe's H ~ sigma_+ e^{-i(mu t+phi)}+h.c.) uses sign=+1; RED
    (H ~ sigma_+ e^{+i(mu t-phi)}+h.c.) uses sign=-1 -- this is a genuine
    asymmetry in how the two tones are conventionally written (not just a
    phase difference), re-derived directly rather than assumed. Getting this
    wrong is exactly the failure mode the symmetric-reduction regression test
    below is there to catch.

    Unlike the shared-mu (symmetric) formula, this tone's eta^1 axis ROTATES
    at rate mu_j -- it is NOT the constant sigma_Phi of the combined symmetric
    case. Two tones (blue at mu_+, sign=+1; red at mu_-, sign=-1) are summed
    via ordinary Model `+`; when mu_+ = mu_- and both tones share Omega and
    phi = theta+-psi, the rotating pieces cancel and this reduces EXACTLY to
    the existing combined formula (regression-tested). `tag` ("blue"/"red")
    keeps the two tones' groups distinct.
    """
    def cx(t, mu_j=mu_j, sign=sign, Om_fn=Om_fn, phi_fn=phi_fn):
        return (Om_fn(t) / 2) * math.cos(sign * mu_j * t + phi_fn(t))
    def cy(t, mu_j=mu_j, sign=sign, Om_fn=Om_fn, phi_fn=phi_fn):
        return (Om_fn(t) / 2) * math.sin(sign * mu_j * t + phi_fn(t))
    H = H + term(sigma_x, on=q, coeff=cx, name=f"carrier_{q}_{tag}") \
          + term(sigma_y, on=q, coeff=cy, name=f"carrier_{q}_{tag}")
    if order < 1:
        return H
    for md in modes:
        a, adag = ops[md.name]
        e_jm, nu_m = md.eta[j], md.nu
        def gx(t, mu_j=mu_j, sign=sign, e=e_jm, nu_m=nu_m, Om_fn=Om_fn, phi_fn=phi_fn):
            return -(e / 2) * Om_fn(t) * cmath.exp(1j * nu_m * t) \
                   * math.cos(sign * mu_j * t + phi_fn(t) + math.pi / 2)
        def gy(t, mu_j=mu_j, sign=sign, e=e_jm, nu_m=nu_m, Om_fn=Om_fn, phi_fn=phi_fn):
            return -(e / 2) * Om_fn(t) * cmath.exp(1j * nu_m * t) \
                   * math.sin(sign * mu_j * t + phi_fn(t) + math.pi / 2)
        sdf = f"sdf_{q}_{md.name}_{tag}"
        H = H + hconj(term({q: sigma_x, md.name: adag}, coeff=gx, name=sdf)
                      + term({q: sigma_y, md.name: adag}, coeff=gy, name=sdf))
        if order >= 2:
            n_op = adag @ a
            pref = -(e_jm ** 2) / 4
            def h2x(t, mu_j=mu_j, sign=sign, pref=pref, nu_m=nu_m, Om_fn=Om_fn, phi_fn=phi_fn):
                return pref * Om_fn(t) * cmath.exp(2j * nu_m * t) * math.cos(sign * mu_j * t + phi_fn(t))
            def h2y(t, mu_j=mu_j, sign=sign, pref=pref, nu_m=nu_m, Om_fn=Om_fn, phi_fn=phi_fn):
                return pref * Om_fn(t) * cmath.exp(2j * nu_m * t) * math.sin(sign * mu_j * t + phi_fn(t))
            def h0x(t, mu_j=mu_j, sign=sign, pref=pref, Om_fn=Om_fn, phi_fn=phi_fn):
                return pref * Om_fn(t) * math.cos(sign * mu_j * t + phi_fn(t))
            def h0y(t, mu_j=mu_j, sign=sign, pref=pref, Om_fn=Om_fn, phi_fn=phi_fn):
                return pref * Om_fn(t) * math.sin(sign * mu_j * t + phi_fn(t))
            two_n_plus_1 = 2 * n_op + np.eye(md.n_max + 1)
            ld2 = f"ld2_{q}_{md.name}_{tag}"
            H = H + hconj(term({q: sigma_x, md.name: adag @ adag}, coeff=h2x, name=ld2)
                          + term({q: sigma_y, md.name: adag @ adag}, coeff=h2y, name=ld2)) \
                  + term({q: sigma_x, md.name: two_n_plus_1}, coeff=h0x, name=ld2) \
                  + term({q: sigma_y, md.name: two_n_plus_1}, coeff=h0y, name=ld2)
    return H


# ---------------------------------------------------------------------------
# 2) + 3) stop after the Lamb-Dicke expansion (pre-RWA), order 1 and 2
# ---------------------------------------------------------------------------

def _ms_lamb_dicke(modes, n_ions, amplitudes, phases, motion_phases,
                   order, rwa, prefix="q", amplitude_red=None):
    N = n_ions
    Om = _as_funcs(amplitudes, N, "amplitudes")
    ph = _as_funcs(phases, N, "phases")
    psi = _as_funcs(motion_phases, N, "motion_phases")
    # a constant spin phase lets cos/sin(phi) be folded at build time rather than
    # recomputed on every ODE step -- the common case
    ph_raw = list(phases) if not (np.isscalar(phases) or callable(phases)) else [phases] * N
    ph_c = [_const_of(x) for x in ph_raw]
    if order >= 2 and len(modes) > 1:
        raise NotImplementedError(
            "ms_lamb_dicke2 supports one mode only: the eta^2 expansion of "
            "prod_m e^{i eta_m X_m} also contains cross-mode terms "
            "-eta_m eta_m' X_m X_m' (m < m'), which are not implemented. "
            "Refusing rather than silently dropping them -- use ms_lamb_dicke1 "
            "for the multi-mode pre-RWA model.")

    asym = amplitude_red is not None or any(md.detune_red is not None for md in modes)
    if rwa and asym:
        raise ValueError("asymmetric (detune_red/amplitude_red) drives are only "
                         "meaningful pre-RWA (rwa=False): the RWA spin-dependent "
                         "force already assumes one shared mu.")
    mu = None if rwa else _beat(modes, N)     # (N,) -- one beat drives every mode
    single = len(modes) == 1

    if asym:
        Om_red = _as_funcs(amplitude_red if amplitude_red is not None else amplitudes,
                           N, "amplitude_red")
        # per-mode red mu: this mode's own detune_red if given, else symmetric
        # (falls back to that mode's `detune`, i.e. this mode isn't asymmetric)
        mu_red_modes = [_beat([md], N, tone=("detune_red" if md.detune_red is not None
                                            else "detune"))
                       for md in modes]
        ops = {md.name: (annihilation(md.n_max), annihilation(md.n_max).conj().T)
              for md in modes}
        H = Model({**{f"{prefix}{j}": 2 for j in range(N)},
                  **{md.name: md.n_max + 1 for md in modes}})
        for j in range(N):
            q = f"{prefix}{j}"
            phi_blue = (lambda t, ph=ph[j], psi=psi[j]: ph(t) + psi(t))
            phi_red = (lambda t, ph=ph[j], psi=psi[j]: ph(t) - psi(t))
            H = _tone_group(H, q, j, modes, ops, mu[j], Om[j], phi_blue, order, "blue")
            mu_red_j = [mu_red_modes[m][j] for m in range(len(modes))]
            # one mu per mode for red (may differ mode-to-mode if only some
            # modes were given detune_red); _tone_group takes one mu_j, so
            # call it once per mode when mu_red isn't uniform across modes
            if len(set(np.round(mu_red_j, 9))) == 1:
                H = _tone_group(H, q, j, modes, ops, mu_red_j[0], Om_red[j], phi_red,
                                order, "red", sign=-1)
            else:
                for m, md in enumerate(modes):
                    H = _tone_group(H, q, j, [md], ops, mu_red_j[m], Om_red[j], phi_red,
                                    order, "red", sign=-1)
        return H

    subs = {**{f"{prefix}{j}": 2 for j in range(N)},
            **{md.name: md.n_max + 1 for md in modes}}
    H = Model(subs)
    # mode operators depend only on the mode, not the ion -- build once, not N times
    ops = {md.name: (annihilation(md.n_max), annihilation(md.n_max).conj().T)
           for md in modes}
    for j in range(N):
        q = f"{prefix}{j}"
        Omj, phj, psij = Om[j], ph[j], psi[j]
        # sigma_phi(t) split into constant sx / sy with time-dependent weights;
        # Phi = phi + pi/2 -> cos(Phi) = -sin(phi), sin(Phi) = cos(phi)
        cs = _trig(phj, ph_c[j])          # t -> (cos phi, sin phi), folded if constant
        if not rwa:
            # The bichromatic drive factor Omega(t) cos(mu t + psi(t)) is common to
            # EVERY term of this ion (carrier, sdf, ld2) -- compute it once per H(t).
            muj = mu[j]
            drive = _memo1(lambda t, Omj=Omj, psij=psij, muj=muj:
                           Omj(t) * math.cos(muj * t + psij(t)))

            # ---- carrier (eta^0): ONCE per ion, not once per mode ----
            def cx(t, drive=drive, cs=cs):
                return drive(t) * cs(t)[0]
            def cy(t, drive=drive, cs=cs):
                return drive(t) * cs(t)[1]
            H = H + term(sigma_x, on=q, coeff=cx, name=f"carrier_{q}") \
                  + term(sigma_y, on=q, coeff=cy, name=f"carrier_{q}")

        for md in modes:
            a, adag = ops[md.name]
            e_jm, d_jm = md.eta[j], md.detune[j]
            # single mode keeps the historical group name; multi-mode tags the mode
            sdf = f"sdf_{q}" if single else f"sdf_{q}_{md.name}"
            if rwa:
                # post-RWA spin-dependent force: hconj( sigma_Phi (x) f_{j,m}(t) a_m^dag )
                kern = _memo1(lambda t, e=e_jm, d=d_jm, Omj=Omj, psij=psij:
                              -(e * Omj(t) / 2) * cmath.exp(-1j * (d * t + psij(t))))
                def fx(t, kern=kern, cs=cs):
                    return kern(t) * (-cs(t)[1])
                def fy(t, kern=kern, cs=cs):
                    return kern(t) * cs(t)[0]
                H = H + hconj(term({q: sigma_x, md.name: adag}, coeff=fx, name=sdf)
                              + term({q: sigma_y, md.name: adag}, coeff=fy, name=sdf))
                continue
            # ---- eta^1: -eta_{j,m} Omega cos(mu t + psi) sigma_Phi X_m(t) ----
            nu_m = md.nu
            enu = _memo1(lambda t, nu_m=nu_m: cmath.exp(1j * nu_m * t))
            def gx(t, e=e_jm, drive=drive, enu=enu, cs=cs):
                return -e * drive(t) * enu(t) * (-cs(t)[1])
            def gy(t, e=e_jm, drive=drive, enu=enu, cs=cs):
                return -e * drive(t) * enu(t) * cs(t)[0]
            H = H + hconj(term({q: sigma_x, md.name: adag}, coeff=gx, name=sdf)
                          + term({q: sigma_y, md.name: adag}, coeff=gy, name=sdf))
            if order >= 2:
                # ---- eta^2: -(eta_{j,m}^2/2) Omega cos(mu t + psi) sigma_phi
                #             (a^2 e^{-2i nu t} + a^dag^2 e^{2i nu t} + 2n + 1) ----
                n_op = adag @ a
                pref = -(e_jm ** 2) / 2
                e2nu = _memo1(lambda t, nu_m=nu_m: cmath.exp(2j * nu_m * t))
                def h2x(t, pref=pref, drive=drive, e2nu=e2nu, cs=cs):
                    return pref * drive(t) * e2nu(t) * cs(t)[0]
                def h2y(t, pref=pref, drive=drive, e2nu=e2nu, cs=cs):
                    return pref * drive(t) * e2nu(t) * cs(t)[1]
                def h0x(t, pref=pref, drive=drive, cs=cs):
                    return pref * drive(t) * cs(t)[0]
                def h0y(t, pref=pref, drive=drive, cs=cs):
                    return pref * drive(t) * cs(t)[1]
                two_n_plus_1 = 2 * n_op + np.eye(md.n_max + 1)
                H = H + hconj(term({q: sigma_x, md.name: adag @ adag}, coeff=h2x,
                                   name=f"ld2_{q}")
                              + term({q: sigma_y, md.name: adag @ adag}, coeff=h2y,
                                     name=f"ld2_{q}")) \
                      + term({q: sigma_x, md.name: two_n_plus_1}, coeff=h0x, name=f"ld2_{q}") \
                      + term({q: sigma_y, md.name: two_n_plus_1}, coeff=h0y, name=f"ld2_{q}")
    return H


def ms_lamb_dicke1(participation=None, eta=None, detune=None, amplitudes=1.0,
                   phases=0.0, n_max=None, nu=None, rwa=False, prefix="q",
                   mode="mode", *, modes=None, motion_phases=0.0,
                   detune_red=None, amplitude_red=None) -> Model:
    """MS Hamiltonian truncated at FIRST order in the Lamb-Dicke expansion,
    as a composable term-layer `Model`.

    Groups: `carrier_qj` (one per ion) and, per ion and mode, `sdf_qj` (single
    mode) or `sdf_qj_<modename>` (multi-mode) -- or, when the drive is
    asymmetric (see below), `carrier_qj_blue`/`_red` and `sdf_qj_<modename>_blue`/
    `_red`, kept as separate groups since they are genuinely different tones.

    rwa=False (default): pre-RWA -- keeps the off-resonant carrier and the
    counter-rotating spin-motion terms (needs `nu` on every mode). rwa=True:
    drops them, leaving exactly the spin-dependent force whose Magnus closed
    form is `MSMagnus` (used to cross-validate the two). Amplitudes, spin phases
    AND motion phases may all be time-dependent here.

    detune_red, amplitude_red: give the RED tone an independent detuning
    and/or amplitude from the BLUE tone (`detune`/`amplitudes`) -- an
    asymmetric bichromatic drive, e.g. for a detuned-carrier sigma_z. Per ion
    (scalar broadcasts); None (default) means that tone matches blue, i.e.
    today's symmetric drive, unchanged. rwa=True + asymmetric is refused (the
    RWA spin-dependent force already assumes one shared mu). NOT supported by
    `MSMagnus`: an asymmetric eta^1 term has a rotating spin axis, so the
    argument that terminates its Magnus series does not obviously carry over."""
    mds, n_ions = _mode_list(modes, participation, eta, detune, n_max, nu, mode,
                             amplitudes, phases, detune_red)
    return _ms_lamb_dicke(mds, n_ions, amplitudes, phases, motion_phases,
                          order=1, rwa=rwa, prefix=prefix, amplitude_red=amplitude_red)


def ms_lamb_dicke2(participation=None, eta=None, detune=None, amplitudes=1.0,
                   phases=0.0, n_max=None, nu=None, prefix="q", mode="mode",
                   *, modes=None, motion_phases=0.0,
                   detune_red=None, amplitude_red=None) -> Model:
    """MS Hamiltonian truncated at SECOND order in the Lamb-Dicke expansion
    (adds groups ld2_qj: the a^2/a^dag^2 and (2n+1) corrections). Pre-RWA by
    construction -- every eta^2 term oscillates fast (mu, mu +- 2nu), so under
    the RWA order 2 would reduce back to order 1; simulating their off-resonant
    effect is the point of this builder. Needs `nu`.

    ONE MODE ONLY: the multi-mode eta^2 expansion also has cross-mode terms
    (-eta_m eta_m' X_m X_m'), which are not implemented -- this raises rather
    than dropping them silently.

    detune_red, amplitude_red: see `ms_lamb_dicke1` -- same asymmetric-tone
    support, same restrictions."""
    mds, n_ions = _mode_list(modes, participation, eta, detune, n_max, nu, mode,
                             amplitudes, phases, detune_red)
    return _ms_lamb_dicke(mds, n_ions, amplitudes, phases, motion_phases,
                          order=2, rwa=False, prefix=prefix, amplitude_red=amplitude_red)


# ---------------------------------------------------------------------------
# phase-space plotting
# ---------------------------------------------------------------------------

def expectation_alpha(evolution, ts, mode="mode") -> np.ndarray:
    """<a>(t) of the named mode from any evolution with trace_out -- the
    measured phase-space trajectory. NOTE: for a spin-dependent force acting
    on a spin superposition, the +alpha and -alpha branches average out and
    <a> ~ 0; to see one branch, evolve an eigenstate of the relevant
    sigma_{Phi_j}. Returns complex array (len(ts),)."""
    ts = np.asarray(ts)
    rho = np.asarray(evolution.trace_out(*[n for n in evolution.subsystems if n != mode],
                                         t=ts))
    a = annihilation(rho.shape[-1] - 1)
    return np.einsum("nij,ji->n", rho, a)


def plot_phase_space(alphas, labels=None, ax=None):
    """Phase-space trajectories: Re alpha vs Im alpha.

    `alphas`: complex array (n_times,), (n_times, n_ions), or the
    (n_times, n_ions, n_modes) stack that `MSMagnus.alpha_trajectory(ts)` returns
    (flattened to one trajectory per ion-mode pair). Also takes
    `expectation_alpha(evolution, ts)`. Time is always the FIRST axis: shape is
    never inferred, since (n_times, n_ions) and (n_ions, n_times) are
    indistinguishable whenever both are plausible.
    Start marked with a dot, end with a cross (loop closure check: the cross
    returns to the origin at T = 2*pi/detune for constant amplitude)."""
    import matplotlib.pyplot as plt
    alphas = np.asarray(alphas, dtype=complex)
    auto = None
    if alphas.ndim == 1:
        alphas = alphas[:, None]      # one trajectory
    elif alphas.ndim == 3:            # (n_times, n_ions, n_modes)
        n_t, n_i, n_m = alphas.shape
        auto = [f"ion {j}, mode {m}" for j in range(n_i) for m in range(n_m)]
        alphas = alphas.reshape(n_t, n_i * n_m)
    elif alphas.ndim != 2:
        raise ValueError(f"alphas must be (n_times,), (n_times, n_ions) or "
                         f"(n_times, n_ions, n_modes), got shape {alphas.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    n_traj = alphas.shape[1]
    for j in range(n_traj):
        traj = alphas[:, j]
        label = (labels[j] if labels is not None
                 else (auto[j] if auto is not None else f"ion {j}"))
        (line,) = ax.plot(traj.real, traj.imag, label=label)
        ax.plot(traj.real[0], traj.imag[0], "o", color=line.get_color(), ms=6)
        ax.plot(traj.real[-1], traj.imag[-1], "x", color=line.get_color(), ms=8)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlabel(r"Re $\alpha$")
    ax.set_ylabel(r"Im $\alpha$")
    ax.set_aspect("equal")
    ax.legend()
    return ax
