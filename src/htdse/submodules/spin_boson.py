"""General spin-boson physics: a spin driven by any number of TONES, coupled
to any number of bosonic MODES, at a chosen order of the Lamb-Dicke expansion.

This is the general layer -- it does not know what an ion, a cavity, or a
neutral atom is. `trapped_ion.py` specializes it; `molmer_sorensen.py` is one
recipe built from it (two tones at +-(nu+delta)).

THE LADDER (do not re-derive; this is what every builder here implements):

    H(t) = sum_k (Om_k(t)/2) [ s+ e^{-i(mu_k t + phi_k(t))} prod_m e^{i eta_m X_m(t)} + h.c. ]
    X_m(t) = a_m e^{-i nu_m t} + a_m^dag e^{+i nu_m t}

`mu_k` is TONE k's signed offset from the spin's resonance (w_laser - w_0).
`lamb_dicke=` selects how far `e^{i eta X}` is expanded:

    lamb_dicke=1: keep O(eta^1)  -- linear spin-motion coupling
    lamb_dicke=2: keep O(eta^2)  -- adds a^2/a^dag^2 and a Stark-shift-like term
                  PER mode, plus -- whenever more than one mode is driven --
                  the cross term between every pair of DIFFERENT modes,
                  eta_m*eta_m' * X_m(t) X_m'(t), oscillating at the SUM
                  (nu_m+nu_m') and DIFFERENCE (nu_m-nu_m') of the two mode
                  frequencies. Not optional: a shared drive tone genuinely
                  couples two modes together at this order, and dropping it
                  would silently miss that (see PHYSICS.md section 8).
    lamb_dicke=None: no expansion -- exact, see the closed-form gate below

`rwa=` drops everything except the single resonant term of each tone: the bare
carrier if `mu_k == 0`, else the eta^1 motional-sideband term nearest resonance
(detuning `mu_k - nu_m`). Only defined at `lamb_dicke=1` -- eta^2 terms are all
fast-rotating and have no resonant piece to keep.

ONE TONE, what you get after RWA:

    Tone(offset=0)                -> (Om/2) sigma_phi                    carrier
    Tone(offset=-nu)              -> g (s+ a + h.c.), g = eta*Om/2        JC (red sideband)
    Tone(offset=+nu)              -> g (s+ a^dag + h.c.)                 anti-JC (blue sideband)
    Tone(offset=+-(nu+delta)), x2 -> sigma_Phi (x) (f a^dag + f* a)      Molmer-Sorensen force

MS is the red-sideband JC plus the blue-sideband anti-JC of a SECOND tone --
`molmer_sorensen.py` is nothing but that pairing plus the closed form for it.

EXACTNESS CONTRACT: at `lamb_dicke=None, rwa=False` this Hamiltonian has no
Lamb-Dicke expansion and no vibrational RWA -- the only remaining errors are
Fock truncation and ODE tolerance, both already controllable through this
package. Permanently NOT relaxable here: two levels per spin, the optical RWA
(this Hamiltonian is already written in the interaction picture with the
w0+w_laser terms dropped), and dipole/plane-wave coupling (`e^{i eta X}` is
what THAT coupling looks like -- a different coupling, e.g. a cavity mode, is
`jaynes_cummings`/`rabi` below, not a special case of this).

TWO COUPLING MECHANISMS, deliberately not unified into one function:
  - recoil (this module's tone/eta machinery): the drive enters as
    e^{i eta (a+a^dag)} -- a spin moving in a light field. Trapped ions and
    neutral atoms in tweezers/lattices, same physics.
  - linear/dipole (`jaynes_cummings`, `rabi`): g(s+ a + h.c.) from the start --
    cavity QED, circuit QED. No eta, no Lamb-Dicke expansion.
  They meet exactly at `lamb_dicke=1, rwa=True`: a red-detuned tone
  (`Tone(offset=-nu)`) reduces to `jaynes_cummings(g=1j*eta*Omega/2, ...)` --
  a regression test, not a shared code path (merging them behind one function
  with mutually-exclusive keyword arguments is exactly the API disease this
  module replaces). The factor of `1j` is not a convention mismatch to paper
  over: the Lamb-Dicke expansion of `e^{i eta X}` starts `1 + i eta X + ...`,
  so recoil coupling carries an intrinsic quarter-turn spin-phase relative to
  a dipole coupling written directly as `g(s+ a + h.c.)` -- the same quarter
  turn as Monroe's `Phi = phi + pi/2` sideband axis.
"""
import cmath
import math
from typing import NamedTuple

import numpy as np
from scipy import sparse as _sp
from scipy.linalg import expm

from ..core.subsystems import embed
from ..core.system import System
from ..core.terms import Model, plus_hc, term
from .harmonic_oscillator import annihilation, number_operator
from .spin import sigma_x, sigma_y, sigma_plus


TONE_TABLE = """\
One spin, one tone, after RWA (lamb_dicke=1, rwa=True):

    tone offset          result                              name
    -------------------  ----------------------------------  ---------------------------
    0                    (Omega/2) sigma_phi                 carrier, no motional content
    -nu                  g(sigma_+ a + h.c.), g=i*eta*Om/2    red sideband = Jaynes-Cummings
    +nu                  g(sigma_+ a^dag + h.c.)              blue sideband = anti-JC
    +-(nu+delta), 2 tones  sigma_Phi (x) (f a^dag + f* a)     Molmer-Sorensen force

MS is red-sideband JC plus blue-sideband anti-JC of a second tone: not a
primitive, a superposition of two rows above.

The i in the JC bridge is not a bug: the Lamb-Dicke expansion of e^{i eta X}
starts 1 + i eta X + ..., so recoil coupling carries an intrinsic quarter-turn
spin phase relative to a dipole coupling written directly as g(s+ a + h.c.)
(jaynes_cummings/rabi -- no eta, never derived from this table).

lamb_dicke x rwa, the full ladder:

                    rwa=False (pre-RWA)              rwa=True
    lamb_dicke=None full e^{i eta X(t)}, exact        --
    lamb_dicke=1    keep O(eta^1)                     resonant term only (table above)
    lamb_dicke=2    keep O(eta^1) + O(eta^2)           -- (eta^2 has no resonant piece)
"""


def explain():
    """Print the tone table and approximation ladder -- what to pass
    `driven_spins` for and what you get back, without opening source."""
    print(TONE_TABLE)


class Tone(NamedTuple):
    """One drive tone: amplitude Om(t) at SIGNED offset `offset` = w_laser -
    w_0 from the spin's resonance, optical phase `phase`.

    offset: signed. A negative offset is what older MS code called the "red"
            tone (sign=-1) -- s+ e^{+i(mu t - phi)} + h.c. is identically
            s+ e^{-i((-mu) t + phi)} + h.c., so the sign of `offset` alone
            carries that distinction; there is no separate flag.
            May also be callable, mu(t) -- an instantaneous (chirped)
            detuning -- at `rwa=False` only: RWA's "keep whichever sideband
            is nearest resonance" has no answer when the detuning itself
            moves, so `rwa=True` with a callable offset raises. The phase
            that appears everywhere `mu t` would is then the accumulated
            phase int_0^t mu(t') dt', found by quadrature (see `_phase_of`) --
            a real cost per RHS evaluation, unlike the free `mu*t` a constant
            offset gets, so only pay it when you need a genuine chirp.
    amp:    scalar, callable f(t), or a per-spin sequence of either.
    phase:  scalar, callable f(t), or a per-spin sequence of either.
    """
    offset: float
    amp: object = 1.0
    phase: object = 0.0


class Mode(NamedTuple):
    """One bosonic mode: trap/cavity frequency `nu`, Lamb-Dicke coupling
    `eta` (scalar broadcasts, or a per-spin array -- the participation of
    each spin in this mode), Fock truncation `n_max`."""
    nu: float
    eta: object
    n_max: int
    name: str = "mode"

    @classmethod
    def from_participation(cls, nu, eta, b, n_max, name="mode"):
        """`Mode(nu, eta=eta*b, n_max, name)` -- the arithmetic every caller
        otherwise repeats at the call site. `eta`: the bare Lamb-Dicke
        parameter (one number). `b`: per-spin participation (e.g. COM mode
        of two ions is `[1, 1]/sqrt(2)`) -- scalar or array."""
        return cls(nu=nu, eta=eta * np.asarray(b, dtype=float), n_max=n_max, name=name)


# ---------------------------------------------------------------------------
# argument normalization (ported from molmer_sorensen.py, unchanged)
# ---------------------------------------------------------------------------

def _real(xi, what):
    z = complex(xi)
    if z.imag != 0:
        raise ValueError(f"{what}: got complex value {xi!r}, but these are real "
                         f"quantities -- encode the phase via `phase=` instead.")
    return z.real


def _as_funcs(x, n, what):
    """Normalize `x` (scalar | callable | sequence of either, length n) into a
    list of n callables f(t)."""
    if np.isscalar(x) or callable(x):
        x = [x] * n
    if len(x) != n:
        raise ValueError(f"{what}: expected {n} entries (one per spin), got {len(x)}")
    return [xi if callable(xi) else (lambda t, c=_real(xi, what): c) for xi in x]


def _as_consts(x, n, what):
    """Like _as_funcs but requires plain numbers (no callables) -- used where
    a time-dependent value would break a closed-form derivation."""
    if np.isscalar(x):
        x = [x] * n
    if any(callable(xi) for xi in x):
        raise ValueError(f"{what} must be constants here (see caller's docstring)")
    if len(x) != n:
        raise ValueError(f"{what}: expected {n} entries (one per spin), got {len(x)}")
    return [float(xi) for xi in x]


def _eval(fn, t):
    """Evaluate a coefficient callable at a scalar or array time.

    Fast path first: most drive parameters are constants or numpy-aware
    expressions, and both evaluate on the whole grid in one call. `np.vectorize`
    is a Python-level loop over every grid point, so it is the LAST resort."""
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


def _memo1(fn):
    """One-slot memo on the argument -- every term of one spin/tone/mode is
    evaluated at the SAME t inside a single H(t) call."""
    last_t, last_v = [None], [None]

    def g(t):
        if t != last_t[0]:
            last_t[0], last_v[0] = t, fn(t)
        return last_v[0]
    return g


def _phase_of(mu):
    """mu (a tone's offset: a constant, or a callable instantaneous detuning
    mu(t)) -> Phi(t), the phase that belongs where `mu * t` would go: the
    accumulated phase int_0^t mu(t') dt'.

    Constant mu: Phi(t) = mu*t exactly, no integration -- the common case
    stays free. Callable mu: Phi(t) is found by quadrature from 0 to t,
    fresh on every call -- the real cost of a genuinely time-dependent
    detuning; a memo would help only for a monotonically-advancing solve,
    which an adaptive stepper's stage evaluations don't guarantee."""
    if callable(mu):
        from scipy.integrate import quad
        def Phi(t):
            return quad(mu, 0.0, t)[0] if t != 0.0 else 0.0
        return Phi
    return lambda t, mu=mu: mu * t


def _norm_mode(md: Mode, n_spins: int) -> Mode:
    eta = np.broadcast_to(np.asarray(md.eta, dtype=float), (n_spins,)).copy()
    return Mode(float(md.nu), eta, int(md.n_max), str(md.name))


# ---------------------------------------------------------------------------
# one tone's contribution to one spin
# ---------------------------------------------------------------------------

def _tone_group(H, q, j, modes, ops, mu, amp_fn, phase_fn, lamb_dicke, rwa, tag):
    """Add tone (offset `mu`, amplitude `amp_fn`, phase `phase_fn`)'s
    contribution for spin `q` (index j into the modes' per-spin eta arrays)
    to Model H, at the given `lamb_dicke` order (1 or 2) and RWA setting."""
    if rwa:
        if lamb_dicke != 1:
            raise ValueError("rwa=True is only defined at lamb_dicke=1 -- the "
                             "eta^2 terms are all fast-rotating and have no "
                             "resonant piece to keep")
        if callable(mu):
            raise ValueError(
                "rwa=True needs a constant Tone.offset -- 'keep whichever "
                "sideband is nearest resonance' has no fixed answer when the "
                "detuning itself is time-dependent (a chirp can sweep through "
                "several). Use rwa=False (pre-RWA, lamb_dicke=1 or 2) for a "
                "chirped tone.")
        if mu == 0:
            # on resonance: no motional sideband is near resonance either, so
            # the only surviving piece is the bare carrier.
            def cx(t, amp_fn=amp_fn, phase_fn=phase_fn):
                return (amp_fn(t) / 2) * math.cos(phase_fn(t))
            def cy(t, amp_fn=amp_fn, phase_fn=phase_fn):
                return (amp_fn(t) / 2) * math.sin(phase_fn(t))
            return H + term(sigma_x, on=q, coeff=cx, name=f"carrier_{q}_{tag}") \
                     + term(sigma_y, on=q, coeff=cy, name=f"carrier_{q}_{tag}")
        for md in modes:
            a, adag = ops[md.name]
            e_jm, nu_m = md.eta[j], md.nu
            # X_m(t) has TWO possible resonant combinations with this tone:
            #   d_blue = mu - nu_m  (near resonance for mu ~ +nu_m: "blue-type")
            #   d_red  = mu + nu_m  (near resonance for mu ~ -nu_m: "red-type")
            # Keep whichever is smaller -- the other is the fast piece RWA drops.
            d_blue, d_red = mu - nu_m, mu + nu_m
            if abs(d_blue) <= abs(d_red):
                # sigma_+ (x) adag coefficient: i*(eta*Om/2)*exp(-i(d_blue*t+phase))
                kern = _memo1(lambda t, e=e_jm, d=d_blue, amp_fn=amp_fn, phase_fn=phase_fn:
                              1j * (e * amp_fn(t) / 2) * cmath.exp(-1j * (d * t + phase_fn(t))))
                def fx(t, kern=kern):
                    return kern(t) / 2
                def fy(t, kern=kern):
                    return kern(t) * 1j / 2
            else:
                # sigma_- (x) adag coefficient: -i*(eta*Om/2)*exp(+i(d_red*t+phase))
                kern = _memo1(lambda t, e=e_jm, d=d_red, amp_fn=amp_fn, phase_fn=phase_fn:
                              -1j * (e * amp_fn(t) / 2) * cmath.exp(1j * (d * t + phase_fn(t))))
                def fx(t, kern=kern):
                    return kern(t) / 2
                def fy(t, kern=kern):
                    return -kern(t) * 1j / 2
            sdf = f"sdf_{q}_{md.name}_{tag}"
            H = H + plus_hc(term({q: sigma_x, md.name: adag}, coeff=fx, name=sdf)
                          + term({q: sigma_y, md.name: adag}, coeff=fy, name=sdf))
        return H

    # ---- pre-RWA: carrier + eta^lamb_dicke ----
    Phi = _phase_of(mu)   # Phi(t) = mu*t for a constant mu; a chirp otherwise
    def cx(t, Phi=Phi, amp_fn=amp_fn, phase_fn=phase_fn):
        return (amp_fn(t) / 2) * math.cos(Phi(t) + phase_fn(t))
    def cy(t, Phi=Phi, amp_fn=amp_fn, phase_fn=phase_fn):
        return (amp_fn(t) / 2) * math.sin(Phi(t) + phase_fn(t))
    H = H + term(sigma_x, on=q, coeff=cx, name=f"carrier_{q}_{tag}") \
          + term(sigma_y, on=q, coeff=cy, name=f"carrier_{q}_{tag}")

    for md in modes:
        a, adag = ops[md.name]
        e_jm, nu_m = md.eta[j], md.nu
        def gx(t, Phi=Phi, e=e_jm, nu_m=nu_m, amp_fn=amp_fn, phase_fn=phase_fn):
            return -(e / 2) * amp_fn(t) * cmath.exp(1j * nu_m * t) \
                   * math.cos(Phi(t) + phase_fn(t) + math.pi / 2)
        def gy(t, Phi=Phi, e=e_jm, nu_m=nu_m, amp_fn=amp_fn, phase_fn=phase_fn):
            return -(e / 2) * amp_fn(t) * cmath.exp(1j * nu_m * t) \
                   * math.sin(Phi(t) + phase_fn(t) + math.pi / 2)
        sdf = f"sdf_{q}_{md.name}_{tag}"
        H = H + plus_hc(term({q: sigma_x, md.name: adag}, coeff=gx, name=sdf)
                      + term({q: sigma_y, md.name: adag}, coeff=gy, name=sdf))
        if lamb_dicke >= 2:
            n_op = adag @ a
            pref = -(e_jm ** 2) / 4
            def h2x(t, Phi=Phi, pref=pref, nu_m=nu_m, amp_fn=amp_fn, phase_fn=phase_fn):
                return pref * amp_fn(t) * cmath.exp(2j * nu_m * t) * math.cos(Phi(t) + phase_fn(t))
            def h2y(t, Phi=Phi, pref=pref, nu_m=nu_m, amp_fn=amp_fn, phase_fn=phase_fn):
                return pref * amp_fn(t) * cmath.exp(2j * nu_m * t) * math.sin(Phi(t) + phase_fn(t))
            def h0x(t, Phi=Phi, pref=pref, amp_fn=amp_fn, phase_fn=phase_fn):
                return pref * amp_fn(t) * math.cos(Phi(t) + phase_fn(t))
            def h0y(t, Phi=Phi, pref=pref, amp_fn=amp_fn, phase_fn=phase_fn):
                return pref * amp_fn(t) * math.sin(Phi(t) + phase_fn(t))
            two_n_plus_1 = 2 * n_op + np.eye(md.n_max + 1)
            ld2 = f"ld2_{q}_{md.name}_{tag}"
            H = H + plus_hc(term({q: sigma_x, md.name: adag @ adag}, coeff=h2x, name=ld2)
                          + term({q: sigma_y, md.name: adag @ adag}, coeff=h2y, name=ld2)) \
                  + term({q: sigma_x, md.name: two_n_plus_1}, coeff=h0x, name=ld2) \
                  + term({q: sigma_y, md.name: two_n_plus_1}, coeff=h0y, name=ld2)

    if lamb_dicke >= 2:
        # CROSS-mode eta^2 term: expanding prod_m e^{i eta_m X_m(t)} to second
        # order picks up not just each mode's own eta_m^2 X_m^2 (above) but,
        # for every pair of DIFFERENT modes, (i eta_m X_m)(i eta_m' X_m') =
        # -eta_m eta_m' X_m(t) X_m'(t) -- a genuine physical effect (a
        # spin-motion coupling between two modes through their shared drive
        # tone) that a per-mode-only expansion silently drops whenever more
        # than one mode is driven. X_m(t) X_m'(t) expands into two Hermitian
        # pieces (each self-adjoint under the SAME plus_hc trick as the
        # self-term's adag@adag/2n+1 split above):
        #   sum-type:  adag_m adag_m' e^{+i(nu_m+nu_m')t} + h.c.  (two-phonon
        #              creation/annihilation together -- driven at the SUM of
        #              the two mode frequencies)
        #   diff-type: adag_m a_m'    e^{+i(nu_m-nu_m')t}  + h.c.  (a phonon
        #              moved from mode m' to mode m -- driven at the
        #              DIFFERENCE of the two mode frequencies)
        # Verified against a from-scratch two-mode expansion of the exact
        # displacement-operator product (no series) in tests/test_molmer_sorensen.py.
        for mi, md1 in enumerate(modes):
            e1, nu1 = md1.eta[j], md1.nu
            a1, adag1 = ops[md1.name]
            for md2 in modes[mi + 1:]:
                e2, nu2 = md2.eta[j], md2.nu
                a2, adag2 = ops[md2.name]
                pref_c = -(e1 * e2) / 2
                def csx(t, Phi=Phi, pref_c=pref_c, nu1=nu1, nu2=nu2,
                       amp_fn=amp_fn, phase_fn=phase_fn):
                    return pref_c * amp_fn(t) * cmath.exp(1j * (nu1 + nu2) * t) \
                           * math.cos(Phi(t) + phase_fn(t))
                def csy(t, Phi=Phi, pref_c=pref_c, nu1=nu1, nu2=nu2,
                       amp_fn=amp_fn, phase_fn=phase_fn):
                    return pref_c * amp_fn(t) * cmath.exp(1j * (nu1 + nu2) * t) \
                           * math.sin(Phi(t) + phase_fn(t))
                def cdx(t, Phi=Phi, pref_c=pref_c, nu1=nu1, nu2=nu2,
                       amp_fn=amp_fn, phase_fn=phase_fn):
                    return pref_c * amp_fn(t) * cmath.exp(1j * (nu1 - nu2) * t) \
                           * math.cos(Phi(t) + phase_fn(t))
                def cdy(t, Phi=Phi, pref_c=pref_c, nu1=nu1, nu2=nu2,
                       amp_fn=amp_fn, phase_fn=phase_fn):
                    return pref_c * amp_fn(t) * cmath.exp(1j * (nu1 - nu2) * t) \
                           * math.sin(Phi(t) + phase_fn(t))
                ldx = f"ld2x_{q}_{md1.name}_{md2.name}_{tag}"
                H = H + plus_hc(term({q: sigma_x, md1.name: adag1, md2.name: adag2},
                                    coeff=csx, name=ldx)
                              + term({q: sigma_y, md1.name: adag1, md2.name: adag2},
                                    coeff=csy, name=ldx)) \
                      + plus_hc(term({q: sigma_x, md1.name: adag1, md2.name: a2},
                                    coeff=cdx, name=ldx)
                              + term({q: sigma_y, md1.name: adag1, md2.name: a2},
                                    coeff=cdy, name=ldx))
    return H


# ---------------------------------------------------------------------------
# the primitive
# ---------------------------------------------------------------------------

def driven_spins(tones, spins, modes, lamb_dicke=1, rwa=False, prefix=None,
                 sparse=False) -> Model:
    """The Hamiltonian of `spins` driven by `tones`, coupled to `modes`, at
    Lamb-Dicke order `lamb_dicke` (1 or 2; use the closed-form gate or the
    exact builder for the un-expanded Hamiltonian). See the module docstring
    for the tone table and the exactness contract.

    tones: list of `Tone`.
    spins: list of subsystem names, e.g. `["q0", "q1"]` -- each becomes a
           2-dim subsystem. `prefix` is ignored if spins is already given as
           names; kept only for parity with callers that used to pass a count.
    modes: list of `Mode`.
    sparse: at `lamb_dicke=1/2` this is just `driven_spins(...).sparse()` done
        for you; at `lamb_dicke=None` it is the only way to get a sparse
        `exact_drive` (that rung has no `Model` to call `.sparse()` on
        afterward -- see `exact_drive`).
    """
    for md in modes:
        if not hasattr(md, "nu"):
            raise TypeError(
                f"{md!r} has no `nu` -- `driven_spins` needs "
                "`spin_boson.Mode(nu, eta, n_max)`.")
    if lamb_dicke is None:
        return exact_drive(tones, spins, modes, sparse=sparse)
    if lamb_dicke not in (1, 2):
        raise ValueError(f"lamb_dicke must be 1, 2, or None, got {lamb_dicke!r}")
    n = len(spins)
    modes = [_norm_mode(md, n) for md in modes]
    subs = {**{q: 2 for q in spins}, **{md.name: md.n_max + 1 for md in modes}}
    H = Model(subs).sparse(sparse)
    ops = {md.name: (annihilation(md.n_max), annihilation(md.n_max).conj().T)
           for md in modes}
    for k, tone in enumerate(tones):
        amp_fns = _as_funcs(tone.amp, n, f"tones[{k}].amp")
        phase_fns = _as_funcs(tone.phase, n, f"tones[{k}].phase")
        for j, q in enumerate(spins):
            H = _tone_group(H, q, j, modes, ops, tone.offset, amp_fns[j],
                            phase_fns[j], lamb_dicke, rwa, tag=f"tone{k}")
    return H


# ---------------------------------------------------------------------------
# linear/dipole coupling -- NOT derived from driven_spins (see module docstring)
# ---------------------------------------------------------------------------

def jaynes_cummings(g, spin, mode, n_max, detuning=0.0, name="jc") -> Model:
    """H = detuning * a^dag a + g (sigma_+ a + h.c.) -- the resonant (RWA)
    dipole coupling of one spin to one bosonic mode. `g` may be a scalar or
    f(t). This is the `lamb_dicke=1, rwa=True` limit of a red-sideband tone,
    with g = eta*Omega/2 -- see `driven_spins`."""
    a = annihilation(n_max)
    H = plus_hc(term({spin: sigma_plus, mode: a}, coeff=g, name=name))
    if detuning:
        H = H + term(detuning * number_operator(n_max), on=mode, name=f"{name}_detuning")
    return H


def exact_drive(tones, spins, modes, sparse: bool = False) -> System:
    """The un-expanded, un-RWA'd Hamiltonian: no Lamb-Dicke expansion of
    `e^{i eta X(t)}`, at all -- this is `driven_spins(..., lamb_dicke=None)`.

    Returns a `System`, not a `Model`: `sigma_+ (x) D(t)` is a genuinely
    t-dependent matrix, not a sum of (scalar coefficient) x (fixed operator),
    so this rung has no `+` / `.replace()`. That is a property of the physics
    at this rung, not a limitation being papered over.

    Cost: `e^{i eta X(t)}` is exactly the displacement operator
    `D(i eta e^{i nu t}) = R(nu t) D(i eta) R(nu t)^dag`, `R` diagonal in the
    Fock basis -- one `expm` per (spin, mode) at construction, then O(dim^2)
    per call via a diagonal conjugation, not per-call `expm`.

    sparse: embed `sigma_+` and the displacement block as scipy CSR instead of
        dense arrays, so `.hamiltonian(t)`'s O(dim^2) cost becomes O(nnz).
        There is no `.sparse()` to call afterward the way a `Model` has --
        this rung has no `Model` underneath -- so it must be chosen here.
        `.hamiltonian(t)` still densifies (same contract as `Model`); the
        solver reads sparse `H(t)` from `._h_native` instead.
    """
    n = len(spins)
    modes = [_norm_mode(md, n) for md in modes]
    subsystems = {**{q: 2 for q in spins}, **{md.name: md.n_max + 1 for md in modes}}
    mode_names = tuple(md.name for md in modes)

    # D0[j][m] = e^{i eta_{j,m} (a_m + a_m^dag)}, the displacement at t=0 for
    # spin j's own participation in mode m. R[m] rotates it into D_m(t).
    D0 = [[np.asarray(expm(1j * md.eta[j] * (annihilation(md.n_max)
                                              + annihilation(md.n_max).conj().T)))
          for md in modes] for j in range(n)]
    R_diag = [np.exp(1j * np.arange(md.n_max + 1)) for md in modes]  # e^{i n}, scale by nu*t

    def displacement(j, t):
        out = None
        for m, md in enumerate(modes):
            phase = R_diag[m] ** (md.nu * t)  # e^{i n nu t} per Fock level
            Dm = (phase[:, None] * D0[j][m]) * phase.conj()[None, :]
            out = Dm if out is None else np.kron(out, Dm)
        return out

    class _ExactDrive(System):
        def __init__(self):
            self.subsystems = dict(subsystems)

        @property
        def dim(self):
            d = 1
            for v in self.subsystems.values():
                d *= v
            return d

        def _build(self, t):
            sp_local = _sp.csr_matrix(sigma_plus) if sparse else sigma_plus
            eye_full = (_sp.identity(self.dim, dtype=complex, format="csr")
                        if sparse else np.eye(self.dim, dtype=complex))
            H = _sp.csr_matrix((self.dim, self.dim), dtype=complex) if sparse \
                else np.zeros((self.dim, self.dim), dtype=complex)
            for k, tone in enumerate(tones):
                amp_fns = _as_funcs(tone.amp, n, f"tones[{k}].amp")
                phase_fns = _as_funcs(tone.phase, n, f"tones[{k}].phase")
                Phi = _phase_of(tone.offset)
                for j, q in enumerate(spins):
                    D = displacement(j, t)
                    if sparse and modes:
                        D = _sp.csr_matrix(D)
                    coeff = (amp_fns[j](t) / 2) * cmath.exp(-1j * (Phi(t) + phase_fns[j](t)))
                    Sp = embed(sp_local, self.subsystems, q)
                    Dfull = embed(D, self.subsystems, mode_names) if modes else eye_full
                    term_mat = coeff * (Sp @ Dfull)
                    H = H + term_mat + term_mat.conj().T
            return H.tocsr() if sparse else H

        def hamiltonian(self, t):
            H = self._build(t)
            return H.toarray() if sparse else H

        def _h_native(self, t):
            return self._build(t)

        def __repr__(self):
            return f"exact_drive({len(tones)} tone(s), spins={spins}, modes={[md.name for md in modes]})"

        def __add__(self, other):
            raise TypeError(
                "exact_drive(...) (lamb_dicke=None) has no '+': sigma_+ (x) D(t) is a "
                "genuinely t-dependent matrix, not a sum of scalar-coefficient x "
                "fixed-operator terms, so there is nothing for '+' to compose. Sum "
                "the Hamiltonians yourself inside a System subclass, or work at "
                "lamb_dicke=1/2 where the physics IS a sum of terms.")

        def replace(self, **kwargs):
            raise AttributeError(
                "exact_drive(...) (lamb_dicke=None) has no .replace(): that's a "
                "Model operation on named term groups, and this rung has none (see "
                "the '+' error for why). Build a new exact_drive(...) with the "
                "changed tones/modes instead.")

    return _ExactDrive()


def rabi(g, spin, mode, n_max, detuning=0.0, name="rabi") -> Model:
    """H = detuning * a^dag a + g sigma_x (x) (a + a^dag) -- the quantum Rabi
    model: the dipole coupling WITHOUT the rotating-wave approximation
    (Jaynes-Cummings + anti-Jaynes-Cummings). `g` may be a scalar or f(t)."""
    a = annihilation(n_max)
    adag = a.conj().T
    H = term({spin: sigma_x, mode: a + adag}, coeff=g, name=name)
    if detuning:
        H = H + term(detuning * number_operator(n_max), on=mode, name=f"{name}_detuning")
    return H
