"""Molmer-Sorensen mechanism suite: bichromatic spin-dependent force on N spins
coupled to one motional mode, at three levels of approximation.

All mechanisms share one argument set:
    participation : array (N,) -- mode participation b_j per ion
    eta           : Lamb-Dicke parameter (participation multiplies it per ion)
    detune        : delta -- symmetric detuning of the bichromatic beat from the
                    sideband (gate detuning; loop closes at T = 2*pi/delta)
    amplitudes    : per-ion Rabi amplitude Omega_j -- scalar or callable f(t),
                    or a single value applied to all ions
    phases        : per-ion spin phase phi_j -- scalar (or callable, where allowed)
    n_max         : Fock truncation of the mode
    nu            : trap frequency (only the pre-RWA builders need it)

CONVENTIONS (hbar = 1). Interaction picture w.r.t. qubit and mode frequencies,
bichromatic beat at mu = nu + detune, spin phase phi_j, Phi_j = phi_j + pi/2:

Post-RWA spin-dependent force (Lamb-Dicke order 1 + dropping all terms
oscillating at mu, 2nu, ...):

    H_RWA(t) = sum_j sigma_{Phi_j}^{(j)} (x) ( f_j(t) a^dag + f_j(t)* a ),
    f_j(t)   = -(eta b_j Omega_j(t) / 2) e^{-i delta t}

Because all sigma_{Phi_j} commute, [H(t1), H(t2)] is a pure spin operator and
the Magnus series TERMINATES at second order -- `MSMagnus` implements the
resulting exact unitary

    U(t) = exp( sum_j sigma_{Phi_j} (alpha_j(t) a^dag - alpha_j(t)* a) )
         * exp( i sum_{jk} Theta_jk(t) sigma_{Phi_j} sigma_{Phi_k} )

    alpha_j(t)  = -i int_0^t f_j(t') dt'                (phase-space trajectory)
    Theta_jk(t) = int_0^t dt1 int_0^t1 dt2 Im[f_j(t1) f_k(t2)*]   (geometric phase)

Pre-RWA (what `ms_lamb_dicke1/2` build -- "stop after the Lamb-Dicke expansion",
keeping the off-resonant carrier and counter-rotating terms):

    H(t) = sum_j Omega_j(t) cos(mu t) [ sigma_{phi_j}                     (carrier, eta^0)
             - eta b_j sigma_{Phi_j} (a e^{-i nu t} + a^dag e^{+i nu t})   (eta^1)
             - (eta^2 b_j^2 / 2) sigma_{phi_j} (a e^{-i nu t} + a^dag e^{+i nu t})^2 ]  (eta^2)

    with (a e^{-i nu t} + a^dag e^{i nu t})^2 = a^2 e^{-2i nu t} + a^dag^2 e^{2i nu t} + 2n + 1.

Applying the mode RWA to the eta^1 line reproduces H_RWA above (verified in
tests to ~1e-6 process fidelity between MSMagnus and the ODE-solved rwa=True
builder). Note all eta^2 terms oscillate fast (mu, mu +- 2nu); their effect is
an off-resonant correction, which is exactly why you simulate them.

The builders return term-layer `Hamiltonian`s with named groups per ion
("carrier_q0", "sdf_q0", "ld2_q0", ...), so error injection is composition:

    H = ms_lamb_dicke1(...) + pauli_term("Z0", coeff=eps_z)     # static sigma_z error
    H_err = H.replace(sdf_q0=my_miscalibrated_drive_terms)      # swap one ion's drive
"""
import numpy as np
from scipy.linalg import expm

from ..core.mechanism import Mechanism
from ..core.operator import Operator
from ..core.subsystems import embed
from ..core.terms import Hamiltonian, hconj, term
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


def _sigma(phi):
    """sigma_phi = cos(phi) sx + sin(phi) sy."""
    return np.cos(phi) * sigma_x + np.sin(phi) * sigma_y


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

    Helpers: `alpha(t)` / `alpha_trajectory(ts)` (per-ion phase-space
    trajectory) and `geometric_phase(t)` (Theta_jk matrix). Integrals are
    computed by (cumulative) trapezoid on a dense grid with
    `points_per_period` points per 2*pi/|detune| (default 400; refine for
    very fast amplitude modulation).

    Truncation caveat: the closed form is exact for the INFINITE-dimensional
    oscillator ([a, a^dag] = 1); here it is evaluated with truncated
    operators. An ODE solve of the truncated H(t) instead propagates the
    truncated model exactly. The two agree on any state far from the Fock
    edge (choose n_max well above max_j |alpha_j|^2 + initial occupation),
    but differ on states near |n_max> -- so compare via low-Fock states, not
    full-space process fidelity.
    """

    def __init__(self, participation, eta, detune, amplitudes, phases,
                 n_max, points_per_period=400):
        self.participation = np.atleast_1d(np.asarray(participation, dtype=float))
        self.n_ions = len(self.participation)
        self.eta = float(eta)
        self.detune = float(detune)
        self.amplitudes = _as_funcs(amplitudes, self.n_ions, "amplitudes")
        self.phases = _as_consts(phases, self.n_ions, "phases (MSMagnus)")
        self.n_max = int(n_max)
        self.points_per_period = int(points_per_period)
        self.subsystems = {f"q{j}": 2 for j in range(self.n_ions)}
        self.subsystems["mode"] = self.n_max + 1
        self._a = annihilation(self.n_max)
        self._S_cache = None
        self._S_key = None

    @property
    def _S(self):
        """sigma_{Phi_j} with Phi = phi + pi/2, embedded on the joint space.

        Derived from `phases`, so the cache is keyed on them: baking these at
        __init__ would leave `unitary()` using stale spin operators while `f()`
        reads live amplitudes -- two halves of the same gate disagreeing about
        the parameters."""
        key = tuple(self.phases)
        if self._S_cache is None or self._S_key != key:
            self._S_cache = [np.asarray(embed(_sigma(phi + np.pi / 2),
                                              self.subsystems, f"q{j}"))
                             for j, phi in enumerate(key)]
            self._S_key = key
        return self._S_cache

    # -- drive kernel -------------------------------------------------------

    def f(self, j, t):
        """f_j(t) = -(eta b_j Omega_j(t)/2) e^{-i delta t} (coefficient of a^dag)."""
        t = np.asarray(t, dtype=float)
        Om = np.vectorize(self.amplitudes[j])(t) if t.ndim else self.amplitudes[j](float(t))
        return -(self.eta * self.participation[j] * Om / 2) * np.exp(-1j * self.detune * t)

    def _grid(self, t):
        period = 2 * np.pi / max(abs(self.detune), 1e-12)
        n = int(max(2001, self.points_per_period * (t / period + 1)))
        return np.linspace(0.0, t, n)

    # -- Magnus ingredients ---------------------------------------------------

    def alpha(self, t) -> np.ndarray:
        """Phase-space displacement per ion at time t: alpha_j(t) = -i int_0^t f_j.
        Returns complex array (n_ions,). The physical displacement of the mode
        conditioned on the sigma_{Phi_j} = +1 branch is +alpha_j (and -alpha_j
        on the -1 branch)."""
        return self.alpha_trajectory(self._grid(float(t)))[-1]

    def alpha_trajectory(self, ts) -> np.ndarray:
        """alpha_j at every time in `ts` (ascending, ts[0] >= 0): complex array
        (len(ts), n_ions). Feed straight into `plot_phase_space`."""
        ts = np.asarray(ts, dtype=float)
        if ts[0] < 0:
            raise ValueError("MSMagnus is defined for t >= 0")
        # `ts` may serve as the quadrature grid only if it already resolves the
        # integral from its lower limit 0 -- otherwise int_0^t f would silently
        # start at ts[0] and every alpha would be off by the missing int_0^ts[0].
        reuse_ts = (len(ts) > 1000 and ts[0] == 0.0
                    and np.allclose(np.diff(ts), ts[1] - ts[0]))
        grid = ts if reuse_ts else np.union1d(self._grid(ts.max()), ts)
        F = np.array([self.f(j, grid) for j in range(self.n_ions)])          # (N, n)
        alpha = -1j * _cumtrapz(F, grid)                                      # (N, n)
        idx = np.searchsorted(grid, ts)
        return alpha[:, idx].T

    def geometric_phase(self, t) -> np.ndarray:
        """Theta_jk(t) = int_0^t dt1 int_0^t1 dt2 Im[f_j(t1) f_k(t2)*] -- the
        accumulated two-spin (and j = k: global) phase matrix (n_ions, n_ions).
        The entangling angle between ions j != k is Theta_jk + Theta_kj."""
        return self._magnus(float(t))[1]

    def _magnus(self, t):
        """(alpha, Theta) at time t from ONE quadrature grid -- both Magnus
        ingredients are built from the same int_0^t1 f_k, so `unitary()` does
        not grid and integrate f twice."""
        grid = self._grid(t)
        F = np.array([self.f(j, grid) for j in range(self.n_ions)])   # (N, n)
        Fint = _cumtrapz(F, grid)                                     # int_0^t1 f_k
        alpha = -1j * Fint[:, -1]
        integrand = np.einsum("jn,kn->jkn", F, Fint.conj()).imag      # Im[f_j(t1) F_k(t1)*]
        Theta = np.trapezoid(integrand, grid, axis=-1)
        return alpha, Theta

    # -- the gate -------------------------------------------------------------

    def unitary(self, t=None) -> Operator:
        """U(t) = exp(Omega1 + Omega2): spin-dependent displacement times the
        geometric-phase gate (the two commute, so one expm of the sum is exact)."""
        if t is None:
            raise ValueError("MSMagnus.unitary needs an explicit time t")
        t = float(t)
        if t < 0:
            raise ValueError("MSMagnus is defined for t >= 0")
        alpha, Theta = self._magnus(t)
        D = np.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.n_ions):
            Mj = np.asarray(embed(alpha[j] * self._a.conj().T - np.conj(alpha[j]) * self._a,
                                  self.subsystems, "mode"))
            D = D + self._S[j] @ Mj                    # sigma_Phi_j (alpha a^dag - alpha* a)
        G = np.zeros_like(D)
        for j in range(self.n_ions):
            for k in range(self.n_ions):
                G = G + 1j * Theta[j, k] * (self._S[j] @ self._S[k])
        return Operator(expm(D + G))

    @property
    def dim(self):
        return 2 ** self.n_ions * (self.n_max + 1)

    def __repr__(self):
        return (f"MSMagnus(N={self.n_ions}, eta={self.eta}, detune={self.detune}, "
                f"phases={self.phases}, n_max={self.n_max})")


def _cumtrapz(y, x):
    """Cumulative trapezoid along the last axis, with y[..., 0] -> 0."""
    dx = np.diff(x)
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    out = np.zeros_like(y)
    out[..., 1:] = np.cumsum(seg, axis=-1)
    return out


# ---------------------------------------------------------------------------
# 2) + 3) stop after the Lamb-Dicke expansion (pre-RWA), order 1 and 2
# ---------------------------------------------------------------------------

def _ms_lamb_dicke(participation, eta, detune, amplitudes, phases, n_max, nu,
                   order, rwa, prefix="q", mode="mode"):
    participation = np.atleast_1d(np.asarray(participation, dtype=float))
    N = len(participation)
    Om = _as_funcs(amplitudes, N, "amplitudes")
    ph = _as_funcs(phases, N, "phases")
    a = annihilation(n_max)
    adag = a.conj().T
    n_op = adag @ a
    if not rwa and nu is None:
        raise ValueError("pre-RWA builders need the trap frequency nu")
    mu = (nu + detune) if nu is not None else None

    H = Hamiltonian({**{f"{prefix}{j}": 2 for j in range(N)}, mode: n_max + 1})
    for j in range(N):
        b, Omj, phj = participation[j], Om[j], ph[j]
        q = f"{prefix}{j}"
        # sigma_phi(t) split into constant sx / sy with time-dependent weights;
        # Phi = phi + pi/2 -> cos(Phi) = -sin(phi), sin(Phi) = cos(phi)
        if rwa:
            # post-RWA spin-dependent force: hconj( sigma_Phi (x) f_j(t) a^dag )
            def fx(t, b=b, Omj=Omj, phj=phj):
                return -(eta * b * Omj(t) / 2) * np.exp(-1j * detune * t) * (-np.sin(phj(t)))
            def fy(t, b=b, Omj=Omj, phj=phj):
                return -(eta * b * Omj(t) / 2) * np.exp(-1j * detune * t) * np.cos(phj(t))
            H = H + hconj(term({q: sigma_x, mode: adag}, coeff=fx, name=f"sdf_{q}")
                          + term({q: sigma_y, mode: adag}, coeff=fy, name=f"sdf_{q}"))
            continue
        # ---- pre-RWA: carrier (eta^0) ----
        def cx(t, Omj=Omj, phj=phj):
            return Omj(t) * np.cos(mu * t) * np.cos(phj(t))
        def cy(t, Omj=Omj, phj=phj):
            return Omj(t) * np.cos(mu * t) * np.sin(phj(t))
        H = H + term(sigma_x, on=q, coeff=cx, name=f"carrier_{q}") \
              + term(sigma_y, on=q, coeff=cy, name=f"carrier_{q}")
        # ---- eta^1: -eta b Omega cos(mu t) sigma_Phi (a e^{-i nu t} + a^dag e^{i nu t}) ----
        def gx(t, b=b, Omj=Omj, phj=phj):
            return -eta * b * Omj(t) * np.cos(mu * t) * np.exp(1j * nu * t) * (-np.sin(phj(t)))
        def gy(t, b=b, Omj=Omj, phj=phj):
            return -eta * b * Omj(t) * np.cos(mu * t) * np.exp(1j * nu * t) * np.cos(phj(t))
        H = H + hconj(term({q: sigma_x, mode: adag}, coeff=gx, name=f"sdf_{q}")
                      + term({q: sigma_y, mode: adag}, coeff=gy, name=f"sdf_{q}"))
        if order >= 2:
            # ---- eta^2: -(eta^2 b^2/2) Omega cos(mu t) sigma_phi
            #             (a^2 e^{-2i nu t} + a^dag^2 e^{2i nu t} + 2n + 1) ----
            pref = -(eta ** 2) * (b ** 2) / 2
            def h2x(t, pref=pref, Omj=Omj, phj=phj):
                return pref * Omj(t) * np.cos(mu * t) * np.exp(2j * nu * t) * np.cos(phj(t))
            def h2y(t, pref=pref, Omj=Omj, phj=phj):
                return pref * Omj(t) * np.cos(mu * t) * np.exp(2j * nu * t) * np.sin(phj(t))
            def h0x(t, pref=pref, Omj=Omj, phj=phj):
                return pref * Omj(t) * np.cos(mu * t) * np.cos(phj(t))
            def h0y(t, pref=pref, Omj=Omj, phj=phj):
                return pref * Omj(t) * np.cos(mu * t) * np.sin(phj(t))
            two_n_plus_1 = 2 * n_op + np.eye(n_max + 1)
            H = H + hconj(term({q: sigma_x, mode: adag @ adag}, coeff=h2x, name=f"ld2_{q}")
                          + term({q: sigma_y, mode: adag @ adag}, coeff=h2y, name=f"ld2_{q}")) \
                  + term({q: sigma_x, mode: two_n_plus_1}, coeff=h0x, name=f"ld2_{q}") \
                  + term({q: sigma_y, mode: two_n_plus_1}, coeff=h0y, name=f"ld2_{q}")
    return H


def ms_lamb_dicke1(participation, eta, detune, amplitudes, phases, n_max,
                   nu=None, rwa=False, prefix="q", mode="mode") -> Hamiltonian:
    """MS Hamiltonian truncated at FIRST order in the Lamb-Dicke expansion,
    as a composable term-layer Hamiltonian (groups: carrier_qj, sdf_qj).

    rwa=False (default): pre-RWA -- keeps the off-resonant carrier and the
    counter-rotating spin-motion terms (needs `nu`). rwa=True: drops them,
    leaving exactly the spin-dependent force whose Magnus closed form is
    `MSMagnus` (used to cross-validate the two). Amplitudes AND phases may be
    time-dependent here."""
    return _ms_lamb_dicke(participation, eta, detune, amplitudes, phases,
                          n_max, nu, order=1, rwa=rwa, prefix=prefix, mode=mode)


def ms_lamb_dicke2(participation, eta, detune, amplitudes, phases, n_max,
                   nu=None, prefix="q", mode="mode") -> Hamiltonian:
    """MS Hamiltonian truncated at SECOND order in the Lamb-Dicke expansion
    (adds groups ld2_qj: the a^2/a^dag^2 and (2n+1) corrections). Pre-RWA by
    construction -- every eta^2 term oscillates fast (mu, mu +- 2nu), so under
    the RWA order 2 would reduce back to order 1; simulating their off-resonant
    effect is the point of this builder. Needs `nu`."""
    return _ms_lamb_dicke(participation, eta, detune, amplitudes, phases,
                          n_max, nu, order=2, rwa=False, prefix=prefix, mode=mode)


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

    `alphas`: complex array (n_times,) or (n_times, n_ions) -- e.g. from
    MSMagnus.alpha_trajectory(ts) or expectation_alpha(evolution, ts). Time is
    always the FIRST axis: shape is never inferred, since (n_times, n_ions) and
    (n_ions, n_times) are indistinguishable whenever both are plausible.
    Start marked with a dot, end with a cross (loop closure check: the cross
    returns to the origin at T = 2*pi/detune for constant amplitude)."""
    import matplotlib.pyplot as plt
    alphas = np.asarray(alphas, dtype=complex)
    if alphas.ndim == 1:
        alphas = alphas[:, None]      # one trajectory
    elif alphas.ndim != 2:
        raise ValueError(f"alphas must be (n_times,) or (n_times, n_ions), "
                         f"got shape {alphas.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    n_traj = alphas.shape[1]
    for j in range(n_traj):
        traj = alphas[:, j]
        label = labels[j] if labels is not None else f"ion {j}"
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
