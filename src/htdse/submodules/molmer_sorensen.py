"""Molmer-Sorensen: the symmetric two-tone recipe on top of `spin_boson.py`.

MS is not a primitive. It is what you get from `spin_boson.driven_spins`
when the tone list has exactly two entries, symmetric about a mode's
sideband: blue at +(nu+delta), red at -(nu+delta), phases theta +- psi
(spin phase +- motion phase). `ms_tones()` builds exactly that pair; feed it
to `spin_boson.driven_spins(..., lamb_dicke=1)` for the pre-RWA Hamiltonian,
or `lamb_dicke=None` for the exact one.

This module has ONE mode vocabulary, `spin_boson.Mode(nu, eta, n_max, name)`
-- the same object `driven_spins` takes. `ms_closed_form` needs only `eta`
and the detuning `delta` (given separately, see below) from a `Mode`, not
`nu`; `nu` is still accepted so the SAME `Mode` can drive both the ODE path
(`driven_spins(ms_tones(nu, delta, ...), spins, modes)`) and this closed
form (`ms_closed_form(spins, modes, [delta], ...)`) without redefining the
mode's `eta`/`n_max` twice under two different types.

    from htdse.submodules.spin_boson import Mode, driven_spins
    from htdse.submodules.molmer_sorensen import ms_tones, ms_closed_form

    tones = ms_tones(nu, delta, amplitude, theta=0.0, psi=0.0)
    H = driven_spins(tones, ["q0", "q1"], modes, lamb_dicke=1)          # pre-RWA
    H_rwa = driven_spins(tones, ["q0", "q1"], modes, lamb_dicke=1, rwa=True)  # RWA
    gate = ms_closed_form(["q0", "q1"], modes, [delta], amplitude)      # closed form,
                                                                          # cross-checks H_rwa

Conventions follow C. Monroe, "Primer on Molmer-Sorensen Gates in Trapped
Ions" (2021); equation numbers below refer to it. hbar = 1 throughout.

    spin phase    theta_i = (dphi_B + dphi_R)/2      -- the sigma axis in the XY plane
    motion phase  psi_i   = (dphi_B - dphi_R)/2      -- the phase of the optical force

THE CLOSED FORM. Because all sigma_{Phi_i} (Phi = theta + pi/2) commute and
different modes commute, [H_RWA(t1), H_RWA(t2)] is a pure (scalar) spin
operator and the Magnus series TERMINATES at second order -- `ms_closed_form`
returns the resulting exact unitary directly, no ODE solve:

    U(t) = exp( sum_{i,m} sigma_{Phi_i} (alpha_{i,m} a_m^dag - alpha_{i,m}* a_m)
                + i sum_{ij} Theta_ij sigma_{Phi_i} sigma_{Phi_j} )

    alpha_{i,m}(t) = -i int_0^t f_{i,m}                             (Eq. 25/26)
    Theta_ij(t)    = sum_m int_0^t dt1 int_0^t1 dt2 Im[f_{i,m}(t1) f_{j,m}(t2)*]

This is the exact U for the ONE case where the Magnus series terminates:
symmetric two tones, post-RWA, order eta^1. It is `driven_spins(ms_tones(...),
..., lamb_dicke=1, rwa=True)` solved in closed form instead of by ODE -- the
two are cross-validated against each other in the test suite.

Theta is this package's geometric phase; it SUMS OVER MODES (Eq. 27). The
entangling angle between ions i != j is Theta_ij + Theta_ji, which is Monroe's
chi_ij up to his sign/factor convention (see `entangling_angle`).

Truncation caveat: the closed form is exact for the INFINITE-dimensional
oscillator ([a, a^dag] = 1); here it is evaluated with truncated operators.
An ODE solve of the truncated `driven_spins(..., rwa=True)` instead propagates
the truncated model exactly. The two agree on any state far from the Fock
edge, but differ on states near |n_max> -- compare via low-Fock states, not
full-space process fidelity.
"""
import numpy as np
from scipy import sparse as _sp
from scipy.linalg import expm

from scipy.sparse.linalg import expm as _sparse_expm

from ..core.subsystems import embed
from .harmonic_oscillator import annihilation
from .spin_boson import Mode, Tone, _as_consts, _as_funcs, _eval, _sigma
from ..core.unitary import Unitary


def _prepare(spins, modes, delta):
    """Validate spins/modes/delta and broadcast eta/delta to per-ion arrays.

    Returns (n_ions, etas, deltas): etas[m] and deltas[m] are (n_ions,) float
    arrays for modes[m]."""
    n_ions = len(spins)
    if len(set(spins)) != len(spins):
        raise ValueError(f"spin names must be unique, got {spins}")
    modes = list(modes)
    if not modes:
        raise ValueError("`modes` is empty: an MS drive needs at least one mode")
    for md in modes:
        if not hasattr(md, "eta") or not hasattr(md, "n_max"):
            raise TypeError(f"{md!r} is not a `spin_boson.Mode(nu, eta, n_max, name)`")
    names = [md.name for md in modes]
    if len(set(names)) != len(names):
        raise ValueError(f"mode names must be unique, got {names}")
    if not isinstance(delta, (list, tuple)) or len(delta) != len(modes):
        raise ValueError(
            f"delta must be a list with one entry per mode ({len(modes)} here), "
            f"e.g. delta=[value] for a single mode -- got {delta!r}")
    etas = [np.broadcast_to(np.asarray(md.eta, dtype=float), (n_ions,)).copy() for md in modes]
    deltas = [np.broadcast_to(np.asarray(d, dtype=float), (n_ions,)).copy() for d in delta]
    return n_ions, etas, deltas


def ms_tones(nu, delta, amp, theta=0.0, psi=0.0, delta_red=None, amp_red=None):
    """The two-tone MS drive as a `Tone` pair: blue at +(nu+delta),
    red at -(nu+delta_red), phases theta +- psi. Feed straight into
    `spin_boson.driven_spins(tones, spins, modes, lamb_dicke=...)`.

    theta, psi: scalar, per-ion list, or callable f(t) -- combined pointwise.
    This INVERTS the historical direction: theta/psi (spin/motion phase) are
    now the inputs, and the optical phases phi_blue = theta+psi,
    phi_red = theta-psi are what gets derived, not the other way around.

    delta_red, amp_red: give the red tone its own detuning/amplitude,
    independent of the blue tone's (delta/amp) -- an asymmetric bichromatic
    drive, e.g. for a detuned-carrier Stark shift. None (default): red
    matches blue, today's symmetric drive. There is no separate "asymmetric"
    code path at the `driven_spins` level -- two tones differing is just two
    tones; this is only about which two tones `ms_tones` hands you."""
    mu = nu + delta
    mu_red = -(nu + (delta if delta_red is None else delta_red))
    amp_blue = amp
    amp_red = amp if amp_red is None else amp_red

    def add(a, b, sign):
        """a + sign*b, staying a plain number when both are, else a callable."""
        if callable(a) or callable(b):
            af = a if callable(a) else (lambda t: a)
            bf = b if callable(b) else (lambda t: b)
            return lambda t, af=af, bf=bf, sign=sign: af(t) + sign * bf(t)
        return a + sign * b

    def combine(sign):
        is_list = isinstance(theta, (list, tuple)) or isinstance(psi, (list, tuple))
        if not is_list:
            return add(theta, psi, sign)          # one shared value for every ion
        th = list(theta) if isinstance(theta, (list, tuple)) else [theta]
        ps = list(psi) if isinstance(psi, (list, tuple)) else [psi]
        n = max(len(th), len(ps))
        th = th * n if len(th) == 1 else th
        ps = ps * n if len(ps) == 1 else ps
        return [add(th[i], ps[i], sign) for i in range(n)]

    return [Tone(offset=mu, amp=amp_blue, phase=combine(+1)),
            Tone(offset=mu_red, amp=amp_red, phase=combine(-1))]


def _cumtrapz(y, x):
    """Cumulative trapezoid along the last axis, with y[..., 0] -> 0."""
    dx = np.diff(x)
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    out = np.zeros_like(y)
    out[..., 1:] = np.cumsum(seg, axis=-1)
    return out


def ms_closed_form(spins, modes, delta, amplitudes=1.0, phases=0.0,
                   points_per_period=400, *, motion_phases=0.0,
                   sparse=False):
    """Exact (terminated-Magnus) unitary of the post-RWA MS spin-dependent
    force. Defined as a GATE (`.unitary(t)` only) -- `UnitaryEvolution` and
    `DensityMatrixEvolution` consume it directly with no ODE solve.

    Built as a factory FUNCTION (same convention as
    other physics factory functions), not a class the caller instantiates: the
    physics is not object-oriented, so nothing here is.

    spins: ion subsystem names, e.g. ["q0", "q1"] -- same convention as
        `driven_spins`.
    modes: list of `spin_boson.Mode(nu, eta, n_max, name)`. `nu` is carried
        only so the same `Mode` can be reused with `driven_spins` for an ODE
        cross-check; this closed form's own math never touches it.
    delta: list with ONE entry per mode (`delta[m]` is that mode's detuning
        mu - nu, scalar or per-ion array) -- always a list, even for one
        mode (`delta=[value]`), matching `modes` being always a list. No
        shape-guessing between "one mode" and "one detuning."

    Restrictions inherent to the closed form: `phases` must be CONSTANT per
    ion (time-dependent spin phase breaks the commutator structure that
    terminates the Magnus series -- use `driven_spins(..., rwa=True)` and an
    ODE solve for that), and t >= 0. Time-dependent amplitudes are fine.
    Every mode here is assumed symmetric (one detuning per ion-mode, shared
    by both the blue and red tone) -- an asymmetric `ms_tones(delta_red=...)`
    drive has no closed form; solve it via `driven_spins` + an ODE instead.

    sparse: embed the spin/mode operators as scipy CSR and exponentiate with
        `scipy.sparse.linalg.expm` instead of a dense `scipy.linalg.expm`.
        `.unitary(t)` then returns a CSR matrix (sparse in, sparse out, same
        convention as `System.sparse()`) -- call `.toarray()` if you need the
        dense propagator.

    Returned object's helpers: `.alpha(t)` / `.alpha_trajectory(ts)` (per-ion,
    per-mode phase-space trajectory), `.geometric_phase(t)` (Theta_jk, summed
    over modes) and `.entangling_angle(t)`. Integrals are computed by
    (cumulative) trapezoid on a dense grid with `points_per_period` points per
    period of the FASTEST detuning (default 400; refine for very fast
    amplitude modulation).
    """
    n_ions, etas, deltas = _prepare(spins, modes, delta)
    n_modes = len(modes)
    amp_fns = _as_funcs(amplitudes, n_ions, "amplitudes")
    # SPIN phase must be constant: it sets sigma_{Phi_j}, and a time-dependent
    # sigma breaks the commutator structure that terminates the Magnus series.
    phase_consts = _as_consts(phases, n_ions, "phases (ms_closed_form)")
    # MOTION phase may be time-dependent: it enters only through f_{j,m}(t),
    # never through the spin operators, so [H(t1), H(t2)] stays a pure spin
    # operator and the series still terminates.
    motion_fns = _as_funcs(motion_phases, n_ions, "motion_phases")
    subsystems = {q: 2 for q in spins}
    for md in modes:
        subsystems[md.name] = md.n_max + 1
    a_ops = [annihilation(md.n_max) for md in modes]
    dim = 2 ** n_ions
    for md in modes:
        dim *= md.n_max + 1

    def _emb(op, on):
        return embed(_sp.csr_matrix(op), subsystems, on) if sparse \
            else np.asarray(embed(op, subsystems, on))

    def _zeros():
        return (_sp.csr_matrix((dim, dim), dtype=complex) if sparse
                else np.zeros((dim, dim), dtype=complex))

    # sigma_{Phi_j} with Phi = phi + pi/2, embedded on the joint space -- built
    # eagerly, once: only `.unitary()` pays for it, but it's O(n_ions) embeds,
    # not worth a lazy cache.
    S = [_emb(_sigma(phi + np.pi / 2), q) for phi, q in zip(phase_consts, spins)]

    def f(j, m, t):
        """f_{j,m}(t) = -(eta_{j,m} Omega_j(t)/2) e^{-i(delta_{j,m} t + psi_j)}
        (Monroe Eq. 20). The motion phase psi_j enters here and ONLY here."""
        t = np.asarray(t, dtype=float)
        Om = _eval(amp_fns[j], t)
        psi = _eval(motion_fns[j], t)
        return -(etas[m][j] * Om / 2) * np.exp(-1j * (deltas[m][j] * t + psi))

    def kernel(grid):
        return np.array([[f(j, m, grid) for m in range(n_modes)] for j in range(n_ions)])

    def grid_for(t):
        fastest = max((abs(d) for dl in deltas for d in dl), default=0.0)
        period = 2 * np.pi / max(fastest, 1e-12)
        n = int(max(2001, points_per_period * (t / period + 1)))
        return np.linspace(0.0, t, n)

    def magnus(t):
        """(alpha, Theta) at time t from ONE quadrature grid."""
        grid = grid_for(t)
        F = kernel(grid)                                   # (N, M, n)
        Fint = _cumtrapz(F, grid)                           # int_0^t1 f  (N, M, n)
        alpha = -1j * Fint[:, :, -1]                        # (N, M)
        integrand = np.einsum("jmn,kmn->jkn", F, Fint.conj()).imag
        Theta = np.trapezoid(integrand, grid, axis=-1)      # (N, N)
        return alpha, Theta

    class _MSGate:
        def __init__(self):
            self.subsystems = dict(subsystems)

        def alpha(self, t) -> np.ndarray:
            """Phase-space displacement at time t: complex array (n_ions,
            n_modes). The mode-m displacement conditioned on sigma_{Phi_j}=+1
            is +alpha_{j,m} (and -alpha on the -1 branch). Carries e^{-i psi_j}
            -- Monroe Eq. 26."""
            return magnus(float(t))[0]

        def alpha_trajectory(self, ts) -> np.ndarray:
            """alpha_{j,m} at every time in `ts` (ascending, ts[0] >= 0):
            complex array (len(ts), n_ions, n_modes)."""
            ts = np.asarray(ts, dtype=float)
            if ts[0] < 0:
                raise ValueError("ms_closed_form is defined for t >= 0")
            reuse_ts = (len(ts) > 1000 and ts[0] == 0.0
                        and np.allclose(np.diff(ts), ts[1] - ts[0]))
            grid = ts if reuse_ts else np.union1d(grid_for(ts.max()), ts)
            F = kernel(grid)
            alpha_grid = -1j * _cumtrapz(F, grid)
            idx = np.searchsorted(grid, ts)
            return np.moveaxis(alpha_grid[:, :, idx], -1, 0)

        def geometric_phase(self, t) -> np.ndarray:
            """Theta_jk(t), summed over modes (Monroe Eq. 27):
            (n_ions, n_ions)."""
            return magnus(float(t))[1]

        def entangling_angle(self, t) -> np.ndarray:
            """Theta_jk + Theta_kj -- the two-spin angle multiplying
            sigma_{Phi_j} sigma_{Phi_k} in U. Monroe's chi_jk up to his
            sign/factor convention; maximally entangling is |angle| = pi/4."""
            Th = self.geometric_phase(t)
            return Th + Th.T

        def unitary(self, t=None):
            """U(t) = exp(Omega1 + Omega2): spin-dependent displacement
            (summed over modes) times the geometric-phase gate. Dense
            ndarray, or CSR if built with `sparse=True`."""
            if t is None:
                raise ValueError("ms_closed_form(...).unitary needs an explicit time t")
            t = float(t)
            if t < 0:
                raise ValueError("ms_closed_form is defined for t >= 0")
            alpha, Theta = magnus(t)
            D = _zeros()
            for j in range(n_ions):
                for m, md in enumerate(modes):
                    a = a_ops[m]
                    op = alpha[j, m] * a.conj().T - np.conj(alpha[j, m]) * a
                    Mjm = _emb(op, md.name)
                    D = D + S[j] @ Mjm
            G = _zeros()
            for j in range(n_ions):
                for k in range(n_ions):
                    G = G + 1j * Theta[j, k] * (S[j] @ S[k])
            if sparse:
                return _sparse_expm((D + G).tocsc())
            return np.asarray(expm(D + G))

        @property
        def dim(self):
            return dim

        def __repr__(self):
            mode_str = ", ".join(f"{md.name}(n_max={md.n_max})" for md in modes)
            return (f"ms_closed_form(N={n_ions}, modes=[{mode_str}], "
                   f"phases={phase_consts})")

    gate = _MSGate()
    return Unitary(gate.unitary, dim=dim, subsystems=subsystems,
                   name=repr(gate), alpha=gate.alpha,
                   alpha_trajectory=gate.alpha_trajectory,
                   geometric_phase=gate.geometric_phase,
                   entangling_angle=gate.entangling_angle)


def ideal_gate(n_ions, eta, delta, Omega, n_max, participation=None,
               sparse=False):
    """The common case, in one call: an ideal (constant amplitude, zero spin
    phase) symmetric two-tone MS gate on `n_ions` ions, one mode -- the
    closed-form gate to reach for FIRST, before composing tones/modes by hand
    for a specific error study.

    participation: per-ion coupling weight b_j (default: 1 for every ion --
    pass a normalized array, e.g. [1,1]/sqrt(2) for two ions, for a physical
    COM-mode calibration).

    Builds `spin_boson.Mode(nu=0.0, eta, n_max)` internally -- `nu` plays no
    role in the closed form (see `ms_closed_form`), so this convenience
    wrapper doesn't ask you to supply one; build the `Mode` yourself with its
    real `nu` if you also want the matching `driven_spins` ODE cross-check.

    Equivalent to `ms_closed_form(spins, [Mode(0.0, eta*b, n_max)], [delta], Omega, 0.0)`."""
    if participation is None:
        participation = [1.0] * n_ions
    spins = [f"q{j}" for j in range(n_ions)]
    modes = [Mode.from_participation(nu=0.0, eta=eta, b=participation, n_max=n_max)]
    return ms_closed_form(spins, modes, [delta], Omega, [0.0] * n_ions, sparse=sparse)


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
    (n_times, n_ions, n_modes) stack that `ms_closed_form(...).alpha_trajectory(ts)`
    returns (flattened to one trajectory per ion-mode pair). Also takes
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
