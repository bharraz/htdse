# htdse — physics & numerics under the hood

This is the transparency document: for each layer of the package, the physics it
implements, the functions that implement it, and the numerical considerations. Read top
to bottom once; after that the section headers work as a reference.

Prerequisite vocabulary — *subsystem, term, group, registry, System* — is
defined once, in the hierarchy diagram of the [README](README.md). For which functions
to call in what order, see [GUIDE.md](GUIDE.md).

---

## 1. State-vector evolution — the TDSE

**Physics.** ħ = 1 throughout. The time-dependent Schrödinger equation

$$ i\,\frac{d}{dt}\lvert\psi(t)\rangle = H(t)\,\lvert\psi(t)\rangle $$

**Code.** `HamiltonianEvolution(system, psi0, t0=0)`, then `state_at(t)` (scalar or
array of times). Also on it: `trace_out(*names, t=...)` (reduced density matrix, batched
over t), `instantaneous_eigenbasis(t)`, `adiabatic_populations(t)`, `adiabatic_fidelity(t)`.

**Numerics.** Named terms compile once to QuTiP's native constant-operator plus scalar-
coefficient form. QuTiP's `dop853` integrator is the default (`rtol=1e-8`, `atol=1e-10`),
and public results are converted back to NumPy arrays.

Considerations baked into the solver (`core/evolution.py::_QutipSolver`):

- **Lazy + cached.** Nothing integrates until you ask. Requested states are cached; a new
  time outside the cache triggers a real solve from the initial condition, never an
  extrapolation.
- **Frozen systems.** Solved segments are memoized, so mutating a system's parameters
  after binding would silently continue from stale physics. The evolution snapshots the
  system's parameters at construction and raises if they change.
- **Discontinuities.** An adaptive stepper assumes a smooth RHS; a step that straddles a jump
  in H(t) can be accepted with an interpolant fitted across the jump. Systems therefore
  declare `breakpoints()` (e.g. Trotter step edges) and the solver restarts integration at
  each one — no step ever crosses a declared discontinuity.
- **Piecewise-constant fast path.** If a system also sets `piecewise_constant = True`,
  each interval is propagated *exactly*: diagonalize the constant H once
  (`H = V E V†`, Hermitian) and apply $U(\Delta t) = V e^{-iE\Delta t} V^\dagger$. No ODE,
  no stepping error — this is what makes Trotter simulation both fast and honest.
- **Guards.** At construction the evolution checks H(t0) is Hermitian (a sign/conjugation
  error otherwise shows up as mysterious "decay") and refuses systems that carry jump
  operators (closed-system solver would silently ignore the dissipation).
- **Verbosity.** Every real integration prints (system, range, method, tolerances,
  step/eval counts). Wrap optimizer loops in `with htdse.quiet():`.

**Sparse storage (`System.sparse()`).** A term-layer Hamiltonian is a sum of embedded
local operators — Paulis, ladder operators — so its joint matrix is extremely sparse
(fill ~10⁻² at dim 10³, ~10⁻³ at 10⁴). `H.sparse()` flags the model to materialize as
scipy CSR instead of dense ndarrays: each term is embedded via sparse Kronecker products
(non-adjacent factor placement becomes an O(nnz) index remap instead of a dense
reshape/transpose), `hamiltonian(t)` returns CSR, and the RHS above becomes a sparse
matrix–vector product — the ODE solver itself never notices, since it only ever sees the
dense derivative vector. Same physics, verified identical against the dense path in
`tests/test_htdse.py`.

What scales and what doesn't: the *state* stays dense throughout, so a ket evolution is
O(nnz) per RHS call with O(d) memory — dim 10⁴–10⁵ is routine (5 ions × 5 modes at
dim 32768: seconds, ~30 MB for H, where dense would need 17 GB). Propagators and density
matrices are d×d objects regardless of how sparse H is, so U/ρ evolutions gain only in
the products, not in the ceiling. Below dim ~10³ dense is as fast or faster; the toggle
is explicit, sticky under composition (`H.sparse() + other` is sparse), and reversible
(`.sparse(False)`).

---

## 2. Propagator evolution

**Physics.** The propagator obeys the same equation with a matrix initial condition:

$$ i\,\frac{d}{dt}U(t) = H(t)\,U(t), \qquad U(t_0)=\mathbb 1, \qquad
U(t) = \mathcal T\exp\! \Big(-i\! \int_{t_0}^{t}\! H(t')\,dt'\Big) $$

Since $H(t)X$ acts columnwise, the same solver handles it — a ket is just the 1-column case.
(Corollary you can exploit: stack several initial kets as columns and evolve them all in one
solve.)

**Code.** `UnitaryEvolution(system, dim=d)`, `unitary_at(t)`,
`unitarity_defect(t)` = $\max|U^\dagger U - \mathbb 1|$ (how far numerical error has drifted
U off the unitary group — *the* accuracy diagnostic for anything built on U).

**The dual primitive.** $H(t)\to U$ is always well-defined (above). The reverse is not:
$H_{\rm eff} = i\log(U)/t$ is branch-ambiguous (eigenphases fixed only mod $2\pi$) and
collapses real time-dependence into one constant matrix. So a system that is naturally a
*gate* — an analytic Magnus or RWA result, like `ms_closed_form` — implements `.unitary(t)` only,
and `UnitaryEvolution`/`DensityMatrixEvolution` consume it directly with no ODE and no
inversion.

---

## 3. Closed-system density matrices

**Physics.** With no dissipation, a density matrix evolves by conjugation:

$$ \rho(t) = U(t)\,\rho_0\,U^\dagger(t) $$

Mixedness here is *information you set aside* (entanglement with a subsystem you'll trace
out), not information lost to an environment.

**Code.** `DensityMatrixEvolution(system, rho0)` — internally evolves U (one solve) and
conjugates on demand; batched over time via `einsum`. Note ρ inherits U's solver error twice
(U and U†), hence `unitarity_defect` is exposed here too.

---

## 4. Open-system density matrices — Lindblad background

**Why a different equation exists at all.** Sections 1–3 assume the joint state of
everything you track evolves unitarily. That fails when the environment is *not part of the
tracked Hilbert space* — a phonon bath, the EM vacuum, 1/f electrode noise: too large or too
uncharacterized to write an H(t) for. You then posit an equation for the reduced state ρ
directly. Under three standard assumptions — weak coupling (Born), memoryless bath (Markov:
bath correlation time ≪ system timescales), plus a secular/rotating-wave step — the most
general evolution that stays trace-preserving and completely positive is the **GKSL /
Lindblad master equation**:

$$ \dot\rho = -i[H(t),\rho] \;+\; \sum_k \mathcal D[L_k]\,\rho, \qquad
\mathcal D[L]\rho \equiv L\rho L^\dagger - \tfrac12\{L^\dagger L,\,\rho\} $$

How to read it: the commutator is ordinary coherent evolution. Each **jump operator** $L_k$
is one dissipation channel; $L\rho L^\dagger$ is the "quantum jump" (the channel *acting* —
a photon emitted, a phonon absorbed) and $-\tfrac12\{L^\dagger L,\rho\}$ is the matching
no-jump back-action that keeps $\mathrm{Tr}\,\rho = 1$. Convention here: each $L_k$ comes
**pre-scaled by $\sqrt{\text{rate}}$**, so rates never appear separately.

Standard channels, for reference:

| Channel | Jump operator | Effect |
|---|---|---|
| decay / cooling | $\sqrt{\gamma(\bar n{+}1)}\,a$ | relaxes toward the ground state |
| heating | $\sqrt{\gamma\bar n}\,a^\dagger$ | thermal excitation upward |
| pure dephasing (mode) | $\sqrt{\gamma_p}\,a^\dagger a$ | kills coherences, keeps populations |
| qubit decay | $\sqrt{\gamma_1}\,\sigma_-$ | $T_1$ |
| qubit dephasing | $\sqrt{\gamma_\phi/2}\,\sigma_z$ | pure-$T_2$ part |

The dissipator's effect on matrix elements is worth internalizing once. For
$L=\sqrt{\gamma}\,A$: populations get rate equations, and a coherence $\rho_{mn}$ decays at
$\tfrac{\gamma}{2}\big(|A_m|^2{+}|A_n|^2\big)$-type rates set by the anticommutator, while
$L\rho L^\dagger$ *feeds* coherences between other level pairs back in. That interplay is
exactly the $T_2$ story for the thermal mode
(`submodules/harmonic_oscillator.py::ThermalMotionalDecoherence`): the three channels above
give the exact leading-order $|0\rangle,|1\rangle$ coherence decay

$$ \frac{1}{T_2^{\rm lead}} = 2\dot{\bar n} + \frac{\gamma_a}{2} + \frac{\gamma_p}{2},
\qquad \dot{\bar n} = \gamma_a \bar n, $$

a true **lower bound** on the exact $T_2$ (the neglected re-feeding from higher Fock pairs
only slows decay). The commonly quoted $1/(2\dot{\bar n}+\gamma_p/2)$ drops $\gamma_a/2$
and fails as a bound for $\bar n \lesssim 1$.

**Closed vs open decision rule:** a finite subsystem you *can* model (a spectator qubit,
one motional mode) stays in the Hilbert space — unitary evolution + `trace_out`. Lindblad is
only for baths you can't. (The closed-system classes enforce this: they raise on a system
with jump operators.)

**Code.** A system overrides `jump_operators(t) -> [L_k]` (term-layer:
`jump(a, on="mode", coeff=np.sqrt(gamma))` composes like any other group). Then
`LindbladEvolution(system, rho0)`.

**Numerics.** ρ is flattened to a $d^2$ complex vector and the full RHS above is integrated
with the same lazy solver — so cost scales as $d^2$ state size (a 2-qubit ⊗ 15-Fock problem
is a 3600-component ODE; keep $d$ modest or truncate harder). One genuine physics
restriction: **forward-only**. The Lindblad generator is a semigroup — positivity of ρ is
only guaranteed integrating forward; backward integration happily produces negative
eigenvalues. `state_at(t < t0)` is rejected rather than returning a non-physical matrix.

---

## 5. Subsystems: partial trace and embed

**Partial trace.** For $\mathcal H = \mathcal H_1\otimes\cdots\otimes\mathcal H_N$, reshape
ρ into a $2N$-index tensor (one row + one column index per subsystem) and contract the
row/column pair of each traced subsystem:

$$ (\rho_{\rm kept})_{mn} = \sum_a \rho_{(\ldots m \ldots a \ldots),(\ldots n \ldots a \ldots)} $$

`partial_trace(rho, dims, names)` does exactly that reshape+trace, batched over a leading
time axis; every evolution's `trace_out(*names, t=ts)` wraps it. For a pure state the
bipartite special case is the familiar $\rho_A = \Psi\Psi^\dagger$ with
$\Psi_{ma} = \langle m,a|\psi\rangle$.

**Embed.** The inverse-direction adapter: `embed(op, dims, "A")` = $\,op\otimes\mathbb 1$
in the right slot; `embed(M, dims, ("A","C"))` places a *joint* operator on arbitrary — even
non-adjacent — factors (kron with identity, then a reshape/transpose permutation into
registry order). The term layer calls this internally; you only reach for it manually when
comparing operators that live on different spaces.

Which direction to adapt (lift the target up vs reduce the realized state down) is a physics
decision the framework never guesses — you pass it explicitly (see `compare_over`'s
adapters).

---

## 6. Comparison metrics

Explicit functions, chosen by what you hold (there is deliberately no generic `.compare()`
hiding the metric):

$$ F(\psi_1,\psi_2) = |\langle\psi_1|\psi_2\rangle|^2 \qquad\text{(`fidelity`)} 
$$
$$ F(\rho,\psi) = \langle\psi|\rho|\psi\rangle \qquad\text{(`density\_fidelity`, mixed vs pure)} $$

$$ F(U_1,U_2) = \frac{|\mathrm{Tr}(U_1^\dagger U_2)|^2}{d^2} \qquad\text{(`process\_fidelity`, phase-blind)} $$

`compare_over(ts, target, realized, metric, target_adapter=..., realized_adapter=...)` runs
any of these over a time grid, with the embed/trace decision passed in as the adapters.

---

## 7. Trotterization

**Physics.** Discretize $[0,T]$ into $n$ steps and hold H constant on each:

$$ U(T) \approx \prod_{k=n-1}^{0} e^{-i H(t_k^{\rm mid})\,\Delta t} $$

Sampling H at the step *midpoint* makes each factor accurate to $O(\Delta t^3)$ against a
smoothly varying H (midpoint rule), i.e. global error $O(\Delta t^2)$.

**Code.** `TrotterizedSystem(inner, t_start, t_stop, n_steps)` wraps *any* system into
its piecewise-constant version. It declares the step edges as `breakpoints()` (solver never
integrates across an edge) and `piecewise_constant = True` (each step propagated exactly via
QuTiP's diagonal integrator, section 1, while sparse storage remains inside QuTiP). So
"Trotter error" studies compare *only* discretization
physics, with zero ODE-stepping artifacts mixed in — the smooth `inner` evolution and the
Trotterized one are both solved to machine-level accuracy of their respective models.

Error injection composes at the term layer:

```python
H      = driven_spins(tones, spins, modes)        # groups: carrier_q0_tone0, sdf_q0_mode_tone0, ...
H_err  = H + pauli_term("Z0", coeff=eps_z)        # static sigma_z error, just added
mech   = TrotterizedSystem(H_err, 0, T, n)     # discretize the whole thing
```

---

## 8. Spin-boson physics (`submodules/spin_boson.py`, `trapped_ion.py`, `molmer_sorensen.py`)

**MS is not a primitive.** A spin driven by any list of `Tone`s (each a signed offset
$\mu_k$ from resonance, amplitude, phase) coupled to any list of `Mode`s (trap/cavity
frequency $\nu_m$, Lamb-Dicke coupling $\eta_m$) is the general object:

$$ H(t) = \sum_k \tfrac{\Omega_k(t)}{2}\Big[\sigma_+ e^{-i(\mu_k t+\phi_k(t))}
\prod_m e^{i\eta_m X_m(t)} + \text{h.c.}\Big], \qquad
X_m(t) = a_m e^{-i\nu_m t} + a_m^\dagger e^{i\nu_m t} $$

`driven_spins(tones, spins, modes, lamb_dicke=, rwa=)` builds it. MS is what this becomes
for exactly **two** tones, symmetric about a sideband ($\mu=\pm(\nu+\delta)$) —
`molmer_sorensen.ms_tones(nu, delta, amp, theta, psi)` builds that specific pair.

**One tone, after RWA** (`lamb_dicke=1, rwa=True`) — the whole approximation ladder read off
one table:

| tone | result | name |
|---|---|---|
| $\mu=0$ | $\tfrac{\Omega}{2}\sigma_\phi$ | carrier, no motional content |
| $\mu=-\nu$ | $g(\sigma_+a+\text{h.c.})$, $g=i\eta\Omega/2$ | red sideband = Jaynes–Cummings |
| $\mu=+\nu$ | $g(\sigma_+a^\dagger+\text{h.c.})$ | blue sideband = anti-JC |
| $\mu=\pm(\nu+\delta)$, two tones | $\sigma_{\Phi_j}(f_j(t)a^\dagger+f_j^*(t)a)$ | Mølmer–Sørensen |

The factor of $i$ on the JC bridge is not a convention mismatch: the Lamb-Dicke expansion of
$e^{i\eta X}$ starts $1+i\eta X+\dots$, so *recoil* coupling (this table) carries an intrinsic
quarter-turn spin phase relative to a *dipole* coupling written directly as
$g(\sigma_+a+\text{h.c.})$ (`jaynes_cummings`, `rabi` — no $\eta$, no Lamb-Dicke expansion,
never derived from the table above). They are deliberately two code paths, meeting exactly at
this one row.

**Level: stop after the Lamb-Dicke expansion (pre-RWA), `lamb_dicke=1` or `2`.** Expand
$e^{i\eta X(t)}$ in η and keep everything, including the off-resonant carrier and
counter-rotating terms:

$$ H(t) = \sum_j \Omega_j(t)\cos(\mu t+\phi_j)\Big[\underbrace{\sigma_{\phi_j}}_{\text{carrier},\ \eta^0}
\;+\; \underbrace{i\eta_j\,\sigma_{\Phi_j}\big(a e^{-i\nu t} + a^\dagger e^{i\nu t}\big)}_{\eta^1}
\;-\; \underbrace{\tfrac{\eta_j^2}{2}\,\sigma_{\phi_j}\big(a^2 e^{-2i\nu t} + a^{\dagger 2} e^{2i\nu t} + 2\hat n + 1\big)}_{\eta^2}\Big] $$

`driven_spins(..., lamb_dicke=1)` keeps through $\eta^1$; `lamb_dicke=2` through $\eta^2$.
Both return term-layer `System`s with **per-ion, per-tone groups** `carrier_qj_tonek`,
`sdf_qj_modem_tonek`, `ld2_qj_modem_tonek` — so dropping the carrier, miscalibrating one
ion, or swapping a drive is a group operation. Numerical note: these carry oscillations at
$\mu$ and $2\nu$, so the ODE solver has to resolve the trap frequency — cost grows with
$\nu T$. That's inherent to simulating pre-RWA physics.

**Cross-mode $\eta^2$ (more than one mode).** The formula above is one mode's own
$\eta_m^2 X_m^2$ term. Expanding $\prod_m e^{i\eta_m X_m(t)}$ to second order also
produces, for every pair of *different* modes $m\neq m'$, a genuine cross term
$(i\eta_m X_m)(i\eta_{m'}X_{m'}) = -\eta_m\eta_{m'}X_m(t)X_{m'}(t)$ — a coupling the
tone induces BETWEEN two modes, not within one:

$$ X_m(t)X_{m'}(t) = \underbrace{a_m a_{m'}\,e^{-i(\nu_m+\nu_{m'})t} + \text{h.c.}}_{\text{sum frequency}}
\;+\; \underbrace{a_m a_{m'}^\dagger\,e^{-i(\nu_m-\nu_{m'})t} + \text{h.c.}}_{\text{difference frequency}} $$

group `ld2x_qj_modem_modem'_tonek`. This is not optional physics to skip: any
`driven_spins(..., lamb_dicke=2)` call with more than one mode picks it up
automatically (nothing to opt into), since dropping it would silently miss a real
spin-motion-motion coupling whenever two modes share a drive tone. Verified against a
from-scratch expansion of the exact (unexpanded) two-mode displacement-operator
product, at random parameters, away from the same Fock-truncation-edge artifact the
single-mode term already has (`tests/test_molmer_sorensen.py`).

**Level: exact, `lamb_dicke=None`.** No expansion of $e^{i\eta X(t)}$ at all —
`driven_spins(..., lamb_dicke=None)` returns an internal provider (`sigma_+ (x) D(t)`
is a genuinely t-dependent matrix, not a sum of scalar-coefficient × fixed-operator terms,
so this rung has no `+`/`replace()`). Computed via the displacement-operator identity
$e^{i\eta X(t)} = R(\nu t)\,D(i\eta)\,R(\nu t)^\dagger$, $R(\theta)=e^{i\theta a^\dagger a}$
diagonal — one `expm` per (spin, mode) at construction, $O(d^2)$ per call. This is the
reference the Lamb-Dicke expansion is checked against: the carrier Rabi frequency on Fock
state $|n\rangle$ is exactly $\Omega_n = \Omega\,e^{-\eta^2/2}L_n(\eta^2)$ (Laguerre),
which `lamb_dicke=1` (n-independent — $\eta^1$ cannot produce Debye-Waller) and
`lamb_dicke=2` (reproduces the small-$\eta$ expansion $\Omega(1-\eta^2(n+\tfrac12))$) are
both truncations of.

**Level: RWA.** Dropping all fast terms from the $\eta^1$ line (`lamb_dicke=1, rwa=True`)
leaves the spin-dependent force, for the symmetric two-tone MS pair:

$$ H_{\rm RWA}(t) = \sum_j \sigma_{\Phi_j}\big(f_j(t)\,a^\dagger + f_j^*(t)\,a\big),
\qquad f_j(t) = -\tfrac{\eta_j \Omega_j(t)}{2}\,e^{-i\delta t} $$

(All $\eta^2$ terms are fast, so "RWA at second order" would collapse back to this — which is
why `lamb_dicke=2` is pre-RWA only; `rwa=True` is refused there.)

**Level: closed form (`ms_closed_form`).** Because all $\sigma_{\Phi_j}$ commute,
$[H_{\rm RWA}(t_1), H_{\rm RWA}(t_2)]$ is a pure spin operator that commutes with everything,
so the Magnus series **terminates at second order** and the propagator is exact:

$$ U(t) = \exp\! \Big(\sum_j \sigma_{\Phi_j}\big(\alpha_j(t)a^\dagger - \alpha_j^*(t)a\big)\Big)
\exp\! \Big(i\sum_{jk}\Theta_{jk}(t)\,\sigma_{\Phi_j}\sigma_{\Phi_k}\Big) $$

$$ \alpha_j(t) = -i\! \int_0^t\! f_j, \qquad
\Theta_{jk}(t) = \int_0^t\! dt_1\! \int_0^{t_1}\! dt_2\; \mathrm{Im}\big[f_j(t_1)f_k^*(t_2)\big] $$

Read it physically: the first factor is a **spin-dependent displacement** — the mode traces
the phase-space trajectory $\pm\alpha_j(t)$ depending on the $\sigma_{\Phi_j}$ branch (for
constant Ω it's a circle of radius $|f|/\delta$ closing at $T = 2\pi/\delta$). The second is
the **geometric-phase gate**: $\Theta_{jk}$ is (twice) the area swept in phase space, and
$\Theta_{jk}+\Theta_{kj}$ is the entangling angle between ions j,k. `alpha(t)`,
`alpha_trajectory(ts)`, `geometric_phase(t)` expose exactly these; `plot_phase_space` draws
the trajectories (also from measured $\langle a\rangle(t)$ via `expectation_alpha` — evolve a
$\sigma_\Phi$ eigenstate to see one branch, since a superposition averages $\pm\alpha$ to
zero).

(To look at the mode itself rather than its trajectory: `submodules/wigner.py` gives the Wigner
quasiprobability $W(x,p)$ of a Fock-basis ket or reduced ρ — `trace_out` the spins first — with
negativity of $W$ as the visible signature of nonclassicality.)

`ms_closed_form` implements `.unitary(t)` only (an analytic result — section 2's dual
primitive), built by a factory function rather than a public subclass, and its integrals are dense-grid quadrature
(`points_per_period`). Two honesty guards: constant phases required (time-dependent $\phi_j$
breaks the commutator structure that terminated the series — use `driven_spins(...,
rwa=True)` and an ODE solve for that), and remember the closed form is the
infinite-dimensional result: it agrees with an ODE solve of the *truncated* model only on
states away from the Fock edge, so pick `n_max` well above the occupation $|\alpha|^2$
reaches.

**How the levels are meant to be used together:** `ms_closed_form` is the target;
`driven_spins(ms_tones(...), lamb_dicke=1 or 2)` (optionally Trotterized, optionally +
`pauli_term("Z0", ...)` errors, optionally with a swapped/noisy drive group) is the realized
system; `compare_over` with `process_fidelity`/`fidelity` quantifies the gap.
Cross-validation of the suite itself: closed form vs ODE-solved `rwa=True` agree to <1e-7, α
and Θ match their analytic constant-Ω forms, the loop-closure gate equals the pure
geometric-phase gate, the JC/anti-JC/carrier rows of the tone table are each checked
directly, and the exact (`lamb_dicke=None`) rung's carrier Rabi frequency matches the
Laguerre closed form (`tests/test_molmer_sorensen.py`).

---

## 9. The honesty rules (why several of these choices exist)

Numbers read off this framework must be real results of a solve, not artifacts:

1. Never interpolate/extrapolate past solved data — extend the integration instead (§1).
2. Never integrate across a declared discontinuity (§1, §7).
3. Never continue from a mutated system — stale-cache guard (§1).
4. Never silently drop physics — closed-system classes reject dissipative systems (§4);
   non-Hermitian H and invalid ρ₀ are rejected at construction.
5. Never hide what was computed — every integration prints by default (`quiet()` to opt out),
   and every approximation boundary (RWA, LD order, Magnus termination, Fock truncation,
   frame mixing) is either a constructor choice or a warning, not an implicit default.
