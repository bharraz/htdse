# htdse — Physics & Structure Writeup

This is the "how it actually hangs together" document: each layer of the package, the
physics it implements, the functions that implement it, and the numerical considerations.
Read top to bottom once; after that the section headers work as a reference.

---

## 0. The hierarchy in one picture

Everything in the package is one of five kinds of object, stacked:

```
Operator (np.ndarray + metadata)          <- the universal currency: any matrix/vector
   ^
Term                                       <- coeff(t) x local ops on NAMED subsystems
   ^
Model (term layer)                         <- named GROUPS of terms + subsystem registry
   ^                                          (a composed Hamiltonian +/- dissipation;
                                              is itself a Mechanism)
Mechanism                                  <- anything with .hamiltonian(t) / .unitary(t)
   ^                                          / .jump_operators(t)
Evolution classes                          <- lazily solve ONE equation of motion each
   (HamiltonianEvolution, UnitaryEvolution,
    DensityMatrixEvolution, LindbladEvolution)
```

Here is a concrete Hamiltonian with every level labeled — a driven Jaynes–Cummings model:
$$ H(t) =
\underbrace{\tfrac{\omega_0}{2}\sigma_z}_{\texttt{atom}}
+ \underbrace{\omega\,a^\dagger a}_{\texttt{mode}}
+ \underbrace{g(\sigma_+\otimes a + \mathrm{h.c.})}_{\texttt{jc = hconj(term(...))}}
+ \underbrace{\sin(t)\,\tfrac{\Omega}{2}\sigma_x}_{\texttt{drive, coeff}=f(t)}
 $$


```python
#           ┌ one TERM: coefficient x {subsystem: local operator}
#           │  (coeff may be a scalar or a callable f(t))
atom  = term(0.5 * w0 * sigma_z, on="spin", name="atom")     # GROUP "atom":  1 term
mode  = term(w * n_op,           on="mode", name="mode")     # GROUP "mode":  1 term
jc    = hconj(term({"spin": sigma_plus, "mode": a},          # GROUP "jc":    2 terms
                   coeff=g, name="jc"))                      #   (the term + its h.c.)
drive = term(0.5 * Om * sigma_x, on="spin", name="drive",    # GROUP "drive": 1 term,
             coeff=lambda t: np.sin(t))                      #   time-dependent

H = atom + mode + jc + drive        # a Model: 4 named groups
#   ^ '+' merges the SUBSYSTEM REGISTRY {"spin": 2, "mode": n_max+1}
#     (union by NAME; same name must mean same dimension) and the group dicts.
#     No matrix is built here.

H.hamiltonian(t)                    # MATERIALIZATION: only now does the dense
                                    # 2(n_max+1)-dim matrix exist. Static terms are
                                    # embedded+summed once and cached; time-dependent
                                    # terms are added per call.

H_real = H.replace(drive=noisy_drive)   # groups are SWAP HANDLES: same model,
                                        # one entry replaced (this is the whole
                                        # target-vs-realized workflow)
```

The vocabulary, bottom-up:

- **Operator** (`core/operator.py`) — an `np.ndarray` subclass. A Hamiltonian matrix, a
  ket, a density matrix, a propagator are all Operators; they differ by *role*, not type.
- **Term** (`core/terms.py`) — `coefficient × ⊗(local ops)`, each local op tagged with the
  *name* of the subsystem it acts on. The names are what make composition automatic.
- **Group** — a named list of terms inside a `Model`. Groups exist purely so you can
  grab/replace/delete physics wholesale (`H.replace(drive=...)`, `H.without("ld2_q0")`,
  `H.group("jc")`).
- **Registry** — the `{name: dim}` dict a `Model` carries. It defines the canonical
  tensor-product order (first appearance) and is handed automatically to evolutions, so
  `trace_out`/`embed` never need a separately-maintained dims dict.
- **Model** (`core/terms.py`) — the composable object the term layer builds: named groups of
  terms (a Hamiltonian) plus, optionally, named groups of jump operators (dissipation), over
  one subsystem registry. It is *not* a matrix — it materializes `H(t)` (aka `.H(t)`) and
  `jump_operators(t)` on demand, which is what makes it a `Mechanism`. Named for what it is —
  a model of the system — since a "Hamiltonian" that also carries Lindblad channels would be
  a misnomer.
- **Mechanism** (`core/mechanism.py`) — the protocol the evolution layer consumes. A
  term-layer `Model` satisfies it; so does any hand-written class with a
  `.hamiltonian(t)` (and/or `.unitary(t)`, `.jump_operators(t)`).

---

## 1. State-vector evolution — the TDSE

**Physics.** ħ = 1 throughout. The time-dependent Schrödinger equation

$$ i\,\frac{d}{dt}\lvert\psi(t)\rangle = H(t)\,\lvert\psi(t)\rangle $$

**Code.** `HamiltonianEvolution(mechanism, psi0, t0=0)`, then `state_at(t)` (scalar or
array of times). Also on it: `trace_out(*names, t=...)` (reduced density matrix, batched
over t), `instantaneous_eigenbasis(t)`, `adiabatic_populations(t)`, `adiabatic_fidelity(t)`.

**Numerics.** The state is flattened to a complex vector and handed to an adaptive
Runge–Kutta integrator (`scipy.solve_ivp`, RK45 default, `rtol=1e-8`, `atol=1e-10`,
overridable). The right-hand side is literally `-1j * H(t) @ psi` — every RHS evaluation
calls `mechanism.hamiltonian(t)`, which is why the term layer caches its static part.

Considerations baked into the solver (`core/evolution.py::_ExtendableSolver`):

- **Lazy + extend-only.** Nothing integrates until you ask. `state_at(5.0)` solves
  `[t0, 5]`; a later `state_at(8.0)` *continues* from the stored boundary state at 5 —
  same ODE, never re-solved, and **never extrapolated**: a time outside the solved range
  always triggers a real continuation solve. Times inside the range are evaluated with the
  integrator's own dense output (its internal interpolant, accurate to the solve tolerance).
- **Frozen mechanisms.** Solved segments are memoized, so mutating a mechanism's parameters
  after binding would silently continue from stale physics. The evolution snapshots the
  mechanism's parameters at construction and raises if they change.
- **Discontinuities.** An adaptive stepper assumes a smooth RHS; a step that straddles a jump
  in H(t) can be accepted with an interpolant fitted across the jump. Mechanisms therefore
  declare `breakpoints()` (e.g. Trotter step edges) and the solver restarts integration at
  each one — no step ever crosses a declared discontinuity.
- **Piecewise-constant fast path.** If a mechanism also sets `piecewise_constant = True`,
  each interval is propagated *exactly*: diagonalize the constant H once
  (`H = V E V†`, Hermitian) and apply $U(\Delta t) = V e^{-iE\Delta t} V^\dagger$. No ODE,
  no stepping error — this is what makes Trotter simulation both fast and honest.
- **Guards.** At construction the evolution checks H(t0) is Hermitian (a sign/conjugation
  error otherwise shows up as mysterious "decay") and refuses mechanisms that carry jump
  operators (closed-system solver would silently ignore the dissipation).
- **Verbosity.** Every real integration prints (mechanism, range, method, tolerances,
  step/eval counts). Wrap optimizer loops in `with htdse.quiet():`.

---

## 2. Propagator evolution

**Physics.** The propagator obeys the same equation with a matrix initial condition:

$$ i\,\frac{d}{dt}U(t) = H(t)\,U(t), \qquad U(t_0)=\mathbb 1, \qquad
U(t) = \mathcal T\exp\!\Big(-i\!\int_{t_0}^{t}\! H(t')\,dt'\Big) $$

Since $H(t)X$ acts columnwise, the same solver handles it — a ket is just the 1-column case.
(Corollary you can exploit: stack several initial kets as columns and evolve them all in one
solve.)

**Code.** `UnitaryEvolution(mechanism, dim=d)`, `unitary_at(t)`,
`unitarity_defect(t)` = $\max|U^\dagger U - \mathbb 1|$ (how far numerical error has drifted
U off the unitary group — *the* accuracy diagnostic for anything built on U).

**The dual primitive.** $H(t)\to U$ is always well-defined (above). The reverse is not:
$H_{\rm eff} = i\log(U)/t$ is branch-ambiguous (eigenphases fixed only mod $2\pi$) and
collapses real time-dependence into one constant matrix. So a mechanism that is naturally a
*gate* — an analytic Magnus or RWA result, like `MSMagnus` — implements `.unitary(t)` only,
and `UnitaryEvolution`/`DensityMatrixEvolution` consume it directly with no ODE and no
inversion.

---

## 3. Closed-system density matrices

**Physics.** With no dissipation, a density matrix evolves by conjugation:

$$ \rho(t) = U(t)\,\rho_0\,U^\dagger(t) $$

Mixedness here is *information you set aside* (entanglement with a subsystem you'll trace
out), not information lost to an environment.

**Code.** `DensityMatrixEvolution(mechanism, rho0)` — internally evolves U (one solve) and
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
only for baths you can't. (The closed-system classes enforce this: they raise on a mechanism
with jump operators.)

**Code.** A mechanism overrides `jump_operators(t) -> [L_k]` (term-layer:
`jump(a, on="mode", coeff=np.sqrt(gamma))` composes like any other group). Then
`LindbladEvolution(mechanism, rho0)`.

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

**Code.** `TrotterizedMechanism(inner, t_start, t_stop, n_steps)` wraps *any* mechanism into
its piecewise-constant version. It declares the step edges as `breakpoints()` (solver never
integrates across an edge) and `piecewise_constant = True` (each step propagated exactly via
eigendecomposition, section 1). So "Trotter error" studies compare *only* discretization
physics, with zero ODE-stepping artifacts mixed in — the smooth `inner` evolution and the
Trotterized one are both solved to machine-level accuracy of their respective models.

Error injection composes at the term layer:

```python
H      = ms_lamb_dicke1(...)                      # groups: carrier_q0, sdf_q0, ...
H_err  = H + pauli_term("Z0", coeff=eps_z)        # static sigma_z error, just added
mech   = TrotterizedMechanism(H_err, 0, T, n)     # discretize the whole thing
```

---

## 8. The Mølmer–Sørensen suite (`submodules/molmer_sorensen.py`)

One physical setting — N spins + one motional mode, bichromatic beat at
$\mu = \nu + \delta$, spin phase $\phi_j$, $\Phi_j = \phi_j + \pi/2$ — at three levels of
approximation, all sharing `(participation, eta, detune, amplitudes, phases, n_max[, nu])`.

**Level: stop after the Lamb-Dicke expansion (pre-RWA).** Expand the beam's displacement
operator $e^{i\eta b_j(a e^{-i\nu t} + a^\dagger e^{i\nu t})}$ in η and keep everything —
including the off-resonant carrier and counter-rotating terms:

$$ H(t) = \sum_j \Omega_j(t)\cos(\mu t)\Big[\underbrace{\sigma_{\phi_j}}_{\text{carrier},\ \eta^0}
\;-\; \underbrace{\eta b_j\,\sigma_{\Phi_j}\big(a e^{-i\nu t} + a^\dagger e^{i\nu t}\big)}_{\eta^1}
\;-\; \underbrace{\tfrac{\eta^2 b_j^2}{2}\,\sigma_{\phi_j}\big(a^2 e^{-2i\nu t} + a^{\dagger 2} e^{2i\nu t} + 2\hat n + 1\big)}_{\eta^2}\Big] $$

`ms_lamb_dicke1` keeps through $\eta^1$; `ms_lamb_dicke2` through $\eta^2$. Both return
term-layer `Model`s with **per-ion groups** `carrier_qj`, `sdf_qj`, `ld2_qj` — so
dropping the carrier, miscalibrating one ion, or swapping a drive is a group operation.
Numerical note: these carry oscillations at μ and 2ν, so the ODE solver has to resolve the
trap frequency — cost grows with ν·T. That's inherent to simulating pre-RWA physics.

**Level: RWA.** Dropping all fast terms ($\mu$, $2\nu$, …) from the $\eta^1$ line
(`ms_lamb_dicke1(..., rwa=True)`) leaves the spin-dependent force

$$ H_{\rm RWA}(t) = \sum_j \sigma_{\Phi_j}\big(f_j(t)\,a^\dagger + f_j^*(t)\,a\big),
\qquad f_j(t) = -\tfrac{\eta b_j \Omega_j(t)}{2}\,e^{-i\delta t} $$

(All $\eta^2$ terms are fast, so "RWA at second order" would collapse back to this — which is
why `ms_lamb_dicke2` is pre-RWA by construction.)

**Level: closed form (`MSMagnus`).** Because all $\sigma_{\Phi_j}$ commute,
$[H_{\rm RWA}(t_1), H_{\rm RWA}(t_2)]$ is a pure spin operator that commutes with everything,
so the Magnus series **terminates at second order** and the propagator is exact:

$$ U(t) = \exp\!\Big(\sum_j \sigma_{\Phi_j}\big(\alpha_j(t)a^\dagger - \alpha_j^*(t)a\big)\Big)
\exp\!\Big(i\sum_{jk}\Theta_{jk}(t)\,\sigma_{\Phi_j}\sigma_{\Phi_k}\Big) $$

$$ \alpha_j(t) = -i\!\int_0^t\! f_j, \qquad
\Theta_{jk}(t) = \int_0^t\! dt_1\!\int_0^{t_1}\! dt_2\; \mathrm{Im}\big[f_j(t_1)f_k^*(t_2)\big] $$

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

`MSMagnus` implements `.unitary(t)` only (it *is* a gate — section 2's dual primitive), and
its integrals are dense-grid quadrature (`points_per_period`). Two honesty guards: constant
phases required (time-dependent $\phi_j$ breaks the commutator structure that terminated the
series — use the LD builders for that), and remember the closed form is the
infinite-dimensional result: it agrees with an ODE solve of the *truncated* model only on
states away from the Fock edge, so pick `n_max` well above the occupation $|\alpha|^2$
reaches.

**How the three levels are meant to be used together:** `MSMagnus` is the target;
`ms_lamb_dicke1/2` (optionally Trotterized, optionally + `pauli_term("Z0", ...)` errors,
optionally with a swapped/noisy drive group) is the realized mechanism; `compare_over` with
`process_fidelity`/`fidelity` quantifies the gap. Cross-validation of the suite itself:
Magnus vs ODE-solved `rwa=True` builder agree to <1e-7, α and Θ match their analytic
constant-Ω forms, and the loop-closure gate equals the pure geometric-phase gate
(`tests/test_molmer_sorensen.py`).

---

## 9. The honesty rules (why several of these choices exist)

Numbers read off this framework must be real results of a solve, not artifacts:

1. Never interpolate/extrapolate past solved data — extend the integration instead (§1).
2. Never integrate across a declared discontinuity (§1, §7).
3. Never continue from a mutated mechanism — stale-cache guard (§1).
4. Never silently drop physics — closed-system classes reject dissipative mechanisms (§4);
   non-Hermitian H and invalid ρ₀ are rejected at construction.
5. Never hide what was computed — every integration prints by default (`quiet()` to opt out),
   and every approximation boundary (RWA, LD order, Magnus termination, Fock truncation,
   frame mixing) is either a constructor choice or a warning, not an implicit default.
