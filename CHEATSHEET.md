# htdse cheat sheet

One screen per topic: the equation, the call, nothing else. [GUIDE.md](GUIDE.md) walks
the five-step workflow in order; [PHYSICS.md](PHYSICS.md) is the full derivation for
each piece; this page is what you keep open while writing code.

```python
import numpy as np
import htdse as ht
```

---

## Defining a Hamiltonian

### From named pieces (the usual way) — `Model`

$$ H = \sum_k c_k(t)\, O_k $$

Each summand is a coefficient times an operator on one or more *named* subsystems.
`+` merges registries by name and pads identity automatically — nothing is a matrix
until an evolution asks.

```python
from htdse.submodules.spin import sigma_x, sigma_z, sigma_plus
from htdse.submodules.harmonic_oscillator import annihilation, number_operator

a = annihilation(n_max)
H = (ht.term(0.5 * w0 * sigma_z, on="spin", name="atom")
   + ht.term(w * number_operator(n_max), on="mode", name="mode")
   + ht.plus_hc(ht.term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc")))
```

- `coeff` is a number or `f(t)`.
- `name=` is the swap handle: `H.replace(atom=other)`.
- `ht.plus_hc(X)` = `X + X.dag()` — the `g s+ a + h.c.` idiom in one call.
- `ht.hc(X)` = `X.dag()` — JUST the conjugate, as a free function next to `plus_hc`
  instead of a method you have to already know about.
- `ht.show(H, t=0)` — print `H(t)` readably: a Pauli-coefficient table when
  `dim = 2^n`, else the rounded matrix. No more materializing and formatting
  a raw ndarray by hand just to sanity-check what you built.
- `ht.jump(L, on=..., coeff=np.sqrt(gamma), name=...)` — a Lindblad channel, composes
  the same way (`+`), lives in `.jumps` instead of `.groups`.

### Pauli strings, by hand

```python
from htdse.submodules.spin import pauli_term, pauli_sum

H = pauli_term("X0X1", coeff=J) + pauli_term("Z0", coeff=0.1)
H = pauli_sum("0.5 X0X1 + 0.3 Z0 - Z1")     # same thing, one string
```

### A driven spin + motional mode — `spin_boson.py`

$$ H(t) = \tfrac{\Omega(t)}{2}\Big[\sigma_+\, e^{-i(\mu t+\phi)}\prod_m e^{i\eta_m X_m(t)} + \text{h.c.}\Big] $$

One tone, one call — carrier, red/blue sideband, or (two tones) Mølmer–Sørensen, all the
same primitive:

```python
from htdse.submodules.spin_boson import Tone, Mode, driven_spins, explain
explain()   # prints the full tone table + approximation ladder, no source-reading needed

mode = Mode.from_participation(nu=nu, eta=eta_bare, b=1.0, n_max=8)     # one spin
H = driven_spins([Tone(offset=-nu)], ["q0"], [mode], lamb_dicke=1, rwa=True)  # == JC
H_big = driven_spins([Tone(offset=-nu)], ["q0"], [mode], lamb_dicke=1, sparse=True)  # CSR H(t)
```

| you want | `lamb_dicke=` | `rwa=` |
|---|---|---|
| exact, no expansion (returns a `System`, not a `Model`) | `None` | `False` |
| keep η¹ (spin-motion coupling) | `1` | `False` |
| keep η¹+η² (adds Stark-shift-like term) | `2` | `False` |
| resonant term only (JC / anti-JC / MS force) | `1` | `True` |

Two tones at ±(ν+δ) = a Mølmer–Sørensen gate — `ms_tones()` builds the pair,
`ideal_gate()` skips straight to the common case:

```python
from htdse.submodules.molmer_sorensen import ms_tones, ms_closed_form, ideal_gate

tones = ms_tones(nu, delta, Omega, theta=0.0, psi=0.0)
H_rwa = driven_spins(tones, ["q0", "q1"], [mode], lamb_dicke=1, rwa=True)   # ODE-solved
target = ideal_gate(n_ions=2, eta=0.1, delta=0.5, Omega=Omega, n_max=8)     # closed form, no ODE

# same call directly, with the SAME `mode` object above (nu carried but unused by the
# math -- it's what lets one Mode drive both this closed form and H_rwa's ODE solve)
target2 = ms_closed_form(["q0", "q1"], [mode], [delta], amplitudes=Omega, phases=[0.0, 0.0])
target_big = ms_closed_form(["q0", "q1"], [mode], [delta], Omega, [0.0, 0.0], sparse=True)  # CSR unitary

tones_asym = ms_tones(nu, delta, Omega, delta_red=0.65, amp_red=0.9*Omega)  # asymmetric bichromatic drive
                                                                              # (driven_spins/ODE only --
                                                                              # ms_closed_form needs symmetric delta)
```

### Physics that isn't a sum of terms — write a `System`

```python
class GaussianPulse(ht.System):
    def __init__(self, Omega0, sigma):
        self.Omega0, self.sigma = Omega0, sigma
        self.subsystems = {"q": 2}          # opts into the truncation guard

    def unitary(self, t=None):              # implement unitary(t) INSTEAD of
        ...                                 # hamiltonian(t) to skip the ODE solve
```

Inheriting `ht.System` is optional (it's a `Protocol`) — it buys `H()`, `__repr__`,
and clear errors instead of `AttributeError`.

---

## Choosing an evolution

Pick the row matching your equation of motion; everything else about the API is
identical across the four.

| equation | class | construct | query |
|---|---|---|---|
| $i\dot\psi = H\psi$ | `HamiltonianEvolution` | `(system, psi0)` | `.state_at(t)` |
| $i\dot U = HU$ | `UnitaryEvolution` | `(system, dim=d)` | `.unitary_at(t)` |
| $\rho(t)=U\rho_0 U^\dagger$ | `DensityMatrixEvolution` | `(system, rho0)` | `.state_at(t)` |
| $\dot\rho = -i[H,\rho]+\sum_k \mathcal D[L_k]\rho$ | `LindbladEvolution` | `(system, rho0)` | `.state_at(t)` |

```python
ev = ht.HamiltonianEvolution(H, psi0)
psi_T = ev.state_at(T)                 # one time
psis  = ev.state_at(np.linspace(0, T, 200))   # array -- one solve, batched
```

- **Lazy, extend-only.** Nothing integrates until `state_at`/`unitary_at` is called;
  a later call past the solved range continues from the last boundary state — never
  extrapolates. Prints every real integration; silence with `with ht.quiet():`.
- **A `System` is frozen once bound.** Mutating its parameters after building an
  evolution raises rather than silently reusing stale cached segments.
- **`.trace_out(*names, t=...)`** — reduced density matrix on
  `HamiltonianEvolution`/`DensityMatrixEvolution`/`LindbladEvolution`, batched over `t`.
- **`.report(t=None)`** — solved range, propagation method (ODE vs exact
  eigendecomposition), rhs evals, truncation populations, unitarity defect. The
  first thing to call when a result looks wrong.
- **`.unitarity_defect(t)`** on `UnitaryEvolution`/`DensityMatrixEvolution` —
  $\max|U^\dagger U - \mathbb 1|$, the accuracy diagnostic for anything built on $U$.
- **Exact instead of ODE**: a system with `piecewise_constant = True` and
  `breakpoints()` gets propagated by eigendecomposition per interval — faster and
  error-free. `TrotterizedSystem(inner, t0, t1, n_steps)` wraps any system into this.
- **A gate with no `hamiltonian(t)`** (only `.unitary(t)`, e.g. `ms_closed_form`) —
  `UnitaryEvolution`/`DensityMatrixEvolution` consume it directly, no ODE, no
  matrix log.

---

## Composing / injecting an error

```python
noisy = model.replace(drive=drive_noisy)     # swap one named group, keep the rest
H_err = H + pauli_term("Z0", coeff=0.02)     # add a static error term
H.without("carrier_q0")                       # drop a group
H.sparse()                                    # flag for CSR storage (dim gtr~200: worth it)
```

## Reading amplitudes and populations

$$ \langle 00|\psi\rangle $$

```python
ht.bra("00") @ psi        # amplitude <00|psi> -- no .conj()/.T at the call site
abs(ht.bra("00") @ psi)**2   # population
ht.expect(psi, Z)         # <psi|Z|psi> for a ket, Tr(Z rho) for a density matrix --
                           # same call regardless of which Evolution class produced the state
```

## Reading a solved gate — what did it actually implement?

For a state trajectory, `plot_populations`/`bra` above are enough. For a
**propagator/gate** (from `UnitaryEvolution` or a closed form), the question
is usually "what effective Hamiltonian did this implement on the part I
care about, given the rest (e.g. a motional mode) returns to where it
started?" These compose in one direction, each a plain function of the last:

```python
M = ht.project(U, H.subsystems, on="mode", state=0)   # <0|_mode U |0>_mode
c = ht.closure(M)                  # 1.0 = motion actually returned; check before trusting M
H_eff = ht.generator(M, T)         # M ~= exp(-i H_eff T), via logm; Hermitian, traceless
ht.paulis(H_eff)                   # {"XX": ..., "ZI": ...} -- same as pauli_decompose
ht.max_eigenphase(H_eff, T)        # logm branch-cut trust check; want << 1
```

`generator`'s `logm` picks a principal branch per eigenvalue, so once the
gate's eigenphases spread past ~π the extraction wraps and the Pauli
coefficients stop meaning anything — `max_eigenphase` is the number to check
before reading `paulis`, not a certificate that the answer is right.

`magnus`/`magnus_pauli` do a related but different job: perturbative, from
`H(t)` directly, and qubit-only (`dim = 2^n`). This pipeline is the
non-perturbative, post-solve counterpart — it works from the *propagator*,
so it's what you reach for when the system carries a mode.

## Comparing target vs. realized

```python
ht.fidelity(psi1, psi2)                 # |<psi1|psi2>|^2
ht.process_fidelity(U1, U2)             # |Tr(U1^dag U2)|^2 / d^2, phase-blind
ht.density_fidelity(rho, psi)           # <psi|rho|psi>, mixed vs pure

F = ht.compare_over(ts, target_ev, realized_ev, metric=ht.fidelity)
```

`compare_over` takes `target_adapter=`/`realized_adapter=` when the two live on
different Hilbert spaces (embed one up, or trace the other down) — the framework
never guesses which side to adapt.

```python
n_max = ht.converged(lambda n: run_with(n_max=n), values=[6, 8, 10, 12],
                     metric=lambda a, b: 1 - ht.fidelity(a, b))
```

---

## Plotting

```python
from htdse.core.plotting import plot_populations, plot_eigenspectrum, plot_matrix

plot_populations(ts, ev)                 # or a ket/rho trajectory directly
plot_eigenspectrum(ev, ts)               # instantaneous H(t) eigenvalues
plot_matrix(H, t=0, kind="abs")          # heatmap: "abs" (default) / "real" / "imag" / "phase"
```

All four (`plot_populations`, `plot_eigenspectrum`, `plot_matrix`, plus `bra`/
`show`/`project`/`closure`/`generator`/`paulis`/`max_eigenphase` above) are
also importable straight from `htdse` — `ht.plot_matrix(...)`, no submodule path.

Phase-space (Mølmer–Sørensen / any spin-dependent force):

```python
from htdse.submodules.molmer_sorensen import plot_phase_space, expectation_alpha

plot_phase_space(gate.alpha_trajectory(ts))            # analytic, from ms_closed_form
plot_phase_space(expectation_alpha(ev, ts))             # measured <a>(t) from a solve
```

Wigner function of a reduced motional state:

```python
from htdse.submodules.wigner import wigner

rho_mode = ev.trace_out("q0", "q1", t=T)
xs = np.linspace(-3, 3, 200)
W = wigner(rho_mode, xs, xs)             # shape (len(xs), len(xs)); pcolormesh(xs, xs, W)
```

---

## Subsystems: embed / partial_trace

```python
from htdse.core.subsystems import embed, partial_trace

H_full = embed(H_A, dims, "A")                    # H_A (x) I on everything else
rho_A  = partial_trace(rho, dims, ("B", "mode"))   # trace out B and mode
```

`dims` is a `{name: dim}` registry — a `Model`'s `.subsystems`, or an evolution's own.

---

## QuTiP interop (optional, not a dependency)

```python
from htdse.interop.qutip import to_qutip, to_qobj, as_system

H_qt, c_ops = to_qutip(model)                       # Model -> qutip's [H0,[H1,f1],...]
result = qutip.mesolve(H_qt, to_qobj(rho0, model.subsystems), ts, c_ops)

sys = as_system(qobj_or_qobjevo, subsystems={"A": 2})   # qutip -> htdse System
```

Use this to reach `mcsolve`, `steadystate`, `floquet` — deliberately not reimplemented
here — or to pull qutip's state-prep (`coherent`, `thermal_dm`) onto htdse output.

---

## Guards you'll hit on purpose

- Non-Hermitian `H(t0)` — refused at construction.
- A dissipative system (`jump_operators`) handed to a closed-system evolution — refused.
- `state_at` before `t0`, or across a declared discontinuity — refused.
- Fock population reaching the top of a truncated ladder — `TruncationWarning`, not silence.
- A dense `Model` past dim ~200 — one-time `SparseSuggestion`, not an automatic switch.

Full list and *why*: [README.md#guards](README.md#guards).
