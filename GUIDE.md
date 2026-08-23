# htdse user guide

Workflow and reference in one document: read top to bottom once for the five-step
workflow, then the headers work as a lookup while you write code. Rationale and
numerics live in [PHYSICS.md](PHYSICS.md). Runnable copy of the worked example:
[demos/00_guide.ipynb](demos/00_guide.ipynb).

## Minimum viable simulation

```python
import numpy as np, htdse as ht
from htdse.submodules.spin import sigma_x

H = ht.term(0.5 * sigma_x, on="q")                  # a Model
ev = ht.HamiltonianEvolution(H, ht.ket("0"))        # nothing solved yet
psi = ev.state_at(np.linspace(0, np.pi, 50))        # (50, 2) numpy array
```

Everything below is variations on those three lines.

---

The worked example running through the five steps is a Mølmer–Sørensen two-qubit gate:
analytic target vs. a detuned reality.

```python
from htdse.submodules.harmonic_oscillator import fock
from htdse.submodules.spin_boson import Mode, driven_spins
from htdse.submodules.molmer_sorensen import ms_tones, ms_closed_form
from htdse.submodules.spin import pauli_term

delta, eta = 1.0, 0.1                    # gate detuning, Lamb-Dicke parameter
Omega = delta / (eta * np.sqrt(2))       # pi/4 entangling-angle calibration
T = 2 * np.pi / delta                    # loop-closure time
n_max = 12                               # Fock truncation
b = np.array([1, 1]) / np.sqrt(2)        # COM-mode participation
nu = 40.0                                 # trap frequency (pre-RWA builders need it)
```

## Step 1 — compose the target

The target is whatever defines "correct". Here, the analytic Magnus result — a system
defined as a gate (`unitary(t)`), no ODE involved:

```python
mode = Mode.from_participation(nu=nu, eta=eta, b=b, n_max=n_max)
target = ms_closed_form(["q0", "q1"], [mode], [delta], amplitudes=Omega, phases=[0.0, 0.0])
```

For one you build yourself, compose named terms:

```python
H = (ht.term(0.5 * w0 * sigma_z, on="spin", name="atom")
     + ht.term(w * number_op,    on="mode", name="mode")
     + ht.plus_hc(ht.term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc")))
```

Terms tagged `"mode"` land on the same factor; identity-padding is automatic; no matrix
exists until an evolution asks. `coeff` is a number or `f(t)`; `name=` is the swap handle
(`H.replace(atom=other)`); `ht.plus_hc(X)` = `X + X.dag()`, and `ht.hc(X)` = `X.dag()` alone,
as a free function next to `plus_hc` rather than a method you have to already know about.
`ht.jump(L, on=..., coeff=np.sqrt(gamma), name=...)` is a Lindblad channel — composes the
same way, lives in `.jumps` instead of `.groups` (see Recipes below).

`ht.show(H, t=0)` prints `H(t)` readably instead of a raw ndarray — a Pauli-coefficient
table when `dim = 2^n`, else the rounded matrix — so you can sanity-check what you built
before running anything.

**Pauli strings, by hand:**

```python
from htdse.submodules.spin import pauli_term, pauli_sum

H = pauli_term("X0X1", coeff=J) + pauli_term("Z0", coeff=0.1)
H = pauli_sum("0.5 X0X1 + 0.3 Z0 - Z1")     # same thing, one string
```

**A driven spin + motional mode (`spin_boson.py`):** one primitive covers the carrier,
red/blue sideband, and (two tones) Mølmer–Sørensen:

```python
from htdse.submodules.spin_boson import Tone, Mode, driven_spins, explain
explain()   # prints the full tone table + approximation ladder, no source-reading needed

mode1 = Mode.from_participation(nu=nu, eta=eta, b=1.0, n_max=8)     # one spin
H_jc = driven_spins([Tone(offset=-nu)], ["q0"], [mode1], lamb_dicke=1, rwa=True)  # == JC
```

| you want | `lamb_dicke=` | `rwa=` |
|---|---|---|
| exact, no expansion (returns a `System`, not a `Model`) | `None` | `False` |
| keep η¹ (spin-motion coupling) | `1` | `False` |
| keep η¹+η² (Stark-shift-like term, plus cross-mode coupling if >1 mode) | `2` | `False` |
| resonant term only (JC / anti-JC / MS force) | `1` | `True` |

`ideal_gate()` skips straight to the common Mølmer–Sørensen case:

```python
from htdse.submodules.molmer_sorensen import ideal_gate

target2 = ideal_gate(n_ions=2, eta=0.1, delta=0.5, Omega=Omega, n_max=8)   # closed form, no ODE
```

`ms_closed_form(spins, modes, delta, ...)` (used above to build `target`) takes the SAME
`spin_boson.Mode` as `driven_spins` — `nu` is carried but not used by the closed form's own
math, which is what lets one `Mode` drive both this closed form and an ODE cross-check.
`ms_tones(nu, delta, amp, delta_red=..., amp_red=...)` supports an asymmetric bichromatic
drive for `driven_spins`/ODE use; the closed form itself needs a symmetric `delta`.

**Chirped tones** — `Tone(offset=...)` also accepts a callable, `mu(t)`, an instantaneous
(time-dependent) detuning:

```python
chirp = lambda t: -nu + 0.2 + 0.05 * t          # detuning sweeping through resonance
H_chirp = driven_spins([Tone(offset=chirp, amp=Omega)], ["q0"], [mode], lamb_dicke=1)
```

Only at `rwa=False` — RWA's "keep whichever sideband is nearest resonance" has no fixed
answer once the detuning itself moves, so `rwa=True` with a callable `offset` raises.
A constant offset stays free (`Phi(t) = mu*t`, no integration); a callable one is
integrated by quadrature on every RHS evaluation, which is a real solver-speed cost —
worth it only when you need a genuine chirp.

**Physics that isn't a sum of terms — write a `System`:**

```python
class GaussianPulse(ht.System):
    def __init__(self, Omega0, sigma):
        self.Omega0, self.sigma = Omega0, sigma
        self.subsystems = {"q": 2}          # opts into the truncation guard

    def unitary(self, t=None):              # implement unitary(t) INSTEAD of
        ...                                 # hamiltonian(t) to skip the ODE solve
```

Inheriting `ht.System` is optional (it's a `Protocol`) — it buys `H()`, `__repr__`,
and clear errors instead of `AttributeError`. See "Your own system" under Recipes for the
full pattern, including dissipation and the exact-propagation hints.

## Step 2 — build the realized model

Same model, error-bearing pieces swapped in. Here `driven_spins` at
`lamb_dicke=1, rwa=True` -- the ODE-solved counterpart of the closed form above -- with a
5% detuning miscalibration:

```python
eps = 0.05
mode = Mode(nu=nu, eta=eta * b, n_max=n_max)
tones_real = ms_tones(nu, delta * (1 + eps), Omega, theta=[0.0, 0.0])
H_real = driven_spins(tones_real, ["q0", "q1"], [mode], lamb_dicke=1, rwa=True)
```

Realized models are ordinary `Model`s, so error injection is composition — see
"Compose or inject an error" under Recipes.

## Step 3 — evolve

Pick the row matching your equation of motion; everything else about the API is
identical across the four.

| equation | class | construct | query |
|---|---|---|---|
| $i\dot\psi = H\psi$ | `HamiltonianEvolution` | `(system, psi0)` | `.state_at(t)` |
| $i\dot U = HU$ | `UnitaryEvolution` | `(system, dim=d)` | `.unitary_at(t)` |
| $\rho(t)=U\rho_0 U^\dagger$ | `DensityMatrixEvolution` | `(system, rho0)` | `.state_at(t)` |
| $\dot\rho = -i[H,\rho]+\sum_k \mathcal D[L_k]\rho$ | `LindbladEvolution` | `(system, rho0)` | `.state_at(t)` |

`t` is a scalar or an array. Nothing integrates until you ask; asking for a later time
*extends* the existing solve rather than restarting.

```python
from htdse.util import otimes

psi0 = otimes(ht.ket("00"), fock(0, n_max))
with ht.quiet():
    ev = ht.HamiltonianEvolution(H_real, psi0)
    psi_T = ev.state_at(T)

psi_ideal = np.asarray(target.unitary(T)) @ psi0
```

Without `ht.quiet()` every integration prints what it is doing. Leave it on until you
trust the setup.

- **A `System` is frozen once bound.** Mutating its parameters after building an
  evolution raises rather than silently reusing stale cached segments.
- **`.trace_out(*names, t=...)`** — reduced density matrix on
  `HamiltonianEvolution`/`DensityMatrixEvolution`/`LindbladEvolution`, batched over `t`.
- **`.unitarity_defect(t)`** on `UnitaryEvolution`/`DensityMatrixEvolution` —
  $\max|U^\dagger U - \mathbb 1|$, the accuracy diagnostic for anything built on $U$.
- **Exact instead of ODE**: a system with `piecewise_constant = True` and `breakpoints()`
  gets propagated by eigendecomposition per interval — faster and error-free (see
  "Trotterize anything" below).
- **A gate with no `hamiltonian(t)`** (only `.unitary(t)`, e.g. `ms_closed_form`) —
  `UnitaryEvolution`/`DensityMatrixEvolution` consume it directly, no ODE, no matrix log.

## Step 4 — reconcile Hilbert-space mismatches

The realized state lives on spins ⊗ mode; a spin-only target does not. `trace_out` brings
states down, `embed` lifts operators up. Neither is applied for you:

```python
rho_spins = ev.trace_out("mode", t=T)                       # (4, 4) reduced rho
H_big = ht.embed(H_small, {"q0": 2, "q1": 2, "mode": n_max + 1}, ("q0", "q1"))
rho_A  = ht.partial_trace(rho, {"A": 2, "B": 2}, ("B",))     # trace out B, standalone form
```

The registry rides along from the term layer — you never pass a dims dict for a
term-built model. `dims` is any `{name: dim}` registry — a `Model`'s `.subsystems`, or
an evolution's own.

## Step 5 — compare

Metrics are explicit, chosen by what you hold: `fidelity` (kets), `density_fidelity`
(rho vs ket), `process_fidelity` (propagators). There is deliberately no generic
`.compare()` hiding the metric.

```python
print(f"gate fidelity: {ht.fidelity(psi_ideal, psi_T):.4f}")

ts = np.linspace(0, T, 100)
with ht.quiet():
    F = ht.compare_over(ts,
                        ht.UnitaryEvolution(target),        # analytic U(t), no solve
                        ht.UnitaryEvolution(H_real, dim=psi0.shape[0]),
                        metric=ht.process_fidelity)
```

`compare_over` takes `target_adapter=` / `realized_adapter=` when the two sides live on
different spaces (embed one up, or trace the other down) — the framework never guesses
which side to adapt. Plots: `plot_populations(ts, ev)` and `plot_eigenspectrum(ev, ts)` in
`htdse.core.plotting`, phase space via `molmer_sorensen.plot_phase_space`, mode
nonclassicality via `submodules.wigner` — see Plotting below.

---

## Reading amplitudes, populations, and solved gates

$$ \langle 00|\psi\rangle $$

```python
ht.bra("00") @ psi        # amplitude <00|psi> -- no .conj()/.T at the call site
abs(ht.bra("00") @ psi)**2   # population
ht.expect(psi, Z)         # <psi|Z|psi> for a ket, Tr(Z rho) for a density matrix --
                           # same call regardless of which Evolution class produced the state
```

For a **propagator/gate** (from `UnitaryEvolution` or a closed form), the question is
usually "what effective Hamiltonian did this implement on the part I care about, given the
rest (e.g. a motional mode) returns to where it started?" These compose in one direction,
each a plain function of the last:

```python
M = ht.project(U, H.subsystems, on="mode", state=0)   # <0|_mode U |0>_mode
c = ht.closure(M)                  # 1.0 = motion actually returned; check before trusting M
H_eff = ht.generator(M, T)         # M ~= exp(-i H_eff T), via logm; Hermitian, traceless
ht.paulis(H_eff)                   # {"XX": ..., "ZI": ...} -- same as pauli_decompose
ht.max_eigenphase(H_eff, T)        # logm branch-cut trust check; want << 1
```

`generator`'s `logm` picks a principal branch per eigenvalue, so once the gate's
eigenphases spread past ~π the extraction wraps and the Pauli coefficients stop meaning
anything — `max_eigenphase` is the number to check before reading `paulis`, not a
certificate that the answer is right. `magnus`/`magnus_pauli` (see "Checking a run" below)
do a related but different job: perturbative, from `H(t)` directly, and qubit-only
(`dim = 2^n`). This pipeline is the non-perturbative, post-solve counterpart — it works
from the *propagator*, so it's what you reach for when the system carries a mode.

---

## Recipes

**Time-dependent control** — any coefficient can be `f(t)`:

```python
drive = ht.term(0.5 * sigma_x, on="q", coeff=lambda t: Om * np.sin(t), name="drive")
```

From sampled data (a solved pulse), use `ht.sampled_pulse(times, values)`.

**Dissipation** — a jump operator is another named group, pre-scaled by √rate:

```python
open_model = H + ht.jump(a, on="mode", coeff=np.sqrt(gamma), name="decay")
rho_t = ht.LindbladEvolution(open_model, rho0).state_at(ts)
```

**Trotterize anything** — wraps a system into its piecewise-constant version, each step
propagated exactly, so no ODE error contaminates a Trotter-error study:

```python
from htdse.submodules.trotter import TrotterizedSystem
mech = TrotterizedSystem(H_real, 0, T, n_steps=64)
```

**Compose or inject an error** — groups are the handles:

```python
realized = model.replace(drive=noisy_drive)     # swap one named group, keep the rest (unknown name raises)
H_err    = H + pauli_term("Z0", coeff=0.02)     # add a static error term
bare     = model.without("carrier_q0")          # drop a group
one      = model.group("jc")                    # extract a group
```

**Large Hilbert spaces** — flip to sparse; everything downstream follows (sparse matvecs,
`expm_multiply` on the Trotter path). Worth it above dimension ~10³, necessary near 10⁴:

```python
ev = ht.HamiltonianEvolution(H_big.sparse(), psi0)
```

`spin_boson.driven_spins(..., sparse=True)` and `molmer_sorensen.ms_closed_form(...,
sparse=True)` do the same for the exact (`lamb_dicke=None`) and closed-form rungs, which
have no `Model` underneath to call `.sparse()` on afterward. Kets scale; density matrices
and propagators are d×d regardless, so for the biggest spaces stay with
`HamiltonianEvolution`.

**Your own system** — see [README](README.md#extending-it) for the full pattern
(dissipation via `jump_operators(t)`, the `breakpoints()`/`piecewise_constant` exact-propagation
hints, sparse-matrix returns). Systems are frozen once handed to an evolution; mutating
parameters afterwards raises.

**Solver control** — `rtol=`, `atol=`, `method=` pass through to `scipy.solve_ivp`;
`verbose=False` per evolution or `ht.quiet()` globally; `check_mutation=False` skips the
stale-physics guard in an optimizer's inner loop, and only there.

## Plotting

```python
from htdse.core.plotting import plot_populations, plot_eigenspectrum, plot_matrix

plot_populations(ts, ev)                 # or a ket/rho trajectory directly
plot_eigenspectrum(ev, ts)               # instantaneous H(t) eigenvalues
plot_matrix(H, t=0, kind="abs")          # heatmap: "abs" (default) / "real" / "imag" / "phase"
```

All three, plus `bra`/`show`/`project`/`closure`/`generator`/`paulis`/`max_eigenphase`/
`expect` above, are also importable straight from `htdse` (`ht.plot_matrix(...)`, no
submodule path).

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

## Checking a run

**`ev.report()`** — the solved range, how it was propagated, rhs evaluations, guard status,
population at the top of each truncated ladder, unitarity defect, trace. Reads state
already tracked, so it costs nothing and never triggers a solve.

```python
print(ev.report())
#   solved_range    [0, 57.1199]
#   propagation     RK45, rtol=1e-08, atol=1e-10
#   mutation_guard  active
#   truncation      mode=0.101        <- 10% at the ceiling: this run is compromised
```

**Fock truncation.** A truncated mode sets `a†|n_max⟩ = 0`, which the solver cannot detect:
the norm stays 1 and the integration converges cleanly onto the wrong Hamiltonian. So the
check is on the state, and it warns when population reaches the top:

```python
ht.truncation_populations(psi, ev.subsystems)   # {'mode': 3.1e-09} — inspect directly

ht.HamiltonianEvolution(H, psi0, truncation=1e-9)      # stricter threshold
ht.HamiltonianEvolution(H, psi0, ladders=("mode",))    # check only these factors
ht.HamiltonianEvolution(H, psi0, truncation=False)     # off for this evolution
with ht.no_truncation_check(): ...                     # off globally
```

Factors of dimension ≥ 3 are checked by default (a dim-2 factor is a qubit, whose top level
is an ordinary state, not a ceiling); name real ladders with `ladders=` if you have a
genuine qudit. `quiet()` does not suppress these. In a test suite or overnight sweep,
promote them: `warnings.simplefilter("error", ht.TruncationWarning)`.

**`converged(fn, values, tol=)`** — sweeps a setting (`n_max`, `rtol`, Trotter steps) and
stops at the first value where the answer stops moving. Whatever `fn` returns must be
comparable across the sweep: states at different `n_max` live in different-dimensional
spaces, so return a scalar observable or a fidelity against a fixed target. The default
metric raises rather than compare the wrong thing.

```python
print(ht.converged(gate_error, [4, 6, 8, 10, 12], tol=1e-8, parameter="n_max"))
```

**`magnus_pauli(H, T)`** — the effective generator per Magnus order, decomposed over Pauli
strings, so an unintended `YZ` coupling shows up named and sized rather than buried in a
fidelity. Perturbative and qubit-only (`dim = 2^n`) — see `06_what_is_my_pulse_generating.ipynb`;
contrast the propagator-based, non-perturbative pipeline in "Reading amplitudes, populations,
and solved gates" above.

## Talking to QuTiP

`htdse.interop.qutip` is a lazy bridge. qutip is not a dependency; nothing imports it until
you call this.

```python
from htdse.interop.qutip import to_qutip, to_qobj, as_system

H_q, c_ops = to_qutip(model)        # qutip's native [H0, [H1, f1]] form (its fast path)
qutip.mcsolve(H_q, to_qobj(psi0, model.subsystems), ts, c_ops)
```

Compose here, solve there for what htdse does not implement: `mcsolve`, `steadystate`,
`floquet`. The reverse works too — `as_system(qobj)` wraps a QuTiP object so htdse's
evolutions and guards consume it, and qutip's measures take htdse output through `to_qobj`.

htdse's registry is *ordered*, qutip's `dims` is *positional*, and they must agree.
`to_qobj` checks that dimensions multiply out; it cannot check the order.

## Guards you'll hit on purpose

- Non-Hermitian `H(t0)` — refused at construction.
- A dissipative system (`jump_operators`) handed to a closed-system evolution — refused.
- `state_at` before `t0`, or across a declared discontinuity — refused.
- Fock population reaching the top of a truncated ladder — `TruncationWarning`, not silence.
- A dense `Model` past dim ~200 — one-time `SparseSuggestion`, not an automatic switch.

Full list and *why*: [README.md#guards](README.md#guards).

## The demo ladder

| notebook | exercises |
|---|---|
| `00_guide.ipynb` | this document, runnable |
| `01_jaynes_cummings_composition.ipynb` | Step 1: the term layer, `replace()` |
| `02_two_qubit_crosstalk.ipynb` | Step 4: `embed` / `trace_out` |
| `03_motional_dephasing.ipynb` | dissipation, `LindbladEvolution` |
| `04_single_qubit_gate_error.ipynb` | a hand-written `System` |
| `05_ms_two_qubit_gate.ipynb` | the whole stack |
| `06_what_is_my_pulse_generating.ipynb` | `magnus_pauli` |
