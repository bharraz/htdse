# htdse user guide

What to type, in order. Rationale and numerics live in [PHYSICS.md](PHYSICS.md).
Runnable copy: [demos/00_guide.ipynb](demos/00_guide.ipynb).

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
from htdse.submodules.molmer_sorensen import MSMagnus, ms_lamb_dicke1
from htdse.submodules.spin import pauli_term

delta, eta = 1.0, 0.1                    # gate detuning, Lamb-Dicke parameter
Omega = delta / (eta * np.sqrt(2))       # pi/4 entangling-angle calibration
T = 2 * np.pi / delta                    # loop-closure time
n_max = 12                               # Fock truncation
b = np.array([1, 1]) / np.sqrt(2)        # COM-mode participation
```

## Step 1 — compose the target

The target is whatever defines "correct". Here, the analytic Magnus result — a mechanism
defined as a gate (`unitary(t)`), no ODE involved:

```python
target = MSMagnus(b, eta, delta, Omega, [0.0, 0.0], n_max)
```

For one you build yourself, compose named terms:

```python
H = (ht.term(0.5 * w0 * sigma_z, on="spin", name="atom")
     + ht.term(w * number_op,    on="mode", name="mode")
     + ht.plus_hc(ht.term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc")))
```

Terms tagged `"mode"` land on the same factor; identity-padding is automatic; no matrix
exists until an evolution asks.

## Step 2 — build the realized model

Same model, error-bearing pieces swapped in. Here the pre-RWA Lamb-Dicke builder with a
5% detuning miscalibration:

```python
eps = 0.05
H_real = ms_lamb_dicke1(b, eta, delta * (1 + eps), Omega, [0.0, 0.0], n_max, rwa=True)
```

Realized models are ordinary `Model`s, so error injection is composition:

```python
H_real = H_real + pauli_term("Z0", coeff=0.02)      # stray sigma_z on ion 0
noisy  = model.replace(drive=noisy_drive)           # swap one named group
```

## Step 3 — evolve

| Evolving | Class | Query |
|---|---|---|
| a state vector | `HamiltonianEvolution(mech, psi0)` | `state_at(t)` |
| a propagator | `UnitaryEvolution(mech, dim=d)` | `unitary_at(t)` |
| a closed-system density matrix | `DensityMatrixEvolution(mech, rho0)` | `state_at(t)` |
| an open-system density matrix | `LindbladEvolution(mech, rho0)` | `state_at(t)` |

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

## Step 4 — reconcile Hilbert-space mismatches

The realized state lives on spins ⊗ mode; a spin-only target does not. `trace_out` brings
states down, `embed` lifts operators up. Neither is applied for you:

```python
rho_spins = ev.trace_out("mode", t=T)                       # (4, 4) reduced rho
H_big = ht.embed(H_small, {"q0": 2, "q1": 2, "mode": n_max + 1}, ("q0", "q1"))
```

The registry rides along from the term layer — you never pass a dims dict for a
term-built model.

## Step 5 — compare

Metrics are explicit, chosen by what you hold: `fidelity` (kets), `density_fidelity`
(rho vs ket), `process_fidelity` (propagators).

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
different spaces. Plots: `plot_populations(ts, ev)` and `plot_eigenspectrum(ev, ts)` in
`htdse.core.plotting`, phase space via `molmer_sorensen.plot_phase_space`, mode
nonclassicality via `submodules.wigner`.

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

**Trotterize anything** — wraps a mechanism into its piecewise-constant version, each step
propagated exactly, so no ODE error contaminates a Trotter-error study:

```python
from htdse.submodules.trotter import TrotterizedMechanism
mech = TrotterizedMechanism(H_real, 0, T, n_steps=64)
```

**Swap or drop physics** — groups are the handles:

```python
realized = model.replace(drive=noisy_drive)     # swap (unknown name raises)
bare     = model.without("carrier_q0")          # drop
one      = model.group("jc")                    # extract
```

**Large Hilbert spaces** — flip to sparse; everything downstream follows (sparse matvecs,
`expm_multiply` on the Trotter path). Worth it above dimension ~10³, necessary near 10⁴:

```python
ev = ht.HamiltonianEvolution(H_big.sparse(), psi0)
```

Kets scale; density matrices and propagators are d×d regardless, so for the biggest spaces
stay with `HamiltonianEvolution`.

**Your own mechanism** — see [README](README.md#extending-it). Mechanisms are frozen once
handed to an evolution; mutating parameters afterwards raises.

**Solver control** — `rtol=`, `atol=`, `method=` pass through to `scipy.solve_ivp`;
`verbose=False` per evolution or `ht.quiet()` globally; `check_mutation=False` skips the
stale-physics guard in an optimizer's inner loop, and only there.

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
fidelity. See `06_what_is_my_pulse_generating.ipynb`.

## Talking to QuTiP

`htdse.interop.qutip` is a lazy bridge. qutip is not a dependency; nothing imports it until
you call this.

```python
from htdse.interop.qutip import to_qutip, to_qobj, as_mechanism

H_q, c_ops = to_qutip(model)        # qutip's native [H0, [H1, f1]] form (its fast path)
qutip.mcsolve(H_q, to_qobj(psi0, model.subsystems), ts, c_ops)
```

Compose here, solve there for what htdse does not implement: `mcsolve`, `steadystate`,
`floquet`. The reverse works too — `as_mechanism(qobj)` wraps a QuTiP object so htdse's
evolutions and guards consume it, and qutip's measures take htdse output through `to_qobj`.

htdse's registry is *ordered*, qutip's `dims` is *positional*, and they must agree.
`to_qobj` checks that dimensions multiply out; it cannot check the order.

## The demo ladder

| notebook | exercises |
|---|---|
| `00_guide.ipynb` | this document, runnable |
| `01_jaynes_cummings_composition.ipynb` | Step 1: the term layer, `replace()` |
| `02_two_qubit_crosstalk.ipynb` | Step 4: `embed` / `trace_out` |
| `03_motional_dephasing.ipynb` | dissipation, `LindbladEvolution` |
| `04_single_qubit_gate_error.ipynb` | a hand-written `Mechanism` |
| `05_ms_two_qubit_gate.ipynb` | the whole stack |
| `06_what_is_my_pulse_generating.ipynb` | `magnus_pauli` |
