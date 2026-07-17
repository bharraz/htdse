# htdse user guide

The steps you take to run a simulation, in order. One worked example carries through all
five steps — the Mølmer–Sørensen two-qubit gate, comparing the analytic target against a
detuned "reality" — followed by short recipes for everything you'd bolt onto that loop.
The physics and numerics behind every step live in [PHYSICS.md](PHYSICS.md); this
document is only about *what to type*.

Runnable copy of this walkthrough: [demos/00_guide.ipynb](demos/00_guide.ipynb).

## Setup

```python
import numpy as np
import htdse as ht
from htdse.submodules.spin import pauli_term
from htdse.submodules.harmonic_oscillator import fock
from htdse.submodules.molmer_sorensen import MSMagnus, ms_lamb_dicke1

delta, eta = 1.0, 0.1                    # gate detuning, Lamb-Dicke parameter
Omega = delta / (eta * np.sqrt(2))       # pi/4 entangling-angle calibration
T = 2 * np.pi / delta                    # loop-closure time
n_max = 12                               # Fock truncation
b = np.array([1, 1]) / np.sqrt(2)        # COM-mode participation
```

## Step 1 — compose the target

The target is whatever defines "correct". Here it's the analytic Magnus result for the
MS gate — a mechanism defined as a *gate* (`.unitary(t)`), no ODE involved:

```python
target = MSMagnus(b, eta, delta, Omega, [0.0, 0.0], n_max)
```

For a target you build yourself, compose it from named terms (this is the term layer —
see the hierarchy diagram in the [README](README.md)):

```python
H = (ht.term(0.5 * w0 * sigma_z, on="spin", name="atom")
     + ht.term(w * number_op,    on="mode", name="mode")
     + ht.hconj(ht.term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc")))
```

Subsystem *names* do the tensor bookkeeping: terms tagged `"mode"` land on the same
factor, identity-padding is automatic, and no matrix exists until an evolution asks.

## Step 2 — build the realized model

Reality is the same model with the error-bearing pieces swapped in. The pre-RWA
Lamb-Dicke builder is the "what the hardware actually does" side; give it a 5% detuning
miscalibration:

```python
eps = 0.05
H_real = ms_lamb_dicke1(b, eta, delta * (1 + eps), Omega, [0.0, 0.0], n_max, rwa=True)
```

Because realized models are ordinary term-layer `Model`s, error injection is composition:

```python
H_real = H_real + pauli_term("Z0", coeff=0.02)      # stray sigma_z on ion 0
noisy  = model.replace(drive=noisy_drive)           # swap one named group wholesale
```

## Step 3 — evolve with the class matching your object

| You are evolving | Class | Query |
|---|---|---|
| a state vector | `HamiltonianEvolution(mech, psi0)` | `state_at(t)` |
| a propagator | `UnitaryEvolution(mech, dim=d)` | `unitary_at(t)` |
| a closed-system density matrix | `DensityMatrixEvolution(mech, rho0)` | `state_at(t)` |
| an open-system density matrix (jump operators) | `LindbladEvolution(mech, rho0)` | `state_at(t)` |

`t` may be a scalar or an array. Nothing integrates until you ask, and asking again for
a later time *extends* the existing solve rather than restarting it.

```python
from htdse.util import otimes

psi0 = ht.Operator(otimes(ht.ket("00"), fock(0, n_max)))   # |00> x |vac>
with ht.quiet():
    ev = ht.HamiltonianEvolution(H_real, psi0)
    psi_T = ev.state_at(T)

psi_ideal = np.asarray(target.unitary(T)) @ np.asarray(psi0)
```

(Without `ht.quiet()` every real integration prints what it's doing — leave that on
until you trust your setup.)

## Step 4 — reconcile Hilbert-space mismatches

The realized state lives on spins ⊗ mode; a spin-only target doesn't. Two explicit
adapters, never guessed: `trace_out` brings realized states down, `embed` lifts
operators up.

```python
rho_spins = ev.trace_out("mode", t=T)                       # (4, 4) reduced rho
H_big = ht.embed(H_small, {"q0": 2, "q1": 2, "mode": n_max + 1}, ("q0", "q1"))
```

The subsystem registry rides along from the term layer automatically — you never pass a
dims dict for a term-built model.

## Step 5 — compare and visualize

Metrics are explicit functions, chosen by what you hold: `fidelity` (kets),
`density_fidelity` (rho vs ket), `process_fidelity` (propagators).

```python
print(f"gate fidelity: {ht.fidelity(psi_ideal, psi_T):.4f}")

ts = np.linspace(0, T, 100)
with ht.quiet():
    F = ht.compare_over(ts,
                        ht.UnitaryEvolution(target),        # analytic U(t), no solve
                        ht.UnitaryEvolution(H_real, dim=psi0.shape[0]),
                        metric=ht.process_fidelity)
```

`compare_over` takes optional `target_adapter=` / `realized_adapter=` callables when the
two sides live on different spaces (this is where a `partial_trace` or an embed goes —
your physics decision, passed explicitly). For plots: `plot_populations(ts, ev)` and
`plot_eigenspectrum(ev, ts)` in `htdse.core.plotting`, phase-space trajectories via
`molmer_sorensen.plot_phase_space`, mode nonclassicality via `submodules.wigner`.

---

## Recipes

**Time-dependent control** — any coefficient can be `f(t)`:

```python
drive = ht.term(0.5 * sigma_x, on="q", coeff=lambda t: Om * np.sin(t), name="drive")
```

**Dissipation** — a jump operator is just another named group (pre-scaled by
√rate), and it composes with `+`; evolving it requires `LindbladEvolution`:

```python
open_model = H + ht.jump(a, on="mode", coeff=np.sqrt(gamma), name="decay")
rho_t = ht.LindbladEvolution(open_model, rho0).state_at(ts)
```

**Trotterize anything** — wraps any mechanism into its piecewise-constant version;
each step is then propagated exactly (no ODE error mixed into your Trotter-error study):

```python
from htdse.submodules.trotter import TrotterizedMechanism
mech = TrotterizedMechanism(H_real, 0, T, n_steps=64)
```

**Inject a coherent error** — literal addition, in the same frame:

```python
H_err = H_real + pauli_term("Z0", coeff=eps_z)
```

**Swap or drop a piece of physics** — groups are the handles:

```python
realized = model.replace(drive=noisy_drive)     # swap (unknown name raises)
bare     = model.without("carrier_q0")          # drop
one      = model.group("jc")                    # extract
```

**Large Hilbert spaces** — flip the model to sparse storage; everything downstream
switches automatically (sparse matvecs, `expm_multiply` on the Trotter path). Worth it
above dimension ~10³; mandatory around 10⁴, where a dense H no longer fits in memory:

```python
ev = ht.HamiltonianEvolution(H_big.sparse(), psi0)
```

Kets scale; density matrices and propagators are d×d objects regardless, so for the
biggest spaces stay with `HamiltonianEvolution`.

**A mechanism the term layer doesn't fit** — subclass `Mechanism` and implement
whichever the physics naturally gives (`hamiltonian(t)`, `unitary(t)`,
`jump_operators(t)`):

```python
class RabiDrive(ht.Mechanism):
    def __init__(self, Omega):
        self.Omega = Omega
    def hamiltonian(self, t):
        return ht.Operator(0.5 * self.Omega * sigma_x)
```

Mechanisms are frozen once handed to an evolution — mutating parameters afterwards
raises (build a new one instead).

**Solver control** — `HamiltonianEvolution(..., rtol=1e-10, atol=1e-12, method="DOP853")`
passes through to `scipy.solve_ivp`; `verbose=False` per evolution or `ht.quiet()`
globally; `check_mutation=False` to skip the stale-physics guard in an optimizer's inner
loop (and only there).

## The demo ladder

In order of increasing complexity, each notebook stating which section it exercises:

| notebook | exercises |
|---|---|
| `00_guide.ipynb` | this document, runnable |
| `01_jaynes_cummings_composition.ipynb` | Step 1: the term layer, `replace()` |
| `02_two_qubit_crosstalk.ipynb` | Step 4: `embed` / `trace_out` |
| `03_motional_dephasing.ipynb` | recipes: dissipation, `LindbladEvolution` |
| `04_single_qubit_gate_error.ipynb` | recipes: a hand-written `Mechanism` |
| `05_ms_two_qubit_gate.ipynb` | the whole stack, all five steps |
