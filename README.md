# htdse

NOTE: Written with significant help from AI (Claude). Built over many revisions, stemming from human design.

A framework for a recurring physics problem: given a **target Hamiltonian** (what a system is
*supposed* to do) and a **mechanism** that produces **realized dynamics** from actual
experimental parameters (Trotterization, adiabatic ramps, motional coupling, amplitude-shaped
pulses, dissipation, or combinations of these), evolve both, compare them, and see how reality
deviates from the target — without rewriting the same TDSE/comparison/plotting scaffolding for
every experiment.

## Design

**Everything that can be represented as an array should be.** A Hamiltonian matrix, a
propagator, a density matrix, a state vector — these are all `Operator(np.ndarray)` instances,
distinguished by role, not by type. Numpy operations always just work on them.

**Five words, defined once** (bottom-up — the composition rules below rely on all five):

- **subsystem** — a named physical factor of the Hilbert space, e.g. `"spin"` (dim 2) or
  `"mode"` (dim `n_max+1`). Just a `(name, dimension)` pair; the *name* is what lets pieces
  line up automatically.
- **term** — one product: a coefficient (number or `f(t)`) × local operators, each tagged with
  the **subsystem** name it acts on (e.g. `½ω₀·σ_z` on `"spin"`).
- **group** — a *named* bundle of terms you can swap or drop as a unit (`atom`, `jc`, `drive`…).
- **registry** — the `{name: dimension}` map of all subsystems a model spans; fixes tensor
  order and rides along to evolutions automatically.
- **Model** — the whole composed object: named **groups** of terms (± jump operators) over one
  **registry**. Not a matrix — it produces `H(t)` on demand, which makes it a **mechanism**
  (the thing an evolution integrates).

**Hamiltonians are composable, by subsystem name.** The term layer (`htdse.term`) builds them
as `Model` objects — the working representation *above* the dense matrix. A `Model` is a sum of named term groups,
each term = coefficient (scalar or `f(t)`) × local operators tagged with named subsystems.
`+` unions the subsystem registries (same name ⇒ same physical subsystem, enforced) and merges
the groups; identity-padding and factor ordering are automatic; the joint dense matrix only
exists when an evolution asks for `H(t)`:

```python
atom = term(0.5 * w0 * sigma_z, on="spin", name="atom")
mode = term(w * number_op,      on="mode", name="mode")
jc   = hconj(term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc"))  # g s+ a + h.c.
H    = atom + mode + jc      # Jaynes–Cummings; names did the embedding
```

Named groups are swap-out handles — the target-vs-reality workflow in one call:

```python
realized = model.replace(drive=noisy_drive)     # same model, one entry swapped
```

Lindblad channels compose identically (`jump(a, on="mode", coeff=np.sqrt(gamma))` is just
another named group), time-dependent control is `coeff=lambda t: ...`, and `f(t) * H` scales a
whole Hamiltonian. Two things the framework will not do for you: it will not check frames
(literal addition is only meaningful with all terms in one frame — tag terms with `frame=` and
mixing tags warns), and it will not apply approximations (composing the dipole coupling
`g*sx⊗(a+a†)` gives the quantum *Rabi* model; Jaynes–Cummings is the post-RWA term you write
yourself, as above).

**A mechanism is just `params -> H(t)`** (and/or `-> U`, and/or jump operators). A term-layer
`Model` *is* a mechanism; hand-written `Mechanism` subclasses work exactly as before.
A mechanism defined only as a gate (analytic Magnus/RWA result) implements `.unitary(t)` and
is consumed directly by `UnitaryEvolution`/`DensityMatrixEvolution` — no lossy U → H inversion.
Mechanisms are **frozen once handed to an evolution**: mutating parameters afterwards is
detected and raises (the memoized solution would silently be stale physics).

**The evolution result is lazy.** Building an evolution integrates nothing. `state_at(t)`
solves only the range actually asked for; querying beyond it *extends* the solve by continuing
from the nearest solved point — never interpolating/extrapolating past real data. Every real
integration prints what it's doing by default; wrap optimization loops in `with htdse.quiet():`.

**Discontinuous H(t) is never integrated across.** A mechanism declares `breakpoints()` (e.g.
Trotter step edges); the solver restarts there, because an adaptive stepper straddling a jump
can silently accept an interpolant fitted to the wrong physics. If it also declares
`piecewise_constant = True`, each interval is propagated *exactly* via the eigendecomposition
of H (no ODE at all) — `TrotterizedMechanism` wraps any mechanism this way.

**Four evolution classes, one per equation of motion:**

| Class | Equation | Use for |
|---|---|---|
| `HamiltonianEvolution` | $i\dot\psi = H(t)\psi$ | evolving a state vector |
| `UnitaryEvolution` | $i\dot U = H(t)U$ | evolving a propagator (`unitary_at`, `unitarity_defect`) |
| `DensityMatrixEvolution` | $\rho(t)=U(t)\rho_0U(t)^\dagger$ | a *closed*-system density matrix |
| `LindbladEvolution` | $\dot\rho=-i[H,\rho]+\sum_k\mathcal D[L_k]\rho$ | an *open*-system density matrix |

All of them validate their inputs at construction (H Hermitian, ρ₀ Hermitian/trace-1) and the
closed-system ones **refuse** a mechanism carrying jump operators rather than silently ignoring
its dissipation. Every time-parametrized method (`state_at`, `trace_out`, `adiabatic_*`) accepts
a scalar t or an array.

**Hilbert-space mismatches: two explicit adapters, never guessed.** `embed(op, dims, names)`
lifts an operator into the joint space — including onto several, possibly *non-adjacent*
factors — and `trace_out`/`partial_trace` (batched over time) brings realized states down.
Term-layer `Model`s carry their own `subsystems` registry, so evolutions and `trace_out`
pick it up automatically. `compare_over(ts, target, realized, metric, ...)` runs the
target-vs-realized loop with explicit metrics (`fidelity`, `process_fidelity`,
`density_fidelity`) and optional per-side embed/trace adapters.

**Package layout**

```
htdse/
  src/htdse/
    core/            # universal: Operator, Mechanism, terms (composable `Model`s),
                     # the four evolution classes, subsystems (embed/partial_trace),
                     # compare_over, plotting, config (quiet)
    submodules/      # reusable physics: spin (Paulis, sigma±, pauli_term/pauli_sum),
                     # harmonic_oscillator (a, a†, n, Fock, ThermalMotionalDecoherence),
                     # trotter (TrotterizedMechanism),
                     # molmer_sorensen (MS suite: MSMagnus analytic gate with alpha(t)/
                     # Theta(t) helpers; ms_lamb_dicke1/2 pre-RWA term-layer builders
                     # with per-ion swappable groups; phase-space trajectory plotting),
                     # wigner (Wigner function W(x,p) of a Fock-basis mode, + plot)
    util.py          # otimes, ket, fidelity, projector, ... (no physics-domain assumptions)
  demos/             # worked notebooks (see Workflow)
  tests/             # `python tests/test_htdse.py`           — 46 checks incl. analytic anchors
                     # `python tests/test_molmer_sorensen.py` — 18 checks, MS cross-validation
```

## Physics

**Schrödinger equation** ($\hbar = 1$):
$i\,d\lvert\psi\rangle/dt = H(t)\lvert\psi\rangle$, propagator
$U(t,0)=\mathcal T\exp(-i\int_0^t H)$. $H(t)\to U$ is always computable; $U\to H$ is
branch-ambiguous ($\log$ of a unitary), which is why mechanisms expose `.hamiltonian(t)`
and/or `.unitary(t)` independently.

**Partial trace / embed** as before: $\rho_A=\Psi\Psi^\dagger$ generalized to any number of
named subsystems and batched trajectories; embedding is $H\otimes I$ generalized to arbitrary
(non-adjacent) factor placement.

**Fidelities**: $|\langle\psi_1|\psi_2\rangle|^2$ (pure), $\langle\psi|\rho|\psi\rangle$
(mixed vs pure), $|\mathrm{Tr}(U_1^\dagger U_2)|^2/d^2$ (process).

**Adiabatic diagnostics**: populations in the instantaneous eigenbasis of $H(t)$. Caveat: at a
(near-)degeneracy, eigh's ordering/basis is arbitrary, so per-level quantities can jump exactly
where gaps close.

### Open-system (Lindblad) evolution

Same GKSL equation and conventions as before ($L_k$ pre-scaled by $\sqrt{\text{rate}}$),
forward-only (backward integration of a Lindbladian isn't positivity-preserving).

**Damped, dephased thermal mode** ($L_1=\sqrt{\gamma_a\bar n}\,a^\dagger$,
$L_2=\sqrt{\gamma_a(\bar n+1)}\,a$, $L_3=\sqrt{\gamma_p}\,\hat n$): the exact leading-order
decay rate of the $|0\rangle,|1\rangle$ coherence follows directly from the dissipators
(cooling $\gamma_a(\bar n+1)/2$, heating $\tfrac32\gamma_a\bar n$, dephasing $\gamma_p/2$):

$$ \frac{1}{T_2^{\text{lead}}} = 2\dot{\bar n} + \frac{\gamma_a}{2} + \frac{\gamma_p}{2},
\qquad \dot{\bar n}=\gamma_a\bar n $$

This is a true **lower bound** on the exact $T_2$ — the only correction (heating re-feeding
coherence from higher Fock pairs) slows the decay. The often-quoted
$T_2\approx 1/(2\dot{\bar n}+\gamma_p/2)$ drops $\gamma_a/2$ and is **not** a bound in general:
verified numerically, at $\bar n \lesssim 1$ the exact $T_2$ falls below it (5× below at
$\bar n=0.05$); it is only valid for $\bar n \gg 1$.

## Workflow

1. **Compose your target model** from terms (or reuse `submodules/`; hand-written `Mechanism`
   subclasses still work for anything the term layer doesn't fit). Build the **realized**
   version with `model.replace(...)` — same model, error-bearing entries swapped in, including
   dissipative ones.
2. **Evolve** with the class matching your object (ket / propagator / closed ρ / open ρ).
   Subsystem structure rides along from the term layer; reconcile any remaining space mismatch
   explicitly with `embed`/`trace_out`.
3. **Compare and visualize**: `compare_over` with an explicit fidelity, `plot_populations`
   (accepts an evolution, a ket trajectory, or a ρ trajectory), `plot_eigenspectrum`.

`demos/`, in order of increasing complexity:

- **`single_qubit_gate_error.ipynb`** — the core loop, no subsystems.
- **`two_qubit_crosstalk.ipynb`** — embed/trace_out across a Hilbert-space mismatch.
- **`jaynes_cummings_composition.ipynb`** — the term layer: JC composed by name, vacuum Rabi
  check against $\cos^2(gt)$, drive swap-out via `replace()`.
- **`motional_dephasing.ipynb`** — `LindbladEvolution`, the corrected $T_2$ bound in practice.
- **`ms_two_qubit_gate.ipynb`** — the whole stack: `MSMagnus` as target, the pre-RWA Lamb-Dicke
  builders as reality, phase-space trajectories and Wigner functions of the mode.
