# htdse

NOTE: Written with significant help from AI (Claude). Built over many revisions, stemming from human design.

A small, transparent time-dependent Schrödinger/Lindblad solver.

- **Transparent.** No wrapper types. You hand it numpy arrays and get numpy arrays back
  (scipy CSR if you asked for sparse). Every intermediate is inspectable, and every
  approximation is a constructor argument rather than a hidden default.
- **Lightweight.** numpy, scipy, matplotlib. 
- **Extensible.** A `System` is anything with `hamiltonian(t)` (or `unitary(t)`).

It composes Hamiltonians from *named* pieces, so building a variant of a model (with an
error term, a swapped drive, a different approximation) is a one-line edit rather than a
rewrite. Comparing the variant to the original is what most of the package is for.

## Install

```
pip install -e .
```

## Quickstart

**Example**: A Rabi drive, and the same drive with a 5% amplitude error plus a stray detuning:

```python
import numpy as np
import htdse as ht
from htdse.submodules.spin import sigma_x, sigma_z # submodules add convenience

target = ht.term(0.5 * sigma_x, on="q", name="drive")          # H = (Omega/2) sigma_x

noisy = (ht.term(0.5 * 1.05 * sigma_x, on="q", name="drive")   # 5% amplitude error
         + ht.term(0.02 * sigma_z, on="q", name="detuning"))   # + stray detuning
realized = target.replace(drive=noisy)                          # same model, one group swapped

ts = np.linspace(0, 4 * np.pi, 200)
with ht.quiet():
    F = ht.compare_over(ts,     # Compares the two evolutions over the passed times, given the passed metric
                        ht.HamiltonianEvolution(target, ht.ket("0")),
                        ht.HamiltonianEvolution(realized, ht.ket("0")),
                        metric=ht.fidelity)
print(f"worst-case fidelity: {F.min():.4f}")
```

## The five things you need

The package exports about thirty names. These five cover most work; everything else is
either a convenience or an escape hatch you will find when you need it.

| | |
|---|---|
| `ht.term(op, on="name")` | one piece of a Hamiltonian, tagged with the subsystem it acts on. Returns a Model. |
| `+` | compose pieces into a `Model` (names do the tensor bookkeeping) |
| `ht.HamiltonianEvolution(model, psi0)` | solve it (or `Unitary` / `DensityMatrix` / `Lindblad`) |
| `.state_at(t)` | the answer, at a time or an array of times |
| `ht.fidelity(a, b)` | compare two answers |

## The hierarchy

**You evolve a System.** That is the one sentence to remember. A `System` is anything that
answers "what are the dynamics at time `t`?" — it implements `hamiltonian(t)` and/or
`unitary(t)`. Nothing else is required.

`Model` is not a layer above or below that. It is one *convenient way* to build a System:
you write your physics as a sum of named terms and it handles the tensor bookkeeping,
caching, and swapping for you. A hand-written class is the other way, for physics that
isn't a sum of terms.

```mermaid
%%{init: {"flowchart": {"rankSpacing": 50, "nodeSpacing": 40}}}%%
flowchart TB
    OP["<b>numpy array</b><br/>a Hamiltonian, a ket, a density matrix, a propagator."]
    MODEL["<b>Model</b> — the convenient path<br/>
    named groups of terms over a registry of subsystems, both dicts keyed by label.<br/>Group labels replace/retrieve/remove physics; subsystem labels fix the embedding order.<br/>Not a matrix — it builds H(t) on demand<br/><i>built by term() / jump() / pauli_sum() / ms_lamb_dicke1() ...</i>"]

    OWN["<b>your own class</b> — the general path for physics that isn't a sum of terms:<br/>a closed-form gate, a wrapper, a bridge<br/><i>MSMagnus, TrotterizedSystem, as_mechanism(Qobj)</i>"]

    SYS["<b>System</b> - a Protocol <br/>hamiltonian(t) and/or unitary(t), plus optional jump_operators(t)<br/>"]

    EVOALL["<b>Evolution</b> — every one works the same way<br/><b>you give it:</b> a System, a starting point, and optionally a start time<br/><b>you ask it:</b> state_at(t) for the answer at one time, or at every time in an array<br/>also report() for what the solve did, and trace_out(name) to discard a subsystem<br/>"]

    subgraph EVO["pick the one matching your equation of motion"]
        direction LR
        E1["<b>HamiltonianEvolution</b><br/>i ψ' = H ψ<br/>—<br/>start from a state vector<br/>get back the state at t"]
        E2["<b>UnitaryEvolution</b><br/>i U' = H U<br/>—<br/>start from just the dimension<br/>get back the propagator at t"]
        E3["<b>DensityMatrixEvolution</b><br/>ρ(t) = U ρ₀ U†<br/>—<br/>start from a density matrix<br/>get back ρ at t, closed system"]
        E4["<b>LindbladEvolution</b><br/>ρ' = −i[H,ρ] + Σ D[L]ρ<br/>—<br/>start from a density matrix<br/>get back ρ at t, with dissipation"]
    end

    OP --> MODEL
    OP --> OWN
    MODEL -- "satisfies" --> SYS
    OWN -- "satisfies" --> SYS
    SYS -- "is integrated by" --> EVOALL
    EVOALL --> EVO
```

Internally a `Model` stores each summand as a private `_Term`, because a coefficient that
is `f(t)` can't be folded into a matrix until you know `t`. You never construct or see one.

The load-bearing idea in the `Model` path is the **subsystem name**. Two operators tagged
`"spin"` act on the same tensor factor, so `+` lines them up and identity-pads
automatically. You never write `⊗ I` by hand, and no joint matrix exists until an evolution
asks for `H(t)`.

```python
atom = term(0.5 * w0 * sigma_z, on="spin", name="atom")
mode = term(w * number_op,      on="mode", name="mode")
jc   = plus_hc(term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc"))  # g s+ a + h.c.
H    = atom + mode + jc      # Jaynes–Cummings; names did the embedding
```

## Extending it

Two ways to build a System:

| Your physics is… | You write… | Examples in the package |
|---|---|---|
| a sum of named pieces | **a `Model`** | `ms_lamb_dicke1`, `pauli_sum`, `term`, `jump` |
| a closed-form `U(t)`, or a wrapper | **a class satisfying `System`** | `MSMagnus`, `TrotterizedSystem`, `as_mechanism` |

A common pattern for using `Model` would be something like: 

```python
def my_drive(Omega, eps, delta) -> ht.Model:
    return (ht.term(0.5 * Omega * (1 + eps) * sigma_x, on="q", name="drive")
            + ht.term(delta * sigma_z, on="q", name="detuning"))
```

Call it with different arguments to get a different System. You never mutate a system, which
matters because an evolution freezes its System at binding and rejects later edits.

When the physics isn't a sum of terms, or a Unitary of specific form, write the class:

A shaped resonant pulse is the clearest case. Every `H(t) = (Ω(t)/2)σx` commutes with
itself at different times, so the time-ordered exponential collapses to the **pulse area**
— `U(t)` is closed-form and integrating an ODE for it would be wasted work. A `Model`
cannot express this: a Model is a sum, and it only ever produces `H(t)`.

```python
from scipy.special import erf

class GaussianPulse(ht.System):
    """Resonant Gaussian pulse. All H(t) commute, so U depends only on the
    accumulated area theta(t) = integral of Omega -- no ODE needed."""
    def __init__(self, Omega0, sigma):
        self.Omega0, self.sigma = Omega0, sigma
        self.subsystems = {"q": 2}                  # opts into the truncation guard

    def unitary(self, t=None):
        area = 0.5 * self.Omega0 * self.sigma * np.sqrt(2 * np.pi) * \
               (erf(t / (self.sigma * np.sqrt(2))) + 1)
        return np.cos(area / 2) * np.eye(2) - 1j * np.sin(area / 2) * sigma_x

ev = ht.UnitaryEvolution(GaussianPulse(Omega0=1.0, sigma=2.0), dim=2)
ev.unitary_at(6.0)          # returned directly -- the ODE solver never runs
```

Implementing `unitary(t)` instead of `hamiltonian(t)` is what tells the solver to skip the
integration entirely. (The reverse doesn't work: `H → U` is always well-defined, but
`U → H` needs a matrix log, which is branch-ambiguous.)

Dissipation is the other case. Add `jump_operators(t)` and `LindbladEvolution` picks it up
— a bath too large to model as a subsystem, with a rate you can make time-dependent:

```python
class Heating(ht.System):
    """Motional heating whose rate ramps during the gate."""
    def __init__(self, n_max, gamma):
        self.n_max, self.gamma = n_max, gamma
        self.subsystems = {"mode": n_max + 1}
        self._adag = creation(n_max)

    def hamiltonian(self, t):
        return np.zeros((self.n_max + 1,) * 2, dtype=complex)   # pure decoherence

    def jump_operators(self, t):
        return [np.sqrt(self.gamma * t) * self._adag]           # sqrt(rate) convention
```

Two optional hints — `breakpoints()` and `piecewise_constant` — tell the solver where
`H(t)` jumps and whether it is constant between jumps. Declaring both buys exact
propagation instead of adaptive stepping, which is how `TrotterizedSystem` works.

Inheriting `ht.System` is optional: it's a `Protocol`, and the evolutions duck-type every
attribute they read. Subclassing `ht.System` buys you the `H()` alias, a readable `__repr__`, and clear
errors instead of `AttributeError`.

Leveraging the conveniences baked into the `Model` class when writing your own system is easy: 
- **Sparce Matrices**: The solver branches on whether the matrix *you returned* is sparse, so return a CSR and you get the sparce path.
- **Truncation Guard**: expose a `subsystems` dict (as `MSMagnus` does) or pass `subsystems=` to the evolution.

When writing your own system, what you give up is the `Model` algebra: `+`,
`.replace()`, `.without()`, automatic identity padding, and the materialization cache. For
a closed-form `U(t)` most of that is moot anyway, and `embed()` is still available as a standalone utility function.

Everything in `submodules/` is written against this same protocol, with no privileged
access. `molmer_sorensen` is the largest example if you want a template. Submodule contributions are encouraged!

## Guards

The solver refuses several things instead of silently returning a plausible wrong answer:
a non-Hermitian `H`, an invalid `rho0`, a dissipative system handed to a closed-system
solver, integration across a declared discontinuity, extrapolation past solved data, and a
system mutated after binding. Population reaching the top of a truncated ladder raises a
`TruncationWarning`. See [GUIDE.md](GUIDE.md#checking-a-run).

## Where to go

| You want | Go to |
|---|---|
| To run your first simulation, step by step | [GUIDE.md](GUIDE.md) |
| The physics and numerics under the hood | [PHYSICS.md](PHYSICS.md) |
| Worked examples, increasing complexity | [demos/](demos/) |
| To use QuTiP for part of the job | `htdse.interop.qutip` — [GUIDE.md](GUIDE.md#talking-to-qutip) |
| What a function does exactly | its docstring — written as the reference manual |
| To update code written against an older version | [MIGRATION.md](MIGRATION.md) |

**Package layout**

```
src/htdse/
  core/            # System protocol, terms (composable Models), the four evolution
                   # classes, embed/partial_trace, compare_over, converged,
                   # truncation guard, plotting
  interop/         # optional bridges (qutip), imported lazily, never a dependency
  submodules/      # reusable physics: spin, harmonic_oscillator, trotter,
                   # molmer_sorensen (MS gate suite), wigner
  magnus.py        # magnus / magnus_pauli: what a pulse effectively generates
  util.py          # otimes, ket, fidelity, sampled_pulse, ...
demos/             # worked notebooks (start at 00)
tests/             # python tests/test_htdse.py ; python tests/test_molmer_sorensen.py
```
