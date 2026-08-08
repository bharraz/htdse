# Migration

Breaking changes, newest first. There are no back-compat aliases — the old names are gone,
so a stale reference fails loudly at import time rather than quietly doing the wrong thing.

Fastest way to find what needs changing:

```
grep -rn "Mechanism\|hconj\|Operator(\|truncated=" your_notebooks/ your_code/
```

---

## `Mechanism` → `System` (and it's now a Protocol)

`Mechanism` never sat *above* `Model` in a hierarchy — a `Model` doesn't become one, it
**is** one, and so is `MSMagnus`, which contains no `Model` at all. It was the contract
beside the data stack, not a floor in it. Renamed to `System` because the thing you hand an
evolution is a system, and *you evolve a System* is the sentence the package should teach.

It is also a `typing.Protocol` now rather than a concrete base class. This changes no
behavior: the evolutions already duck-typed every attribute they read (`getattr` for
`piecewise_constant`, `breakpoints`, `jump_operators`, `subsystems`; no `isinstance` check
anywhere), so inheriting was always optional. Subclassing still works and still buys the
`H()` alias, a readable `__repr__`, and clear errors instead of `AttributeError`.

| Old | New |
|---|---|
| `ht.Mechanism` | `ht.System` |
| `from htdse.core.mechanism import Mechanism` | `from htdse.core.system import System` |
| `class MyThing(ht.Mechanism):` | `class MyThing(ht.System):` |
| `TrotterizedMechanism` | `TrotterizedSystem` |
| `evolution.mechanism` | `evolution.system` |

```python
# before
from htdse.core.mechanism import Mechanism
class RabiDrive(Mechanism):
    def hamiltonian(self, t): return 0.5 * self.Omega * sigma_x

# after
class RabiDrive(ht.System):
    def hamiltonian(self, t): return 0.5 * self.Omega * sigma_x
```

One thing that did *not* change name: `interop.qutip.as_mechanism`. It's a function, and
renaming it is a separate decision.

**You may not need the class at all.** If your physics is a sum of named pieces, the
idiomatic route is a plain function returning a `Model` — you keep `+`, `.replace()`,
automatic identity padding, and the materialization cache, and your parameters live in
function arguments instead of object state:

```python
def my_drive(Omega, eps, delta) -> ht.Model:
    return (ht.term(0.5 * Omega * (1 + eps) * sigma_x, on="q", name="drive")
            + ht.term(delta * sigma_z, on="q", name="detuning"))
```

Write the class when your physics *isn't* a sum of terms: a closed-form `unitary(t)`, a
wrapper, or a bridge. See [README](README.md#extending-it).

## `Term` → `_Term` (private)

`Term` is now internal and should never appear in your code. It only ever existed because a
coefficient that is `f(t)` can't be folded into a matrix until `t` is known, so summands
have to survive until solve time — an implementation detail, not a layer.

Nothing user-facing took or returned a `Term`. If you were reaching into
`model.groups["name"][0]` to inspect one, that still works, but it's spelunking, not API.

Unchanged: `term()` (lowercase, the function) still returns a `Model`. A one-term
Hamiltonian and a fifty-term one are deliberately the same type.

## `hconj` → `plus_hc`

The old name read as "Hermitian conjugate", which is `.dag()` — a different operation.
`plus_hc` names the `X + h.c.` idiom it actually implements.

```python
jc = plus_hc(term({"spin": sigma_plus, "mode": a}, coeff=g))   # g s+ a + h.c.
```

## The `Operator` ndarray subclass is gone

Operators are plain numpy arrays (or scipy CSR when a `Model` is `.sparse()`-flagged).
Drop any `Operator(...)` wrapping — pass the array directly. Its `.params` metadata dict
had no readers.

Wrapping results in `np.asarray()` is also usually unnecessary now. Keep it only around
`.hamiltonian(t)` / `.unitary(t)` / `.state_at(t)` / `.jump_operators(t)`, which can return
sparse matrices.

## `truncated=` → `ladders=`

The truncation-guard keyword that selects *which* subsystems to check was renamed to avoid
colliding with `truncation=`, which sets the *threshold*.

```python
HamiltonianEvolution(H, psi0, truncation=1e-6, ladders=["mode"])
```

---

## New since, worth adopting

- `dag(op)` — Hermitian conjugate, replaces `.conj().T`.
- `ladder_operators(n_max)` — returns `(a, a_dagger, n)` in one call.
- `report()` — on any evolution: solved range, propagation method, rhs evals, truncation
  populations, unitarity defect.
- `converged(fn, values, tol=)` — sweep a parameter until the answer stops moving.
- `htdse.interop.qutip` — `to_qobj` / `from_qobj` / `to_qutip` / `as_mechanism`.
