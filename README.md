# htdse

`htdse` lets quantum-simulation code read like the physics while QuTiP handles
the numerical evolution underneath. Public states and operators are ordinary
NumPy arrays.

```python
import numpy as np
import htdse as ht

H = (ht.term(0.5 * ht.sigma_z, on="spin", name="splitting")
     + ht.term(0.2 * ht.sigma_x, on="spin", name="drive"))

psi = ht.evolve(H, ht.ket("0"), np.linspace(0, 10, 201))
population = ht.population("1", psi)
```

> **API design draft:** this README describes the intended final ergonomics.
> The proposed readout helpers called out below are not implemented yet.

The package supplies the readable layer QuTiP deliberately does not: named
subsystems, composable named Hamiltonian terms, approximation ladders, sequence
instructions, and focused diagnostics.

## Install

```text
pip install -e .
```

NumPy, SciPy, Matplotlib, and QuTiP 5.1 or newer are installed as dependencies.
QuTiP is the only numerical evolution backend; it is compiled and called
internally, so normal code does not handle `Qobj` or choose sparse storage.

## Four moves

### 1. Lay out the Hilbert space

Writing the subsystems first is the most grounded way to start a calculation:

```python
N = 8
subsystems = {"spin": 2, "motion": N}

psi0 = ht.otimes(ht.ket("0"), ht.fock(0, N))
```

The dictionary fixes both dimensions and tensor-product order. It is useful
throughout the calculation for embedding local operators, applying local
unitaries, taking partial traces, and interpreting matrix indices. This
explicit-first workflow is optional: a System can still infer the same registry
from its named terms.

### 2. Build and compose physical pieces

```python
a = ht.annihilation(N)
atom = ht.term(0.5 * 1.0 * ht.sigma_z, on="spin", name="atom")
mode = ht.term(1.0 * (a.conj().T @ a), on="mode", name="mode")
coupling = ht.plus_hc(
    ht.term({"spin": ht.sigma_plus, "mode": a}, coeff=0.1, name="coupling")
)

# Starting with an empty System pins the declared dimensions and ordering.
H = ht.System(subsystems) + atom + mode + coupling
realized = ht.replace(H, coupling=noisy_coupling)
```

Subsystem names perform the identity padding and tensor placement. Group names
make physics replaceable without rebuilding the rest of the Hamiltonian.
`System` values are immutable.

### 3. Evolve or run

```python
states = ht.evolve(H, psi0, times)

# Keep an evolution when you want to ask for states repeatedly.
ev = ht.HamiltonianEvolution(H, psi0)
psi_t = ev.state_at(t)
states = ev.state_at(times)
```

Use a density matrix to select Lindblad evolution automatically when the
System contains jump terms. `evolve(...)` is the short path and returns the
answer directly. The longer-lived `HamiltonianEvolution`, `UnitaryEvolution`,
`DensityMatrixEvolution`, and `LindbladEvolution` values retain `state_at(t)`
and are useful for repeated queries or `report()`.

### 4. Read the answer as physics

The public API should preserve the expressions a physicist would write on
paper. This table is the target ergonomics for the final API. `element`,
`population`, `overlap`, `distance`, `change_basis`, generalized `fidelity`,
the general measurement forms, the ket-accepting form of `partial_trace`, the
`on=` form of `embed`, and the operator-first forms of `apply_unitary` and
`measure` are proposed here and are not implemented yet.

| Operation | Pure state $\lvert\psi\rangle$ | Density matrix $\rho$ |
| :--- | :--- | :--- |
| **Represent a basis state** | $\lvert k\rangle$<br>`psi = ht.ket("1")` | $\lvert k\rangle\langle k\rvert$<br>`rho = ht.projector(ht.ket("1"))` |
| **Represent a general state** | $\lvert\psi\rangle = \sum_i c_i \lvert i\rangle$<br>`psi = c0*ht.ket("0") + c1*ht.ket("1")` | $\rho = \sum_i p_i \lvert\psi_i\rangle\langle\psi_i\rvert$<br>`rho = p0*ht.projector(psi0) + p1*ht.projector(psi1)` |
| **Coefficient or matrix element** | $c_0 = \langle0\vert\psi\rangle$<br>`c0 = psi[0]`<br>or `c0 = ht.element(psi, "0")` | $\rho_{01} = \langle0\vert\rho\vert1\rangle$<br>`rho01 = rho[0, 1]`<br>or `rho01 = ht.element(rho, "0", "1")`<br>For a pure state, $\rho_{01}=c_0c_1^*$. |
| **Element of one subsystem** | A subsystem of an entangled ket generally has no ket of its own.<br>$\langle0\vert\rho_s\vert1\rangle$, where $\rho_s=\operatorname{Tr}_m(\lvert\psi\rangle\langle\psi\rvert)$<br>`coherence = ht.element(psi, "0", "1", on="spin", subsystems=subsystems)` | $\langle0\vert\rho_s\vert1\rangle$, where $\rho_s=\operatorname{Tr}_m(\rho)$<br>`coherence = ht.element(rho, "0", "1", on="spin", subsystems=subsystems)`<br>`p0 = ht.element(rho, "0", on="spin", subsystems=subsystems)` |
| **Population in a basis state** | $\lvert c_k\rvert^2$<br>`p_k = abs(psi[k])**2`<br>or `p_k = ht.population("1", psi)` | $\rho_{kk}$<br>`p_k = rho[k, k].real`<br>or `p_k = ht.population("1", rho)` |
| **Expectation value** | $\langle A\rangle = \langle\psi\vert A\vert\psi\rangle$<br>`mean_A = ht.expect(A, psi)` | $\langle A\rangle = \operatorname{Tr}(A\rho)$<br>`mean_A = ht.Tr(A @ rho)`<br>or `mean_A = ht.expect(A, rho)` |
| **Apply a full-space unitary** | $\lvert\psi'\rangle = U\lvert\psi\rangle$<br>`psi_next = U @ psi`<br>or `psi_next = ht.apply_unitary(U, psi)` | $\rho' = U\rho U^\dagger$<br>`rho_next = U @ rho @ ht.dag(U)`<br>or `rho_next = ht.apply_unitary(U, rho)` |
| **Apply a local unitary** | $U_s\lvert\psi\rangle$<br>`psi_next = ht.apply_unitary(U, psi, on="spin", subsystems=subsystems)` | $U_s\rho U_s^\dagger$<br>`rho_next = ht.apply_unitary(U, rho, on="spin", subsystems=subsystems)` |
| **Embed a local operator** | $A_s = A\otimes I_m$<br>`A_spin = ht.embed(A, on="spin", subsystems=subsystems)`<br>`psi_next = A_spin @ psi` | $A_s = A\otimes I_m$<br>`A_spin = ht.embed(A, on="spin", subsystems=subsystems)`<br>`mean_A = ht.Tr(A_spin @ rho)` |
| **Change basis** | If the columns of $V$ are the new basis, $\lvert\psi\rangle_{new}=V^\dagger\lvert\psi\rangle$<br>`psi_new = ht.dag(V) @ psi`<br>or `psi_new = ht.change_basis(V, psi)` | $\rho_{new}=V^\dagger\rho V$<br>`rho_new = ht.dag(V) @ rho @ V`<br>or `rho_new = ht.change_basis(V, rho)` |
| **Combine subsystems** | $\lvert\psi_A\rangle\otimes\lvert\psi_B\rangle$<br>`psi = ht.otimes(psi_A, psi_B)` | $\rho_A\otimes\rho_B$<br>`rho = ht.otimes(rho_A, rho_B)` |
| **Discard a subsystem** | $\rho_A = \operatorname{Tr}_B(\lvert\psi\rangle\langle\psi\rvert)$<br>`rho_A = ht.partial_trace(psi, subsystems, "B")` | $\rho_A = \operatorname{Tr}_B(\rho)$<br>`rho_A = ht.partial_trace(rho, subsystems, "B")` |
| **Probability of an arbitrary state** | $\lvert\langle\phi\vert\psi\rangle\rvert^2$<br>`p_phi = ht.population(phi, psi)` | $\langle\phi\vert\rho\vert\phi\rangle$<br>`p_phi = ht.population(phi, rho)` |
| **Projective measurement outcome** | $p=\langle\psi\vert P\vert\psi\rangle$, $\lvert\psi'\rangle=P\lvert\psi\rangle/\sqrt p$<br>`post, p = ht.measure(P, psi)` | $p=\operatorname{Tr}(P\rho)$, $\rho'=P\rho P/p$<br>`post, p = ht.measure(P, rho)`<br>Add `on=` and `subsystems=` for a local measurement. |
| **POVM probabilities** | $p_k=\langle\psi\vert E_k\vert\psi\rangle$, $\sum_kE_k=I$<br>`probabilities = [ht.expect(E, psi) for E in effects]` | $p_k=\operatorname{Tr}(E_k\rho)$<br>`probabilities = [ht.Tr(E @ rho) for E in effects]`<br>The effects determine probabilities, not post-measurement states. |
| **General measurement outcome** | $p_k=\lVert M_k\lvert\psi\rangle\rVert^2$, $\lvert\psi_k\rangle=M_k\lvert\psi\rangle/\sqrt{p_k}$<br>`post, p = ht.measure(M_k, psi)` | $p_k=\operatorname{Tr}(M_k^\dagger M_k\rho)$, $\rho_k=M_k\rho M_k^\dagger/p_k$<br>`post, p = ht.measure(M_k, rho)` |
| **Overlap** | $\langle\phi\vert\psi\rangle$<br>`z = ht.overlap(phi, psi)` | $\operatorname{Tr}(\sigma\rho)$ (Hilbert--Schmidt overlap)<br>`z = ht.Tr(sigma @ rho)`<br>This is not the general mixed-state fidelity. |
| **Fidelity** | $F=\lvert\langle\phi\vert\psi\rangle\rvert^2$<br>`F = ht.fidelity(phi, psi)` | $F=\left(\operatorname{Tr}\sqrt{\sqrt{\sigma}\rho\sqrt{\sigma}}\right)^2$<br>`F = ht.fidelity(sigma, rho)`<br>A ket may be used for either argument. |
| **Distance** | $\sqrt{1-\lvert\langle\phi\vert\psi\rangle\rvert^2}$<br>`d = ht.distance(phi, psi)` | $\frac12\lVert\sigma-\rho\rVert_1$<br>`d = ht.distance(sigma, rho)` |
| **Relative phase between states** | $\arg(\langle\phi\vert\psi\rangle)$<br>`phase = ht.relative_phase(phi, psi)` | There is no global phase to compare: $e^{i\theta}\lvert\psi\rangle$ gives the same $\rho$. |
| **Relative phase of components** | $\arg(c_jc_k^*)$<br>`phase_jk = np.angle(c_j * c_k.conj())` | $\arg(\rho_{jk})$<br>`phase_jk = np.angle(rho[j, k])` |

Literal linear algebra is preferred when it is already readable. A helper earns
its place when it removes basis-index conversion, automatically embeds a local
operator, or presents one physical operation consistently for kets and density
matrices. In particular, `element(rho, "01", "10")` is the labelled-basis form
of `rho[1, 2]`; it is not a wrapper object or a new representation.

The proposed `element(state, row, col=None, *, on=None, subsystems=None)` accepts
a flat integer index or a qubit bitstring as a basis selector. One selector
means a ket coefficient, but a density-matrix diagonal element. Two selectors
mean $\langle row\vert\rho\vert col\rangle$. With `on=`, the rest of the system
is traced out first, so the result is always an element of the reduced density
matrix; one selector is therefore a local population. An arbitrary-basis
element remains ordinary algebra: `ht.dag(phi) @ rho @ chi`.

Solver trajectories are arrays with one state per requested time. Accordingly,
`ht.element(states, "0", "1")` returns the entire $\rho_{01}(t)$ trace, one
complex number per time, rather than requiring a Python loop.

Python binary literals are already integers, so `element(rho, 0b01, 0b10)` is
just the readable binary spelling of `element(rho, 1, 2)`. They are not accepted
by `ket`: the integer `0b10` has lost its width and could mean $\lvert10\rangle$,
$\lvert010\rangle$, or Fock state 2. `ket("10")` preserves the intended qubit
register and is the preferred spelling.

`apply_unitary` is conjugation for every square operator, not only density
matrices: `ht.apply_unitary(U, A)` means $UAU^\dagger$. The helper is most useful
with `on=`; for full-space arrays, `U @ psi` and `U @ A @ ht.dag(U)` are usually
clearer.

Every public name in this table belongs at the package top level. Both
`import htdse as ht` and `from htdse import *` expose the same physics vocabulary;
adding a public helper also requires adding it to `htdse.__all__`.

## Ion sequences

```python
mode = ht.Mode(nu=20.0, eta=[0.2, 0.2], n_max=8, name="com")
chain = ht.ion_chain([mode], 2)

seq = ht.sequence(
    ht.rx("q0", np.pi / 2, start=0.0, duration=1.0),
    ht.rz("q1", np.pi / 4, at=1.0),       # virtual frame update
    ht.rxx(("q0", "q1"), np.pi / 2,
           start=1.0, duration=2 * np.pi, mode="com"),
)

psi0 = ht.otimes(ht.ket("00"), ht.fock(0, 8))
states = ht.run(chain, seq, psi0, ht.sequence_times(seq), verbose=False)
```

`tone`, `wait`, and `sequence` are immutable data. Physical `rx`, `ry`, and
`rxx` are conveniences that compile to ordinary tones; `rz` is a virtual frame
change. `ideal_rx`, `ideal_ry`, `ideal_rxx`, and `ms_unitary` describe explicit
outcome-level comparisons.

Detuning follows one convention throughout:
`detuning = omega_laser - omega_0`, so red detuning is negative.

## Numerical guardrails

- Hamiltonians are checked for Hermiticity.
- Invalid density matrices and closed solvers given jump operators are refused.
- Declared breakpoints are included in every solve.
- Population at a truncated Fock-space ceiling raises `TruncationWarning`.
- Evolution objects reject mutation of private dynamics providers after binding.
- `report()` identifies QuTiP, tolerances, solved range, truncation, trace, and
  unitarity diagnostics.

Storage is internal. A large, structurally sparse System gives one suggestion;
`.sparse()` is the single optional override, and results remain NumPy arrays.

For advanced QuTiP-only operations such as `mcsolve` and `steadystate`,
`htdse.interop.qutip.to_qutip` exposes the already compiled System. This is an
escape hatch, not a second backend.

See [GUIDE.md](GUIDE.md) for the API walkthrough and [PHYSICS.md](PHYSICS.md)
for the approximations and equations implemented by the physics modules.
