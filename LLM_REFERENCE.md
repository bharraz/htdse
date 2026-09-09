# htdse machine reference

- Purpose: functional, physics-readable quantum dynamics with QuTiP as the
  numerical backend and NumPy arrays as the public state/operator type.
- Import convention: `import htdse as ht`.
- Units: $\hbar=1$ except `lamb_dicke`, whose accepted inputs are documented in
  SI units.
- Tensor order: insertion order in every `{subsystem: dimension}` mapping.
- Qubit convention: `ket("0")` is the `sigma_z=+1` state;
  `sigma_plus @ ket("1") == ket("0")`.
- Values returned by `System` transformations are new values; `System` and
  `Unitary` are immutable.

## Systems and dynamics

- `System(subsystems=None, groups=None, jumps=None, sparse=False, breakpoints=())`
  — empty/composable dynamics value.
- `term(op, on=None, coeff=1, name=None, frame=None, dims=None) -> System`
  — one Hamiltonian contribution; `op` may be a local matrix, a
  `{subsystem: local_matrix}` product, or a joint matrix with tuple `on` and
  explicit `dims`.
- `jump(op, on=None, coeff=1, name=None, dims=None) -> System` — one Lindblad
  operator; convention is `coeff=sqrt(rate)`.
- `System + System`, `sum(systems)` — compose compatible registries and groups.
- `scalar * System`, `System * scalar_or_f(t)`, unary `-`, and subtraction —
  scale/combine coherent systems; rejected when jumps are present.
- `System.dag()` / `hc(System)` — conjugate Hamiltonian terms and drop jumps.
- `plus_hc(System)` — add the Hermitian conjugate while retaining jumps once.
- `replace(system, **{group_name: replacement})` — replace an existing named
  contribution.
- `without(system, *names)` — remove named Hamiltonian/jump groups.
- `group(system, name)` — extract one named group as a `System`.
- `System.H(t)` / `System.hamiltonian(t)` — dense NumPy $H(t)$.
- `System.jump_operators(t)` — dense NumPy Lindblad matrices.
- `System.breakpoints()` — declared discontinuity times.
- `System.dim`, `System.subsystems`, `System.groups`, `System.jumps` — structure.
- `System.sparse(flag=True)` — select internal sparse QuTiP compilation; public
  results stay NumPy arrays.
- `SparseSuggestion` — performance warning for large sparse-looking systems.
- `Unitary(fn, dim, subsystems=None, name="Unitary", **helpers)` — immutable
  analytic $U(t)$; callable and also exposes `.unitary(t)`.
- Advanced dynamics providers may supply `.hamiltonian(t)` or `.unitary(t)`,
  plus optional `.jump_operators(t)`, `.subsystems`, and `.breakpoints()`.

## Evolution

- `evolve(system, initial, t, t0=0, **solver_options)` — ket selects
  Schrödinger evolution; density matrix selects closed or Lindblad evolution.
- `propagator(system, t, dim=None, t0=0, **solver_options)` — generated or
  analytic unitary.
- `HamiltonianEvolution(system, ket, t0=0, ...)` — `.state_at(t)`,
  `.trace_out(*names, t=...)`, instantaneous-eigenbasis and adiabatic helpers.
- `UnitaryEvolution(system, initial=None, dim=None, t0=0, ...)` —
  `.unitary_at(t)` / `.state_at(t)` and `.unitarity_defect(t)`.
- `DensityMatrixEvolution(system, rho0, t0=0, ...)` — closed density-matrix
  evolution, `.state_at(t)`, `.trace_out(...)`, `.unitarity_defect(t)`.
- `LindbladEvolution(system, rho0, t0=0, ...)` — open evolution,
  `.state_at(t)` and `.trace_out(...)`; backward time is rejected.
- Evolution options include `rtol`, `atol`, `method`, `verbose`, `subsystems`,
  `truncation`, `ladders`, and `check_mutation` where applicable.
- Evolution values expose `.report(t=None)`.

## States, operators, and subsystems

- `ket(bitstring)`, `bra(bitstring)` — computational-basis vectors.
- `projector(state)` — $|\psi\rangle\langle\psi|$.
- `dag(op)` — Hermitian conjugate.
- `Tr(op)` — trace.
- `otimes(*arrays)` — tensor product.
- `embed(op, subsystems, on)` — place a local/joint operator in the registry.
- `partial_trace(state_or_rho, subsystems, trace_out)` — reduced density matrix;
  accepts single states and trajectories.
- `apply_unitary(U, state_or_operator, subsystems=None, on=None)` — $U\psi$ or
  $UAU^\dagger$, optionally on named subsystems.
- `change_basis(V, state_or_operator)` — $V^\dagger\psi$ or $V^\dagger A V$;
  columns of `V` are the new basis.
- `measure(M, state_or_rho, subsystems=None, on=None)` — conditioned outcome;
  returns `(post_state, probability)`.
- `binary_to_index(bitstring)`, `index_to_binary(index, bits=None)` — basis-index
  conversion.
- `sampled_pulse(times, values, kind="linear")` — interpolated coefficient.
- `MAG_THRESHOLD` — shared numerical magnitude threshold.

## Reading and comparing

- `expect(operator, state)` — $\langle\psi|A|\psi\rangle$ or
  $\operatorname{Tr}(A\rho)$; trajectories supported.
- `element(state, row, col=None, on=None, subsystems=None)` — coefficient,
  population, or density-matrix element; local reduction and trajectories
  supported.
- `population(target, state, on=None, subsystems=None)` — basis-state or
  arbitrary-ket population.
- `overlap(a, b)` — ket inner product or matrix Hilbert–Schmidt overlap.
- `fidelity(a, b)` — pure or mixed-state squared fidelity.
- `distance(a, b)` — pure-state distance or density-matrix trace distance.
- `relative_phase(a, b)` — phase of ket overlap.
- `process_fidelity(U, V)` — normalized unitary process fidelity.
- `compare_over(ts, target, realized, metric, target_adapter=None,
  realized_adapter=None)` — compare trajectories.
- `show(H, t=0, tol=1e-10)` — named Pauli decomposition for registered
  two-level subsystems; otherwise a small matrix or labeled dominant basis
  transitions for larger operators.
- `project_block(U, subsystems, on, state=0)` — fixed-subsystem block.
- `closure(M)` — smallest singular value; one means a closed projected block.
- `generator(M, T)` — effective traceless Hermitian generator from matrix log.
- `max_eigenphase(H_eff, T)` — matrix-log branch-cut diagnostic.
- `paulis(M)` / `pauli_decompose(M, tol=...)` — Pauli coefficients.
- `converged(fn, values, tol=1e-6, metric=None, parameter="value", quiet=True)`
  — convergence scan and result record.

## Plotting and analysis

- `plot_populations`, `plot_eigenspectrum`, `plot_matrix`, `plot_phases`,
  `plot_adiabatic_populations` — core plotting helpers.
- `magnus(H, T, order=2, t0=0, n_grid=2001)` — Magnus terms.
- `magnus_pauli(...)` — Magnus terms decomposed in the Pauli basis.

## Numerical controls

- `quiet()` — suppress solver messages in a context.
- `no_truncation_check()` — disable truncation checks in a context.
- `TruncationWarning` — population reached a registered ladder ceiling.
- `truncation_populations(state, subsystems, kind="ket", names=None)` — ceiling
  populations by subsystem.
- Registered dimensions of three or more are treated as possible truncated
  ladders unless `ladders=` says otherwise.

## Spin and harmonic oscillator

- `sigma_x`, `sigma_y`, `sigma_z`, `I2`, `hadamard`, `sigma_plus`,
  `sigma_minus`, `PAULIS`.
- `pauli_term(spec, coeff=1, name=None, n_qubits=None, frame=None, prefix="q")` —
  accepts compact numbered strings (`"X0X1"`) or whitespace-delimited subsystem
  names (`"Xr1 Xq2"`).
- `pauli_sum(spec, n_qubits=None, frame=None, prefix="q")` — sums the same
  numbered or named Pauli terms.
- `spin_operators(J)`, `raising_lowering(J)`; `ht.spin_j.identity(J)` remains
  namespaced.
- `annihilation(n_max)`, `creation(n_max)`, `number_operator(n_max)`,
  `ladder_operators(n_max)`.
- `fock(n, n_max)`, `thermal(nbar, n_max)`.
- `ThermalMotionalDecoherence(n_max, gamma_a=0, nbar=0, gamma_p=0)`.
- `ht.wigner.wigner(state, xs, ps)`, `ht.wigner.plot_wigner(...)`.

## Spin–boson and trapped-ion physics

- `Tone(offset, amp=1, phase=0)` — immutable signed-frequency drive tone.
- `Mode(nu, eta, n_max, name="mode")`; `eta` may be per-spin.
- `Mode.from_participation(nu, eta, b, n_max, name="mode")`.
- `driven_spins(tones, spins, modes, lamb_dicke=1, rwa=False, prefix=None,
  sparse=False)` — exact/Lamb–Dicke/RWA drive ladder.
- `exact_drive(tones, spins, modes, sparse=False)` — unexpanded operator-valued
  interaction.
- `jaynes_cummings(...)`, `rabi(...)` — standard spin–boson systems.
- `explain()` / `TONE_TABLE` — approximation/tone reference.
- `ion_chain(modes, n_ions, prefix="q")` — immutable chain with `.subsystems`,
  `.drive(...)`, `.thermal(...)`, and sideband helpers.
- `tone`, `wait`, `sequence` — immutable sequence instructions.
- `rx`, `ry`, `rz`, `rxx` — physical sequence operations.
- `ideal_rx`, `ideal_ry`, `ideal_rxx`, `ms_unitary` — ideal unitary events.
- `compile_tones(chain, seq)`, `run(chain, seq, initial, times, **kwargs)`.
- `breakpoints(value)`, `sequence_times(seq, points_per_segment=200)`,
  `mode_expectation(states, chain, mode)`.
- Sequence detuning convention: `omega_laser - omega_0`; red is negative.
- `ms_tones(nu, delta, amp, theta=0, psi=0, delta_red=None, amp_red=None)`.
- `ms_closed_form(spins, modes, delta, amplitudes=1, phases=0,
  points_per_period=400, motion_phases=0, sparse=False)` — analytic MS
  `Unitary` with displacement/phase helpers.
- `ideal_gate(...)`, `expectation_alpha(...)`, `plot_phase_space(...)`.
- `lamb_dicke`, `sideband`, `sideband_series`, `thermal_population_series`,
  `thermal_sideband` — trap and sideband formulas.

## Atomic, NV, and Rydberg physics

- `clebsch_gordan(...)`, `wigner_6j(...)` — angular-momentum coefficients.
- `g_s`, `g_l`, `g_sum`, `hyperfine`, `spin_manifold`,
  `couple_reduced_element`, `dipole_couple_matrix`, `hyperfine_matrix`,
  `hyperfine_levels` — atomic structure in coupled bases.
- `zero_field_splitting`, `zeeman`, `hyperfine_tensor`, `isc_dephasing` — NV
  ground-state terms.
- `NVCenter(D, E=0, nuclear_spins=None, prefix_e="e")` — NV system builder.
- `rydberg_interaction`, `dipole_dipole_interaction`, `blockade_radius` —
  tweezer-array interactions.

## Other modules and interop

- `TrotterizedSystem(inner, t_start, t_stop, n_steps, sample="midpoint")` —
  piecewise-constant wrapper.
- `htdse.interop.qutip.to_qobj(array, subsystems=None)` — NumPy to QuTiP.
- `htdse.interop.qutip.to_qutip(system, include_jumps=True)` — compiled QuTiP
  Hamiltonian and collapse operators.
- Module namespaces: `spin`, `spin_j`, `harmonic_oscillator`, `spin_boson`,
  `trapped_ion`, `molmer_sorensen`, `trap`, `atomic`, `angular_momentum`,
  `nv_center`, `rydberg`, `trotter`, and `wigner`.
