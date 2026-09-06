"""Trapped-ion specialization of `spin_boson.py`: a chain of `n_ions` spins
coupled to a table of shared motional `Mode`s.

`IonChain` owns exactly what the general layer doesn't know about an ion:
how many spins there are, which subsystem names they get, and the per-mode
Fock truncation -- so `subsystems` has one source instead of being inferred
from whatever arrays happen to be the right length (see the deleted
`_infer_n_ions` in the old `molmer_sorensen.py`).

Computing a real chain's normal modes (nu_m, b_{i,m}) from trap parameters --
equilibrium positions from the Coulomb + harmonic potential, then Hessian
eigenvectors (James 1998) -- is NOT implemented here. `IonChain.from_trap(...)`
is the documented future home for it.
"""
import numpy as np
from dataclasses import dataclass
from types import MappingProxyType

from .harmonic_oscillator import thermal, number_operator, annihilation
from .spin_boson import Mode, driven_spins, _norm_mode, _as_funcs, _eval
from . import trap as _trap
from scipy.linalg import expm
from ..core.terms import System
from ..core.evolution import HamiltonianEvolution, DensityMatrixEvolution, LindbladEvolution
from ..core.subsystems import embed
from .spin import sigma_x, sigma_y, sigma_z


class IonChain:
    """`n_ions` two-level ions sharing the motional `modes`.

        chain = IonChain(modes=[Mode(nu=nu, eta=eta*b, n_max=8)], n_ions=2)
        H = chain.drive(tones, lamb_dicke=1)

    Spins are named "q0", "q1", ... -- change `prefix=` to rename them.
    """

    def __init__(self, modes, n_ions, prefix="q"):
        n_ions = int(n_ions)
        normalized_modes = []
        for mode in modes:
            normalized = _norm_mode(mode, n_ions)
            eta = np.array(normalized.eta, dtype=float, copy=True)
            eta.setflags(write=False)
            normalized_modes.append(Mode(normalized.nu, eta, normalized.n_max, normalized.name))
        modes = tuple(normalized_modes)
        spins = tuple(f"{prefix}{i}" for i in range(n_ions))
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "n_ions", n_ions)
        object.__setattr__(self, "prefix", str(prefix))
        object.__setattr__(self, "spins", spins)
        object.__setattr__(self, "subsystems", MappingProxyType(
            {**{q: 2 for q in spins}, **{md.name: md.n_max + 1 for md in modes}}))

    def __setattr__(self, name, value):
        raise AttributeError("IonChain values are immutable")

    @property
    def dim(self) -> int:
        d = 1
        for v in self.subsystems.values():
            d *= v
        return d

    def drive(self, tones, lamb_dicke=1, rwa=False):
        """The Hamiltonian of this chain driven by `tones` -- a thin
        pass-through to `spin_boson.driven_spins` so ion-chain code never
        assembles that call by hand."""
        return driven_spins(tones, self.spins, self.modes, lamb_dicke=lamb_dicke, rwa=rwa)

    def thermal(self, nbar):
        """Motional density matrix: each mode independently thermal at `nbar`
        (scalar, or one value per mode), tensored together in mode order.
        `nbar` is an INITIAL-STATE choice, orthogonal to which rung of the
        approximation ladder `drive()` uses.

        Cost note: this returns a density matrix (dim^2 for a mixed motional
        state), so pair it with `DensityMatrixEvolution`/`LindbladEvolution`.
        For large chains, Monte-Carlo averaging over Fock states sampled from
        the thermal distribution is the cheaper alternative and stays in
        state-vector land -- not implemented here."""
        nbars = np.broadcast_to(np.asarray(nbar, dtype=float), (len(self.modes),))
        rho = thermal(float(nbars[0]), self.modes[0].n_max)
        for nb, md in zip(nbars[1:], self.modes[1:]):
            rho = np.kron(rho, thermal(float(nb), md.n_max))
        return rho

    def _mode(self, mode_name):
        for md in self.modes:
            if md.name == mode_name:
                return md
        raise KeyError(f"IonChain has no mode named {mode_name!r}; "
                       f"known modes: {[md.name for md in self.modes]}")

    def sideband_coupling(self, mode_name, n1, n2, phase=True):
        """<n1|exp(i eta (a+a^dagger))|n2> for the named mode's own eta --
        a thin pass-through to `trap.sideband` so callers don't have to pull
        `mode.eta` out by hand."""
        return _trap.sideband(n1, n2, self._mode(mode_name).eta, phase=phase)

    def thermal_sideband(self, mode_name, delta_n, t, nbar, thresh=1e-3):
        """Thermally-averaged sideband Rabi flopping probability on the
        named mode's delta_n-th sideband -- a thin pass-through to
        `trap.thermal_sideband` using the mode's own eta."""
        return _trap.thermal_sideband(nbar, delta_n, self._mode(mode_name).eta, t, thresh=thresh)

    def __repr__(self):
        modes = ", ".join(f"{md.name}(n_max={md.n_max})" for md in self.modes)
        return f"IonChain(n_ions={self.n_ions}, modes=[{modes}])"


@dataclass(frozen=True)
class ScheduledTone:
    """An immutable windowed tone instruction for the Milestone 4 compiler."""
    ions: tuple
    detuning: object
    amplitude: object = 1.0
    phase: object = 0.0
    start: float = 0.0
    duration: float = 0.0
    orders: tuple | None = (0, 1)


@dataclass(frozen=True)
class Wait:
    start: float
    duration: float


@dataclass(frozen=True)
class Sequence:
    instructions: tuple


@dataclass(frozen=True)
class _Gate:
    kind: str
    ions: tuple
    angle: float
    start: float
    duration: float = 0.0
    detuning: object = None


@dataclass(frozen=True)
class _UnitaryEvent:
    kind: str
    ions: tuple
    angle: float
    at: float
    alpha: tuple = ()
    mode: object = None


def tone(ions, detuning, amplitude=1.0, phase=0.0, start=0.0,
         duration=None, orders=(0, 1)) -> ScheduledTone:
    """Create one immutable, windowed tone instruction.

    ``detuning`` is omega_laser - omega_0. ``orders`` selects isolated
    Lamb-Dicke powers; ``None`` requests the unexpanded interaction. The
    instruction is data only; ``compile_tones`` turns it into a System.
    """
    ions = (ions,) if isinstance(ions, str) else tuple(ions)
    start = float(start)
    if duration is None:
        raise ValueError("tone needs an explicit positive duration")
    duration = float(duration)
    if start < 0 or duration <= 0:
        raise ValueError("tone requires start >= 0 and duration > 0")
    if orders is not None:
        orders = tuple(orders)
        if not orders or any(order not in (0, 1, 2) for order in orders):
            raise ValueError("tone orders must be a non-empty subset of (0, 1, 2), or None")
        if len(set(orders)) != len(orders):
            raise ValueError("tone orders must not repeat powers")
    def freeze(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return value
    return ScheduledTone(ions, detuning, freeze(amplitude), freeze(phase),
                         start, duration, orders)


def _gate(kind, ions, angle, start=0.0, duration=None, detuning=None):
    ions = (ions,) if isinstance(ions, str) else tuple(ions)
    if not ions:
        raise ValueError("a gate needs at least one ion")
    start = float(start)
    if start < 0:
        raise ValueError("gate start must be nonnegative")
    if duration is None:
        raise ValueError("physical gates need an explicit positive duration")
    duration = float(duration)
    if duration <= 0:
        raise ValueError("physical gate duration must be positive")
    if kind == "rxx" and len(ions) != 2:
        raise ValueError("rxx needs exactly two ions")
    if kind != "rxx" and len(ions) != 1:
        raise ValueError(f"{kind} needs exactly one ion")
    return _Gate(kind, ions, float(angle), start, duration, detuning)


def rx(ion, angle, start=0.0, duration=None):
    return _gate("rx", ion, angle, start, duration)


def ry(ion, angle, start=0.0, duration=None):
    return _gate("ry", ion, angle, start, duration)


def rxx(ions, angle, start=0.0, duration=None, detuning=None, mode=None):
    return _gate("rxx", ions, angle, start, duration,
                 (detuning, mode))


def rz(ion, angle, at=0.0):
    at = float(at)
    if at < 0:
        raise ValueError("rz timestamp must be nonnegative")
    return _UnitaryEvent("rz", (ion,), float(angle), at)


def _ideal(kind, ions, angle, at=0.0):
    ions = (ions,) if isinstance(ions, str) else tuple(ions)
    if kind in ("ideal_rxx", "ms_unitary") and len(ions) != 2:
        raise ValueError("ideal_rxx needs exactly two ions")
    if kind not in ("ideal_rxx", "ms_unitary") and len(ions) != 1:
        raise ValueError(f"{kind} needs exactly one ion")
    at = float(at)
    if at < 0:
        raise ValueError("unitary event timestamp must be nonnegative")
    return _UnitaryEvent(kind, ions, float(angle), at)


def ideal_rx(ion, angle, at=0.0):
    return _ideal("ideal_rx", ion, angle, at)


def ideal_ry(ion, angle, at=0.0):
    return _ideal("ideal_ry", ion, angle, at)


def ideal_rxx(ions, angle, at=0.0):
    return _ideal("ideal_rxx", ions, angle, at)


def ms_unitary(alpha, theta, ions, mode=None, at=0.0):
    ions = (ions,) if isinstance(ions, str) else tuple(ions)
    if len(ions) != 2:
        raise ValueError("ms_unitary needs exactly two ions")
    if np.isscalar(alpha) and mode is None:
        raise ValueError("ms_unitary needs mode= for a scalar displacement")
    if mode is not None and not np.isscalar(alpha):
        raise ValueError("ms_unitary accepts either mode= with scalar alpha, or a "
                         "mapping of mode names to displacements")
    event = _ideal("ms_unitary", ions, theta, at)
    records = []
    if mode is not None:
        values = [alpha] * len(ions)
        records.extend((ion, str(mode), complex(value))
                       for ion, value in zip(ions, values))
    else:
        for mode_name, values in dict(alpha).items():
            if np.isscalar(values):
                values = [values] * len(ions)
            if len(values) != len(ions):
                raise ValueError("each ms_unitary mode displacement needs one "
                                 "alpha per ion")
            records.extend((ion, str(mode_name), complex(value))
                           for ion, value in zip(ions, values))
    return _UnitaryEvent(event.kind, event.ions, event.angle, event.at,
                         tuple(records))


def wait(start, duration) -> Wait:
    """Create an immutable idle scheduling instruction."""
    start, duration = float(start), float(duration)
    if start < 0 or duration <= 0:
        raise ValueError("wait requires start >= 0 and duration > 0")
    return Wait(start, duration)


def sequence(*instructions) -> Sequence:
    """Create an immutable ordered collection of tone and wait instructions."""
    if not all(isinstance(item, (ScheduledTone, Wait, _Gate, _UnitaryEvent))
               for item in instructions):
        raise TypeError("sequence accepts tone, gate, wait, and unitary instructions")
    return Sequence(tuple(instructions))


def ion_chain(modes, n_ions, prefix="q") -> IonChain:
    """Functional constructor for an immutable ion-chain value."""
    return IonChain(modes, n_ions, prefix=prefix)


def _windowed_functions(value, n, start, end, what):
    from .spin_boson import _as_funcs
    funcs = _as_funcs(value, n, what)
    return [lambda t, fn=fn: fn(t) if start <= t < end else 0.0 for fn in funcs]


def _scheduled_tone(chain, item: ScheduledTone, phase_offsets=None):
    from .spin_boson import Tone, driven_spins
    unknown = [ion for ion in item.ions if ion not in chain.spins]
    if unknown:
        raise KeyError(f"tone names unknown ion(s) {unknown}; chain has {list(chain.spins)}")
    end = item.start + item.duration
    amp = _windowed_functions(item.amplitude, len(item.ions), item.start, end,
                              "tone amplitude")
    phase_funcs = _as_funcs(item.phase, len(item.ions), "tone phase")
    offsets = phase_offsets or {}
    def frame_offset(t, ion):
        return sum(angle for at, angle in offsets.get(ion, ()) if t >= at)
    # Compile the accumulated detuning phase once. Passing the detuning again
    # as Tone.offset would make spin_boson._phase_of perform fresh quadrature
    # during every Hamiltonian evaluation.
    if callable(item.detuning):
        grid = np.linspace(item.start, end, max(2001, int(2001 * (item.duration + 1.0))))
        values = np.asarray(_eval(item.detuning, grid), dtype=float)
        increments = 0.5 * (values[1:] + values[:-1]) * np.diff(grid)
        accumulated = np.concatenate(([0.0], np.cumsum(increments)))
        def detuning_phase(t):
            return np.interp(t, grid, accumulated)
        optical_offset = 0.0
    else:
        # Keep a constant detuning in Tone.offset, but compensate its global
        # time origin below: the public tone phase starts accumulating at the
        # scheduled window start, not at t=0.
        detuning_phase = lambda t: 0.0
        optical_offset = float(item.detuning)
    phase_origin = -optical_offset * item.start
    phase = [lambda t, fn=fn, ion=ion: fn(t) + frame_offset(t, ion)
             + detuning_phase(t) + phase_origin
             for fn, ion in zip(phase_funcs, item.ions)]
    optical = Tone(offset=optical_offset, amp=amp, phase=phase)
    selected_indices = [chain.spins.index(ion) for ion in item.ions]
    modes = tuple(Mode(md.nu, np.array(md.eta[selected_indices], copy=True),
                       md.n_max, md.name) for md in chain.modes)
    if item.orders is None:
        return driven_spins([optical], list(item.ions), modes, lamb_dicke=None)
    selected = None
    for order in item.orders:
        if order == 0:
            component = driven_spins([optical], list(item.ions), [], lamb_dicke=1)
        elif order == 1:
            full = driven_spins([optical], list(item.ions), modes, lamb_dicke=1)
            carrier = driven_spins([optical], list(item.ions), [], lamb_dicke=1)
            component = full - carrier
        else:
            second = driven_spins([optical], list(item.ions), modes, lamb_dicke=2)
            first = driven_spins([optical], list(item.ions), modes, lamb_dicke=1)
            component = second - first
        selected = component if selected is None else selected + component
    return selected


def _physical_gate(chain, item):
    if item.kind == "rx":
        return tone(item.ions, 0.0, amplitude=item.angle / item.duration,
                    phase=0.0, start=item.start, duration=item.duration, orders=(0,))
    if item.kind == "ry":
        return tone(item.ions, 0.0, amplitude=item.angle / item.duration,
                    phase=np.pi / 2, start=item.start, duration=item.duration, orders=(0,))
    # A physical XX pulse is a bichromatic first-order spin-motion drive.  The
    # mode frequency supplies the two signed sideband offsets; no single
    # zero-detuning second-order tone is substituted for this interaction.
    detuning, mode_name = item.detuning
    if mode_name is None:
        if len(chain.modes) != 1:
            raise ValueError("physical rxx needs an explicit mode when the chain "
                             "has multiple modes")
        mode = chain.modes[0]
    else:
        mode = chain._mode(mode_name)
    delta = (np.copysign(2.0 * np.pi, item.angle) / item.duration
             if detuning is None
             else float(detuning))
    if delta == 0:
        raise ValueError("physical rxx needs nonzero detuning")
    eta = np.asarray(mode.eta, dtype=float)
    indices = [chain.spins.index(ion) for ion in item.ions]
    eta_product = float(eta[indices[0]] * eta[indices[1]])
    if eta_product == 0:
        raise ValueError("physical rxx cannot calibrate with zero participation")
    # Calibrate the geometric phase for U=exp(-i*angle*X1 X2/2).  For a
    # closed loop (the default delta*T=+-2*pi), this reduces to
    # Omega^2 = angle*delta^2/(2*pi*eta_i*eta_j); the general expression
    # retains the finite-time sin(delta*T) term.
    geometric = item.duration - np.sin(delta * item.duration) / delta
    if geometric == 0:
        raise ValueError("physical rxx duration and detuning produce no "
                         "geometric phase")
    amp = np.sqrt(abs(item.angle * delta)
                  / (abs(eta_product * geometric)))
    spin_phases = [-np.pi / 2 + (np.pi if eta[i] < 0 else 0.0)
                   for i in indices]
    # With participation signs absorbed into the local spin axes, delta sets
    # the sign of the geometric interaction. Flip one axis when the caller
    # explicitly chooses the opposite detuning sign.
    if item.angle != 0 and np.sign(item.angle) != np.sign(delta):
        spin_phases[0] += np.pi
    spin_phases = tuple(spin_phases)
    nu = mode.nu
    return (tone(item.ions, -(nu + delta), amplitude=amp, phase=spin_phases,
                 start=item.start, duration=item.duration, orders=(1,)),
            tone(item.ions, +(nu + delta), amplitude=amp, phase=spin_phases,
                 start=item.start, duration=item.duration, orders=(1,)))


def _compiled_sequence(chain, seq):
    """Return the physical tone sequence with virtual-Z phases applied."""
    offsets = {ion: [] for ion in chain.spins}
    for item in seq.instructions:
        if isinstance(item, _UnitaryEvent) and item.kind == "rz":
            offsets.setdefault(item.ions[0], []).append((item.at, item.angle))
    out = []
    for item in seq.instructions:
        if isinstance(item, ScheduledTone):
            out.append((item, offsets.copy()))
        elif isinstance(item, _Gate):
            gates = _physical_gate(chain, item)
            if not isinstance(gates, tuple):
                gates = (gates,)
            out.extend((gate, offsets.copy()) for gate in gates)
    return out


def compile_tones(chain: IonChain, seq: Sequence) -> "System":
    """Compile scheduled tones into one continuous, breakpoint-aware System."""
    from ..core.terms import System
    if not isinstance(chain, IonChain):
        raise TypeError("compile_tones needs an IonChain")
    if not isinstance(seq, Sequence):
        raise TypeError("compile_tones needs a sequence(...) value")
    breakpoints = []
    compiled = System(chain.subsystems)
    for item in seq.instructions:
        if isinstance(item, _UnitaryEvent):
            breakpoints.append(item.at)
        elif isinstance(item, (ScheduledTone, Wait, _Gate)):
            breakpoints.extend((item.start, item.start + item.duration))
    for item, offsets in _compiled_sequence(chain, seq):
        compiled = compiled + _scheduled_tone(chain, item, offsets)
    return System(compiled.subsystems, compiled.groups, compiled.jumps,
                  sparse=compiled.is_sparse, breakpoints=breakpoints)


def breakpoints(value):
    if isinstance(value, Sequence):
        points = []
        for item in value.instructions:
            if isinstance(item, _UnitaryEvent):
                points.append(item.at)
            else:
                points.extend((item.start, item.start + item.duration))
        return np.unique(points).astype(float)
    if isinstance(value, System):
        return value.breakpoints()
    raise TypeError("breakpoints needs a sequence or System")


def sequence_times(seq, points_per_segment=200):
    points = breakpoints(seq)
    if len(points) < 2:
        return points
    return np.unique(np.concatenate(
        [np.linspace(a, b, int(points_per_segment), endpoint=False)
         for a, b in zip(points[:-1], points[1:])] + [np.asarray([points[-1]])]))


def _event_unitary(chain, event):
    if event.kind == "ideal_rx":
        local = expm(-0.5j * event.angle * sigma_x)
    elif event.kind == "ideal_ry":
        local = expm(-0.5j * event.angle * sigma_y)
    elif event.kind == "ideal_rxx":
        local = expm(-0.5j * event.angle * np.kron(sigma_x, sigma_x))
    elif event.kind == "rz":
        local = expm(-0.5j * event.angle * sigma_z)
    else:
        # Outcome-defined MS unitary: independent per-ion/per-mode residual
        # displacements followed by the geometric spin interaction.
        displacement = np.zeros((chain.dim, chain.dim), dtype=complex)
        for ion, mode_name, alpha in event.alpha:
            mode = chain._mode(mode_name)
            a = embed(annihilation(mode.n_max), chain.subsystems, mode.name)
            spin = embed(sigma_x, chain.subsystems, ion)
            displacement += alpha * a.conj().T @ spin - np.conj(alpha) * a @ spin
        pair = (embed(sigma_x, chain.subsystems, event.ions[0])
                @ embed(sigma_x, chain.subsystems, event.ions[1]))
        return expm(displacement + 1j * event.angle * pair)
    return embed(local, chain.subsystems, event.ions)


def run(chain: IonChain, seq: Sequence, initial, times, **kwargs):
    """Execute physical intervals and instantaneous unitary events."""
    if not isinstance(chain, IonChain) or not isinstance(seq, Sequence):
        raise TypeError("run needs ion_chain(...) and sequence(...) values")
    times_arr = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(times_arr < 0):
        raise ValueError("run times must be nonnegative")
    if len(times_arr) > 1 and np.any(np.diff(times_arr) < 0):
        raise ValueError("run times must be sorted in nondecreasing order")
    system = compile_tones(chain, seq)
    state = np.asarray(initial, dtype=complex).copy()
    evolution_class = (HamiltonianEvolution if state.ndim == 1
                       else (LindbladEvolution if system.jumps else DensityMatrixEvolution))
    events = sorted(((item.at, i, item) for i, item in enumerate(seq.instructions)
                     if isinstance(item, _UnitaryEvent)), key=lambda x: (x[0], x[1]))
    # Virtual-Z instructions are already compiled into the tone phases. With
    # no state-changing instantaneous event, hand the complete requested grid
    # to QuTiP in one call; solving one tiny interval per plotted point turns
    # Python/Qobj setup into the dominant cost for the normal ion workflow.
    if not any(event.kind != "rz" for _, _, event in events):
        result = evolution_class(system, state, t0=0.0, **kwargs).state_at(times_arr)
        return result[0] if np.ndim(times) == 0 else result
    current = 0.0
    out = []
    event_index = 0
    for target in times_arr:
        while event_index < len(events) and events[event_index][0] <= target:
            at, _, event = events[event_index]
            event_index += 1
            if at < current:
                continue
            if at > current:
                state = evolution_class(system, state, t0=current, **kwargs).state_at(at)
                current = at
            if event.kind != "rz":
                U = _event_unitary(chain, event)
                state = (U @ state if state.ndim == 1
                         else U @ state @ U.conj().T)
        if target > current:
            state = evolution_class(system, state, t0=current, **kwargs).state_at(target)
            current = target
        out.append(state.copy())
    result = np.asarray(out)
    return result[0] if np.ndim(times) == 0 else result


def mode_expectation(states, chain: IonChain, mode):
    md = chain._mode(mode)
    op = embed(number_operator(md.n_max), chain.subsystems, mode)
    arr = np.asarray(states)
    if arr.ndim == 1:
        return np.vdot(arr, op @ arr)
    if arr.ndim == 2 and arr.shape == (chain.dim, chain.dim):
        return np.trace(op @ arr)
    if arr.ndim == 3 and arr.shape[-2:] == (chain.dim, chain.dim):
        return np.einsum("nij,ji->n", arr, op)
    return np.einsum("ni,ij,nj->n", arr.conj(), op, arr)
