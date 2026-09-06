import numpy as np

MAG_THRESHOLD = 1e-8 # Threshold where below this number something is considered 0


def dag(op: np.ndarray) -> np.ndarray:
    """Hermitian conjugate, op^dagger = op.conj().T -- so it's called what it is."""
    return op.conj().T

def projector(state: np.ndarray) -> np.ndarray:
    """Return projector operator for a given state vector."""
    return np.outer(state, state.conj())  # |state><state|

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Quantum-state fidelity for any pair of kets or density matrices.

    For two kets this is |<psi|phi>|^2. If either state is mixed, this is the
    squared Uhlmann fidelity (so the pure/mixed case is <psi|rho|psi>).
    """
    a, b = np.asarray(state1, dtype=complex), np.asarray(state2, dtype=complex)
    if a.ndim == b.ndim == 1:
        return float(np.abs(np.vdot(a, b)) ** 2)
    if a.ndim == 1 and b.ndim == 2:
        return float(np.clip(np.real(np.vdot(a, b @ a)), 0.0, 1.0))
    if a.ndim == 2 and b.ndim == 1:
        return float(np.clip(np.real(np.vdot(b, a @ b)), 0.0, 1.0))
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("fidelity needs two kets or square density matrices of equal dimension")
    evals, evecs = np.linalg.eigh((a + a.conj().T) / 2)
    sqrt_a = (evecs * np.sqrt(np.clip(evals, 0.0, None))) @ evecs.conj().T
    middle = sqrt_a @ b @ sqrt_a
    middle = (middle + middle.conj().T) / 2
    roots = np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None))
    return float(np.clip(np.sum(roots) ** 2, 0.0, 1.0))

def process_fidelity(U1: np.ndarray, U2: np.ndarray) -> float:
    """Process fidelity between two propagators: |Tr(U1^dagger U2)|^2 / d^2.

    Operator-space analogue of state fidelity: overlap of two unitaries via
    the Hilbert-Schmidt inner product, normalized by dimension d so F=1 iff
    U1 and U2 agree up to a global phase.
    """
    d = U1.shape[0]
    return float(np.abs(np.trace(U1.conj().T @ U2)) ** 2 / d ** 2)  # |Tr(U1^dag U2)|^2 / d^2

def Tr(op: np.ndarray) -> complex:
    """Trace of a matrix, Tr(op) = sum_i op[i,i]. Named for how it reads in
    a formula (e.g. population = Tr(Proj @ rho)) rather than np.trace(op)
    inline -- the whole point of this module is physics-equation clarity."""
    return np.trace(op)

def relative_phase(state1: np.ndarray, state2: np.ndarray):
    """Relative phase arg(<state1|state2>) = arg(sum_i conj(state1_i) state2_i).

    Accepts either a single pair of states (1D, shape (dim,) each -> returns
    a float) or two stacked trajectories (2D, shape (n_times, dim) each,
    e.g. from HamiltonianEvolution.state_at(ts) -> returns an array of one
    phase per time), by always summing over the last axis. Uses np.angle on
    the inner product; a result numerically at -pi is snapped to +pi so a
    borderline phase doesn't flip sign between neighboring evaluations.
    """
    state1, state2 = np.asarray(state1), np.asarray(state2)
    inner = np.sum(np.conj(state1) * state2, axis=-1)  # <1|2>, batched over leading axis
    phi = np.angle(inner)
    phi = np.where(np.isclose(phi, -np.pi, atol=0.001), np.pi, phi)  # avoid -pi/+pi flip
    return float(phi) if phi.ndim == 0 else phi

def otimes(*vecs):
    """Given 2 or more matrices, calculate the kronecker product of the entire list"""
    if len(vecs) == 1:
        return vecs[0]
    else:
        return np.kron(vecs[0], otimes(*vecs[1:]))  # recursive Kronecker product

def binary_to_index(bin_str):
    """Returns the binary value of the passed string"""
    return int(bin_str, 2)

def index_to_binary(index, bits=None):
    """Returns the binary string representation of the passed integer.
    
    If 'bits' is provided, the binary string will be padded with leading zeros to
    ensure it has at least 'bits' length. If 'bits' is less than the actual binary length, 
    it will return the standard binary string.
    """
    bin_str = bin(index)[2:]  # Convert to binary and remove the "0b" prefix
    if bits is not None:
        # Pad with leading zeros if needed
        bin_str = bin_str.zfill(int(bits))
    return bin_str

def ket(bitstring):
    """Returns the state vector corresponding to the bitstring (ex: '01' -> [0 1 0 0 ])"""
    n = len(bitstring)
    dim = 2**n

    vec = np.zeros(dim, dtype=complex)
    index = binary_to_index(bitstring)

    vec[index] = 1
    return vec

def bra(bitstring):
    """<bitstring| as a 1-D array, so `bra("00") @ psi` reads as <00|psi> --
    the amplitude, with no `.conj()`/`.T` at the call site. Population is
    `abs(bra(s) @ psi) ** 2`. `bra(s) == ket(s)` here since the computational
    basis is real; conjugation is applied for you so this stays correct if
    you build `psi` in a different (complex) basis."""
    return ket(bitstring).conj()

def sampled_pulse(times, values, kind="linear"):
    """Sampled data (times, values) -> a callable f(t), for `coeff=` / `amplitudes=`.

    Every time-dependent quantity in htdse is a callable f(t); this is the
    bridge from measured or solver-produced samples to that form.

    times, values : 1D arrays of the SAME length -- one value per time point.
    kind:
      "linear"   (default) -- values sit AT the sample times; np.interp ramps
                 linearly between them. This is the usual convention for a
                 solved pulse (a ramp/spline between solved amplitude points).
      "previous" -- zero-order hold: the value in effect from a sample time
                 until the next one (a piecewise-constant/step waveform).

    Outside [times[0], times[-1]] the edge value is held (no extrapolation) --
    querying past the pulse's own window is a setup error, and holding the
    boundary keeps a continuation solve well-defined rather than raising for
    time points reached only through a solver's floating-point roundoff.

    SIGN IS PASSED THROUGH UNTOUCHED -- never clipped, rectified or abs()'d.
    A negative sample is physically meaningful: wherever a drive amplitude
    multiplies a spin operator as a plain signed real (as it does throughout,
    e.g. the MS suite's carrier / spin-dependent-force / eta^2 terms), a sign
    flip IS a pi phase shift, since Omega*sigma_theta == (-Omega)*sigma_{theta+pi}.

    Returns a scalar-in/scalar-out callable that also accepts a numpy array
    (both branches are array-vectorized), which is what lets a caller's array
    fast path evaluate a whole grid in one call instead of looping.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape:
        raise ValueError(f"sampled_pulse: times and values must be the same "
                         f"length, got {times.shape} and {values.shape}. "
                         f"(A common mismatch: a solver's cumulative time "
                         f"BREAKPOINTS array has one more entry than its "
                         f"per-segment amplitude array -- trim or resample "
                         f"one of them first; this function does not guess.)")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sampled_pulse: times must be strictly increasing")

    if kind == "linear":
        return lambda t: np.interp(t, times, values)
    if kind == "previous":
        def step(t):
            idx = np.clip(np.searchsorted(times, t, side="right") - 1,
                         0, len(times) - 1)
            return values[idx]
        return step
    raise ValueError(f"sampled_pulse: kind must be 'linear' or 'previous', got {kind!r}")
