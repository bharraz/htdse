import numpy as np

MAG_THRESHOLD = 1e-8 # Threshold where below this number something is considered 0


def projector(state: np.ndarray) -> np.ndarray:
    """Return projector operator for a given state vector."""
    return np.outer(state, state.conj())  # |state><state|

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calculate quantum state fidelity |⟨ψ₁|ψ₂⟩|²."""
    return np.abs(np.vdot(state1, state2))**2  # |<1|2>|^2

def process_fidelity(U1: np.ndarray, U2: np.ndarray) -> float:
    """Process fidelity between two propagators: |Tr(U1^dagger U2)|^2 / d^2.

    Operator-space analogue of state fidelity: overlap of two unitaries via
    the Hilbert-Schmidt inner product, normalized by dimension d so F=1 iff
    U1 and U2 agree up to a global phase.
    """
    d = U1.shape[0]
    return float(np.abs(np.trace(U1.conj().T @ U2)) ** 2 / d ** 2)  # |Tr(U1^dag U2)|^2 / d^2

def density_fidelity(rho: np.ndarray, psi: np.ndarray) -> float:
    """Fidelity between a density matrix and a pure state: <psi|rho|psi>.

    The general (Uhlmann) mixed-state fidelity needs a matrix square root;
    when one operand is pure it reduces exactly to this expectation value,
    so no sqrtm is needed here.
    """
    return float(np.real(psi.conj() @ rho @ psi))  # <psi|rho|psi>

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
