"""Atomic structure: g-factors, hyperfine energies, and the Zeeman + dipole
coupling matrices built from them -- ported from AMO.jl's `Atomic` module
(https://github.com/yuyichao/AMO.jl), on top of this package's own
`angular_momentum.py` (AMO.jl leans on the external
`WignerSymbols.jl`/`RationalRoots.jl` packages for these; htdse hand-rolls
Clebsch-Gordan/Wigner-6j itself, see that module's docstring for the
convention log).

Two matrix builders here work in the coupled |F, mF> (hyperfine_matrix) or
successively-coupled (dipole_couple_matrix) bases rather than an uncoupled
product basis -- ported faithfully to AMO's *physics*, but written as a
direct double loop over every (signed) mF rather than AMO's optimization of
only iterating mF>=0 and deriving the rest by a parity-based reflection
trick. Same result, simpler to verify (AMO.jl itself has no tests for either
matrix builder -- see the plan's M3 test notes for the coverage added here
instead).
"""
import numpy as np

from .angular_momentum import clebsch_gordan, wigner_6j

g_s = 2.00231930436092   # electron spin g-factor
g_l = 1.0                # electron orbital g-factor


def g_sum(J, J1, g1, J2, g2) -> float:
    """g-factor of J, formed by coupling J1 (g-factor g1) with J2 (g-factor g2)."""
    return (g1 * (J * (J + 1) + J1 * (J1 + 1) - J2 * (J2 + 1))
            + g2 * (J * (J + 1) + J2 * (J2 + 1) - J1 * (J1 + 1))) / (2 * J * (J + 1))


def hyperfine(F, I, J, A, B=0.0, C=0.0) -> float:
    """Hyperfine energy of state F, coupled from nuclear spin I and electronic
    angular momentum J with hyperfine constants A (magnetic dipole), B
    (electric quadrupole, needs I>1/2 and J>1/2), C (magnetic octupole, needs
    I>1 and J>1)."""
    if F < 0 or I < 0 or J < 0:
        raise ValueError(f"hyperfine: F={F}, I={I}, J={J} must all be non-negative")
    F2, I2, J2 = F * (F + 1), I * (I + 1), J * (J + 1)
    K = F2 - (I2 + J2)
    res = A * K / 2
    if I > 0.5 and J > 0.5:
        IJ2 = I2 * J2
        D2 = I * (2 * I - 1) * J * (2 * J - 1)
        Q = 0.375 * K * (K + 1) - 0.5 * IJ2
        res += B * Q / D2
        if I > 1 and J > 1:
            O = 5 * K ** 2 * (K * 0.25 + 1) + K * (I2 + J2 + 3 - 3 * IJ2) - 5 * IJ2
            res += C * O / (D2 * (I - 1) * (J - 1))
    return res


def spin_manifold(*Js):
    """Generator over the successive-coupling chain of angular momenta Js:
    yields tuples where element 0 is Js[0] and element k is the resultant
    total angular momentum after further coupling in Js[k] (k=1,2,...) --
    i.e. every valid intermediate total when Js are coupled one at a time,
    left to right. Empty for no arguments; a single tuple (Js[0],) for one."""
    if not Js:
        return

    def _rec(prev, remaining):
        if not remaining:
            yield (prev,)
            return
        nxt, *rest = remaining
        j = abs(prev - nxt)
        j_hi = prev + nxt
        while j <= j_hi + 1e-9:
            for tail in _rec(j, rest):
                yield (prev,) + tail
            j += 1.0

    yield from _rec(Js[0], list(Js[1:]))


def couple_reduced_element(J1p, J2p, J0, J1, J2, k=1) -> float:
    """Ratio R between reduced matrix elements <J1'||T^k||J2'> = R <J1||T^k||J2>,
    where J1', J2' are formed by coupling a spectator angular momentum J0
    (not itself acted on by the rank-k tensor operator T) to J1, J2
    respectively (Wigner-Eckart theorem bookkeeping)."""
    r = np.sqrt((2 * J1 + 1) * (2 * J2p + 1)) * wigner_6j(J1, J2, k, J2p, J1p, J0)
    return -r if int(round(J0 + J1 + J2p + k)) % 2 else r


def dipole_couple_matrix(L1, L2, Omegas, S=()) -> np.ndarray:
    """Dipole-coupling Hamiltonian matrix between two orbital-angular-momentum
    manifolds L1, L2 (must satisfy |L1-L2| in {0,1}), in the basis formed by
    successively coupling in each spectator spin in S (e.g. electron spin,
    nuclear spin), via the Wigner-Eckart theorem:

        <L1,S..,J1,mJ1| Omega_q |L2,S..,J2,mJ2>
            = Omega[q] * (product of couple_reduced_element factors, one per
              spectator spin) * <J2,mJ2; 1,-q | J1,mJ1>

    i.e. rows are indexed by the L1-side manifold, columns by the L2-side
    manifold (M[i1, i2], not the reverse).

    Omegas = (Omega[q=-1], Omega[q=0], Omega[q=+1]) are the sigma-/pi/sigma+
    polarization coupling strengths. S=() (no spectator spins) reduces to the
    bare two-level dipole selection rule."""
    if L1 < 0 or L2 < 0:
        raise ValueError(f"dipole_couple_matrix: L1={L1}, L2={L2} must both be non-negative")
    if not (np.isclose(abs(L1 - L2), 0) or np.isclose(abs(L1 - L2), 1)):
        raise ValueError(f"dipole_couple_matrix: L1={L1}, L2={L2} not dipole-coupled "
                         f"(|L1-L2| must be 0 or 1)")
    S = tuple(S)
    n_spin = len(S)
    manifolds1 = list(spin_manifold(L1, *S))
    manifolds2 = list(spin_manifold(L2, *S))

    dim1 = sum(int(round(2 * m[-1] + 1)) for m in manifolds1)
    dim2 = sum(int(round(2 * m[-1] + 1)) for m in manifolds2)
    M = np.zeros((dim1, dim2), dtype=complex)

    offset1 = 0
    for j1s in manifolds1:
        j_end1 = j1s[-1]
        block1 = int(round(2 * j_end1 + 1))
        offset2 = 0
        for j2s in manifolds2:
            j_end2 = j2s[-1]
            block2 = int(round(2 * j_end2 + 1))

            scale = 1.0
            for i in range(n_spin):
                scale *= couple_reduced_element(j1s[i + 1], j2s[i + 1], S[i], j1s[i], j2s[i], 1)

            if scale != 0:
                for imj1, mj1 in enumerate(np.arange(-j_end1, j_end1 + 0.5)):
                    for omega_k in (-1, 0, 1):
                        mj2 = mj1 + omega_k
                        if -j_end2 - 1e-9 <= mj2 <= j_end2 + 1e-9:
                            cg = clebsch_gordan(j_end2, mj2, 1, -omega_k, j_end1, mj1)
                            if cg != 0:
                                imj2 = int(round(mj2 + j_end2))
                                M[offset1 + imj1, offset2 + imj2] = Omegas[omega_k + 1] * scale * cg
            offset2 += block2
        offset1 += block1
    return M


def hyperfine_matrix(I, J, Bm=0.0, g_I=0.0, g_J=0.0, Ahf=0.0, Bhf=0.0, Chf=0.0) -> np.ndarray:
    """Hyperfine + linear Zeeman Hamiltonian matrix in the coupled |F, mF>
    basis (F = |I-J| .. I+J), dimension (2I+1)(2J+1). At Bm=0 this is exactly
    block-diagonal in F with each block's diagonal equal to hyperfine(F, I,
    J, ...); at finite field, states of different F but the same mF mix
    (the standard Breit-Rabi problem) -- diagonalize the result (e.g.
    np.linalg.eigh) for the actual energy levels."""
    if I < 0 or J < 0:
        raise ValueError(f"hyperfine_matrix: I={I}, J={J} must both be non-negative")
    F_min, F_max = abs(I - J), I + J
    n_F = int(round(F_max - F_min)) + 1
    Fs = [F_min + k for k in range(n_F)]
    block_dim = {F: int(round(2 * F + 1)) for F in Fs}
    offset = {}
    off = 0
    for F in Fs:
        offset[F] = off
        off += block_dim[F]
    dim = off

    M = np.zeros((dim, dim), dtype=complex)
    for F in Fs:
        Ehf = hyperfine(F, I, J, Ahf, Bhf, Chf)
        offF = offset[F]
        for k in range(block_dim[F]):
            M[offF + k, offF + k] += Ehf

    g_J_eff, g_I_eff = g_J * Bm, g_I * Bm
    if g_J_eff != 0 or g_I_eff != 0:
        for iF, F in enumerate(Fs):
            offF = offset[F]
            for Fp in Fs[iF:]:
                offFp = offset[Fp]
                mF_max = min(F, Fp)
                mF = -mF_max
                while mF <= mF_max + 1e-9:
                    if not (F == Fp and mF == 0):
                        ele = 0.0
                        mJ = max(-J, mF - I)
                        mJ_end = min(J, mF + I)
                        while mJ <= mJ_end + 1e-9:
                            mI = mF - mJ
                            term = g_J_eff * mJ + g_I_eff * mI
                            if term != 0:
                                ele += term * (clebsch_gordan(I, mI, J, mJ, F, mF)
                                              * clebsch_gordan(I, mI, J, mJ, Fp, mF))
                            mJ += 1.0
                        idx1 = offF + int(round(mF + F))
                        idx2 = offFp + int(round(mF + Fp))
                        if idx1 == idx2:
                            M[idx1, idx2] += ele
                        else:
                            M[idx1, idx2] = ele
                            M[idx2, idx1] = ele
                    mF += 1.0
    return M


def hyperfine_levels(I, J, Bm=0.0, g_I=0.0, g_J=0.0, Ahf=0.0, Bhf=0.0, Chf=0.0):
    """(energies, states) of hyperfine_matrix(...), sorted ascending -- the
    full Breit-Rabi answer (AMO.jl builds the matrix but never diagonalizes
    it; np.linalg.eigh is exact here since the matrix is Hermitian)."""
    return np.linalg.eigh(hyperfine_matrix(I, J, Bm, g_I, g_J, Ahf, Bhf, Chf))
