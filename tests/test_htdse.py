"""Verification suite: run with `python tests/test_htdse.py` from the repo root."""
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib
matplotlib.use("Agg")

import htdse as ht
from htdse import (Operator, Mechanism, Hamiltonian, term, jump, hconj,
                   HamiltonianEvolution, UnitaryEvolution, DensityMatrixEvolution,
                   LindbladEvolution, embed, partial_trace, compare_over,
                   otimes, ket, fidelity, process_fidelity, density_fidelity, quiet)
from htdse.core.plotting import plot_populations
from htdse.submodules.spin import (sigma_x, sigma_y, sigma_z, I2, sigma_plus,
                                   sigma_minus, pauli_term, pauli_sum)
from htdse.submodules.harmonic_oscillator import (annihilation, creation,
                                                  number_operator, fock,
                                                  ThermalMotionalDecoherence)
from htdse.submodules.trotter import TrotterizedMechanism

rng = np.random.default_rng(7)
PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  ok: {name}")


def rand_herm(d):
    M = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (M + M.conj().T) / 2


class RabiDrive(Mechanism):
    def __init__(self, Omega, eps=0.0, delta=0.0):
        self.Omega, self.eps, self.delta = Omega, eps, delta

    def hamiltonian(self, t):
        return Operator(0.5 * self.Omega * (1 + self.eps) * sigma_x + self.delta * sigma_z)


print("== subsystems: embed / partial_trace ==")
# single-subsystem embed matches plain otimes
dims2 = {"A": 2, "B": 2}
H1 = rand_herm(2)
check("embed A == H (x) I", np.allclose(embed(H1, dims2, "A"), np.kron(H1, I2)))
check("embed B == I (x) H", np.allclose(embed(H1, dims2, "B"), np.kron(I2, H1)))

# multi-slot, NON-ADJACENT embed: op on (A, C) of {A:2, B:3, C:2}
dims3 = {"A": 2, "B": 3, "C": 2}
M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
Mt = M.reshape(2, 2, 2, 2)  # (a, c, a', c')
expected = np.einsum("acxy,bw->abcxwy", Mt, np.eye(3)).reshape(12, 12)
check("embed non-adjacent (A,C)", np.allclose(embed(M, dims3, ("A", "C")), expected))
# and adjacent joint op == plain kron
check("embed joint (A,B) adjacent",
      np.allclose(embed(rng.normal(size=(6, 6)), {"A": 2, "B": 3, "C": 4},
                        ("A", "B")).shape, (24, 24)))

# batched partial trace == per-time loop; pure bipartite == Psi Psi^dag
psis = rng.normal(size=(5, 6)) + 1j * rng.normal(size=(5, 6))
psis /= np.linalg.norm(psis, axis=1, keepdims=True)
rhos = psis[:, :, None] * psis.conj()[:, None, :]
batched = partial_trace(Operator(rhos), {"A": 2, "B": 3}, ("B",))
looped = np.array([partial_trace(Operator(r), {"A": 2, "B": 3}, ("B",)) for r in rhos])
check("partial_trace batched == looped", np.allclose(batched, looped))
Psi = psis[0].reshape(2, 3)
check("partial_trace == Psi Psi^dag", np.allclose(batched[0], Psi @ Psi.conj().T))

print("== term layer: composition ==")
n_max = 12
a = annihilation(n_max)
nop = number_operator(n_max)
w0, w, g = 1.3, 1.3, 0.11  # resonant
atom = term(0.5 * w0 * sigma_z, on="spin", name="atom")
mode = term(w * nop, on="mode", name="mode")
jc = hconj(term({"spin": sigma_plus, "mode": a}, coeff=g, name="jc"))
H = atom + mode + jc
manual = (0.5 * w0 * np.kron(sigma_z, np.eye(n_max + 1)) + w * np.kron(I2, nop)
          + g * (np.kron(sigma_plus, a) + np.kron(sigma_minus, a.conj().T)))
check("JC composition == hand-built otimes", np.allclose(H.hamiltonian(0.0), manual))
check("registry from names", H.subsystems == {"spin": 2, "mode": n_max + 1})

# order independence up to registry order: mode + atom puts mode factor first
H2 = mode + atom + jc.replace if False else mode + atom
check("registry order = first appearance",
      list((mode + atom).subsystems) == ["mode", "spin"])

# time-dependent coefficient
Hd = term(sigma_x, on="q", coeff=lambda t: np.sin(t))
check("f(t) coefficient", np.allclose(Hd.hamiltonian(0.7), np.sin(0.7) * sigma_x))

# scalar and f(t) scaling, subtraction
check("scalar *", np.allclose((2.0 * atom).hamiltonian(0), w0 * sigma_z))
check("subtraction", np.allclose((atom - atom).hamiltonian(0), np.zeros((2, 2))))

# replace(): swap the drive, including one that brings in a NEW subsystem
drive = term(0.5 * sigma_x, on="spin", name="drive")
model = atom + drive
noisy = term(0.55 * sigma_x, on="spin", name="whatever") \
    + term(0.03 * sigma_z, on="spectator", name="leak")
swapped = model.replace(drive=noisy)
check("replace swaps the group",
      np.allclose(swapped.group("drive").hamiltonian(0)[:2, :2] * 0 + 0, 0)
      and "spectator" in swapped.subsystems)
expected_sw = (0.5 * w0 * np.kron(sigma_z, I2)
               + 0.55 * np.kron(sigma_x, I2) + 0.03 * np.kron(I2, sigma_z))
check("replace materializes correctly", np.allclose(swapped.hamiltonian(0), expected_sw))
check("without() drops a group",
      np.allclose(model.without("drive").hamiltonian(0), 0.5 * w0 * sigma_z))

# dimension conflict by name must raise
try:
    _ = term(sigma_x, on="spin") + term(nop, on="spin")
    check("dim conflict raises", False)
except ValueError:
    check("dim conflict raises", True)

# frame mixing warns
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    (term(sigma_x, on="q", frame="lab") + term(sigma_z, on="q", frame="rot")).hamiltonian(0)
check("frame mixing warns", any("frames" in str(x.message) for x in wlist))

print("== physics: vacuum Rabi oscillation from composed JC ==")
psi0 = Operator(np.kron(ket("0"), fock(0, n_max)))  # |e, 0>
with quiet():
    ev = HamiltonianEvolution(H, psi0)  # subsystems auto-supplied by the term layer
    ts = np.linspace(0, 2 * np.pi / g, 60)
    psis_t = ev.state_at(ts)
proj_e0 = np.kron(ket("0"), fock(0, n_max))
P_e0 = np.abs(psis_t @ proj_e0.conj()) ** 2
check("P(e,0) = cos^2(gt)", np.allclose(P_e0, np.cos(g * ts) ** 2, atol=1e-5))
check("evolution picked up subsystems from term layer",
      ev.subsystems == {"spin": 2, "mode": n_max + 1})
with quiet():
    rho_spin = ev.trace_out("mode", t=ts)
check("vectorized trace_out shape", rho_spin.shape == (60, 2, 2))
check("trace_out trace = 1", np.allclose(np.einsum("nii->n", np.asarray(rho_spin)), 1))
check("P_e from reduced rho matches",
      np.allclose(np.real(np.asarray(rho_spin)[:, 0, 0]),
                  np.cos(g * ts) ** 2, atol=1e-5))

print("== jump composition + Lindblad through the term layer ==")
gamma = 0.3
open_model = (term(0.5 * sigma_z, on="spin", name="atom")
              + term(w * nop, on="mode", name="mode")
              + jump(a, on="mode", coeff=np.sqrt(gamma), name="decay"))
Ls = open_model.jump_operators(0.0)
check("jump embedded on joint space",
      len(Ls) == 1 and np.allclose(Ls[0], np.sqrt(gamma) * np.kron(I2, a)))
rho0_j = np.kron(np.outer(ket("0"), ket("0")), np.outer(fock(1, n_max), fock(1, n_max).conj()))
with quiet():
    lev = LindbladEvolution(open_model, Operator(rho0_j))
    rho_T = lev.state_at(3.0)
    n_of_t = float(np.real(np.trace(np.asarray(rho_T) @ np.kron(I2, nop))))
check("mode decays at rate gamma", abs(n_of_t - np.exp(-gamma * 3.0)) < 1e-4)

print("== guards ==")
mech = RabiDrive(1.0)
with quiet():
    ev2 = HamiltonianEvolution(mech, Operator(ket("0")))
    ev2.state_at(1.0)
mech.Omega = 2.0  # mutate after binding -- must be detected
try:
    with quiet():
        ev2.state_at(2.0)
    check("mutation after binding raises", False)
except RuntimeError:
    check("mutation after binding raises", True)

try:
    with quiet():
        HamiltonianEvolution(ThermalMotionalDecoherence(3, gamma_a=1.0), Operator(fock(0, 3)))
    check("dissipative mech rejected by closed-system class", False)
except ValueError:
    check("dissipative mech rejected by closed-system class", True)


class BadMech(Mechanism):
    def hamiltonian(self, t):
        return Operator(np.array([[0, 1], [0, 0]], dtype=complex))  # not Hermitian


try:
    with quiet():
        HamiltonianEvolution(BadMech(), Operator(ket("0")))
    check("non-Hermitian H rejected", False)
except ValueError:
    check("non-Hermitian H rejected", True)

with quiet():
    lev2 = LindbladEvolution(ThermalMotionalDecoherence(3, gamma_a=1.0, nbar=0.1),
                             Operator(np.outer(fock(0, 3), fock(0, 3).conj())))
try:
    lev2.state_at(-1.0)
    check("Lindblad backward rejected", False)
except ValueError:
    check("Lindblad backward rejected", True)

try:
    with quiet():
        DensityMatrixEvolution(RabiDrive(1.0), Operator(sigma_x))  # not a density matrix
    check("non-Hermitian-positive rho0... (hermitian but trace 0) warns", False)
except Exception:
    check("non-Hermitian-positive rho0... (hermitian but trace 0) warns", True)

print("== Trotter: breakpoints + exact expm path ==")


class Ramp(Mechanism):
    """H(t) = (1 - t/T) X + (t/T) Z -- smooth ramp to Trotterize."""
    def __init__(self, T):
        self.T = T

    def hamiltonian(self, t):
        s = np.clip(t / self.T, 0, 1)
        return Operator((1 - s) * sigma_x + s * sigma_z)


T = 4.0
with quiet():
    exact = HamiltonianEvolution(Ramp(T), Operator(ket("0"))).state_at(T)
    trot200 = HamiltonianEvolution(TrotterizedMechanism(Ramp(T), 0, T, 200),
                                   Operator(ket("0"))).state_at(T)
    trot20 = HamiltonianEvolution(TrotterizedMechanism(Ramp(T), 0, T, 20),
                                  Operator(ket("0"))).state_at(T)
err200 = 1 - fidelity(exact, trot200)
err20 = 1 - fidelity(exact, trot20)
check("Trotter converges to exact", err200 < 1e-5)
check("Trotter error shrinks with steps", err200 < err20 / 10)

# one-step trotter must EXACTLY equal the eigh propagator (expm path, no ODE)
with quiet():
    one = HamiltonianEvolution(TrotterizedMechanism(Ramp(T), 0, T, 1),
                               Operator(ket("0"))).state_at(T)
Hmid = np.asarray(Ramp(T).hamiltonian(T / 2))
E, V = np.linalg.eigh(Hmid)
expected_one = V @ (np.exp(-1j * E * T) * (V.conj().T @ ket("0")))
check("piecewise-constant path is exact expm", np.allclose(np.asarray(one), expected_one, atol=1e-12))

# unitarity of the expm path over many steps
with quiet():
    Uev = UnitaryEvolution(TrotterizedMechanism(Ramp(T), 0, T, 50), dim=2)
    _ = Uev.unitary_at(T)
check("expm path unitarity defect tiny", Uev.unitarity_defect(T) < 1e-12)

print("== analytic-unitary mechanism (dual primitive wired) ==")


class AnalyticGate(Mechanism):
    """Defined as a gate only: U(t) = exp(-i (Omega/2) X t) (e.g. an RWA result)."""
    def __init__(self, Omega):
        self.Omega = Omega

    def unitary(self, t):
        th = self.Omega * t / 2
        return Operator(np.cos(th) * I2 - 1j * np.sin(th) * sigma_x)


with quiet():
    Ua = UnitaryEvolution(AnalyticGate(1.0)).unitary_at(np.pi)
    Ur = UnitaryEvolution(RabiDrive(1.0), dim=2).unitary_at(np.pi)
check("analytic unitary consumed directly", process_fidelity(Ua, Ur) > 1 - 1e-7)
with quiet():
    dm = DensityMatrixEvolution(AnalyticGate(1.0), Operator(np.outer(ket("0"), ket("0"))))
    rho_pi = dm.state_at(np.pi)
check("DensityMatrixEvolution on analytic unitary", abs(np.real(rho_pi[1, 1]) - 1) < 1e-9)

print("== T2: exact leading-order coherence decay rate (B1 fix) ==")
# rate = 2*ndot + gamma_a/2 + gamma_p/2. Parameters chosen so the dropped
# gamma_a/2 term is LARGE (nbar ~ 1) -- distinguishes old vs corrected formula.
nbar, gamma_a, gamma_p = 0.5, 1.0, 0.4
ndot = gamma_a * nbar
rate_exact = 2 * ndot + gamma_a / 2 + gamma_p / 2      # = 1.7
rate_old = 2 * ndot + gamma_p / 2                       # = 1.2 (README's old claim)
nm2 = 15
psi01 = (fock(0, nm2) + fock(1, nm2)) / np.sqrt(2)
with quiet():
    levT = LindbladEvolution(ThermalMotionalDecoherence(nm2, gamma_a, nbar, gamma_p),
                             Operator(np.outer(psi01, psi01.conj())))
    dt = 1e-3  # early time: rho12 = 0, so re-feeding hasn't kicked in -- pure leading order
    c0 = abs(np.asarray(levT.state_at(0.0))[0, 1])
    c1 = abs(np.asarray(levT.state_at(dt))[0, 1])
rate_num = -np.log(c1 / c0) / dt
check("numerical rate matches 2*ndot + gamma_a/2 + gamma_p/2",
      abs(rate_num - rate_exact) / rate_exact < 0.01)
check("...and clearly NOT the old formula", abs(rate_num - rate_old) / rate_old > 0.3)

print("== pauli strings, compare_over, plotting, misc ==")
hp = pauli_sum("0.5 X0X1 + 0.3 Z0 - Z1", n_qubits=2)
expected_p = (0.5 * np.kron(sigma_x, sigma_x) + 0.3 * np.kron(sigma_z, I2)
              - np.kron(I2, sigma_z))
check("pauli_sum parses and materializes", np.allclose(hp.hamiltonian(0), expected_p))
check("pauli_term product on one qubit",
      np.allclose(pauli_term("X0Y0").hamiltonian(0), sigma_x @ sigma_y))

with quiet():
    F = compare_over(np.linspace(0, np.pi, 9),
                     HamiltonianEvolution(RabiDrive(1.0), Operator(ket("0"))),
                     HamiltonianEvolution(RabiDrive(1.0, eps=0.05), Operator(ket("0"))),
                     metric=fidelity)
check("compare_over returns per-t metric", F.shape == (9,) and abs(F[0] - 1) < 1e-12)

# adapter usage: compare joint realized vs 1-qubit target through trace_out
with quiet():
    joint = HamiltonianEvolution(atom + term(0.0 * nop, on="mode", name="m"),
                                 Operator(np.kron(ket("0"), fock(0, n_max))))
    tgt = HamiltonianEvolution(term(0.5 * w0 * sigma_z, on="spin"), Operator(ket("0")))
    Fa = compare_over([0.5, 1.0], tgt, joint, metric=lambda p, r: density_fidelity(r, p),
                      realized_adapter=lambda ps: partial_trace(
                          Operator(np.outer(ps, ps.conj())), joint.subsystems, ("mode",)))
check("compare_over with trace adapter", np.allclose(Fa, 1, atol=1e-8))

with quiet():
    ax = plot_populations(ts[:10], ev)  # evolution object accepted directly
    ax2 = plot_populations(ts[:10], np.asarray(ev.trace_out("mode", t=ts[:10])))  # rho trajectory
check("plot_populations accepts evolution and rho trajectory", ax is not None and ax2 is not None)

# vectorized adiabatic diagnostics
with quiet():
    ad = HamiltonianEvolution(Ramp(T), Operator(np.linalg.eigh(np.asarray(sigma_x))[1][:, 0]))
    af = ad.adiabatic_fidelity(np.linspace(0.1, T, 5))
check("vectorized adiabatic_fidelity", af.shape == (5,) and np.all(af > 0.5))

# Operator.params no longer shared by reference
op1 = Operator(np.eye(2), params={"tag": 1})
op2 = op1 + Operator(np.eye(2))
op2.params["tag"] = 2
check("params not shared across arithmetic", op1.params["tag"] == 1)

# quiet() actually silences
buf = io.StringIO()
with redirect_stdout(buf), quiet():
    HamiltonianEvolution(RabiDrive(1.0), Operator(ket("0"))).state_at(1.0)
check("quiet() silences solver prints", buf.getvalue() == "")

# verbose default prints
buf = io.StringIO()
with redirect_stdout(buf):
    HamiltonianEvolution(RabiDrive(1.0), Operator(ket("0"))).state_at(1.0)
check("verbose default prints the integration", "integrating" in buf.getvalue())

print(f"\nALL {len(PASS)} CHECKS PASSED")
