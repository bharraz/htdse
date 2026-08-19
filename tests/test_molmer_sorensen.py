"""Spin-boson / MS suite verification: `python tests/test_molmer_sorensen.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib
matplotlib.use("Agg")

from htdse import (UnitaryEvolution, HamiltonianEvolution, quiet,
                   process_fidelity, fidelity, otimes, ket)
from htdse.submodules.harmonic_oscillator import fock, thermal
from htdse.submodules.spin import sigma_y, sigma_x, pauli_term
from htdse.submodules.spin_boson import (Tone, Mode, driven_spins, jaynes_cummings,
                                         rabi, exact_drive)
from htdse.submodules.trapped_ion import IonChain
from htdse.submodules.molmer_sorensen import (ms_tones, ms_closed_form, MSMode,
                                              plot_phase_space, expectation_alpha)

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

# common gate parameters: 2 ions, equal participation
b = [1.0, 1.0]
eta, delta, n_max = 0.1, 0.5, 8
Omega = 0.8
nu = 40.0
T = 2 * np.pi / delta          # one loop
phases = [0.0, 0.0]
mu = nu + delta

print("== tone table: one tone at a time, after RWA ==")
mode1q = Mode(nu=nu, eta=eta, n_max=n_max, name="mode")

# carrier: Tone(0) -> (Om/2) sigma_phi, no mode content
H_carrier = driven_spins([Tone(offset=0.0, amp=Omega, phase=0.4)], ["q0"], [mode1q],
                         lamb_dicke=1, rwa=True)
H1q = (Omega / 2) * (np.cos(0.4) * sigma_x + np.sin(0.4) * sigma_y)
I_mode = np.eye(n_max + 1)
check("Tone(0), rwa=True == plain (Om/2) sigma_phi (x) I",
      np.allclose(np.asarray(H_carrier.hamiltonian(0.5)), np.kron(H1q, I_mode), atol=1e-10))

# red sideband -> JC, with the intrinsic i*eta*Omega/2 recoil coupling
g = 1j * eta * Omega / 2
H_jc = jaynes_cummings(g, spin="q0", mode="mode", n_max=n_max)
H_red = driven_spins([Tone(offset=-nu, amp=Omega, phase=0.0)], ["q0"], [mode1q],
                     lamb_dicke=1, rwa=True)
for tt in [0.0, 0.3, 1.1]:
    check(f"red sideband == jaynes_cummings(g=i*eta*Om/2) at t={tt}",
          np.allclose(np.asarray(H_jc.hamiltonian(tt)), np.asarray(H_red.hamiltonian(tt)), atol=1e-10))

# blue sideband -> anti-JC (h.c. of the red case, up to which operator is adag)
H_blue = driven_spins([Tone(offset=nu, amp=Omega, phase=0.0)], ["q0"], [mode1q],
                      lamb_dicke=1, rwa=True)
check("blue sideband produces sigma_+ a^dag content (anti-JC), not sigma_+ a",
      not np.allclose(np.asarray(H_blue.hamiltonian(0.3)), np.asarray(H_red.hamiltonian(0.3)), atol=1e-6))

print("== ms_tones + driven_spins reproduces the old symmetric two-tone builder exactly ==")
mode = Mode(nu=nu, eta=eta * np.array(b), n_max=n_max, name="mode")
tones = ms_tones(nu, delta, Omega, theta=0.0, psi=0.0)
H1 = driven_spins(tones, ["q0", "q1"], [mode], lamb_dicke=1, rwa=False)
H2 = driven_spins(tones, ["q0", "q1"], [mode], lamb_dicke=2, rwa=False)
H_rwa = driven_spins(tones, ["q0", "q1"], [mode], lamb_dicke=1, rwa=True)
check("H1(t) Hermitian", np.max(np.abs(np.asarray(H1.hamiltonian(0.3))
                                       - np.asarray(H1.hamiltonian(0.3)).conj().T)) < 1e-12)
check("H2(t) Hermitian", np.max(np.abs(np.asarray(H2.hamiltonian(0.3))
                                       - np.asarray(H2.hamiltonian(0.3)).conj().T)) < 1e-12)
check("groups per ion, swappable",
      {"carrier_q0_tone0", "carrier_q0_tone1", "sdf_q0_mode_tone0", "sdf_q0_mode_tone1"} <= set(H1.groups))

print("== ms_closed_form == driven_spins(ms_tones(...), lamb_dicke=1, rwa=True), solved by ODE ==")
gate = ms_closed_form(b, eta, delta, Omega, phases, n_max)
rng = np.random.default_rng(3)
lowf = rng.normal(size=n_max + 1) + 1j * rng.normal(size=n_max + 1)
lowf[3:] = 0.0
lowf /= np.linalg.norm(lowf)
test_states = [otimes(ket("00"), fock(0, n_max)),
               otimes(ket("01"), fock(1, n_max)),
               otimes((ket("00") + ket("11")) / np.sqrt(2), lowf)]
with quiet():
    for tt in [0.3 * T, 0.7 * T, T]:
        U_gate = np.asarray(gate.unitary(tt))
        U_ode = np.asarray(UnitaryEvolution(H_rwa, dim=gate.dim).unitary_at(tt))
        worst = min(fidelity(U_gate @ p, U_ode @ p) for p in test_states)
        check(f"closed form == ODE'd RWA on low-Fock states at t={tt/T:.1f}T (worst F={worst:.2e})",
              worst > 1 - 1e-7)

print("== analytic ingredients ==")
c = -eta * 1.0 * Omega / 2
ts = np.linspace(0, T, 400)
alpha_num = gate.alpha_trajectory(ts)[:, 0, 0]
alpha_ana = (c / delta) * (np.exp(-1j * delta * ts) - 1)
check("alpha(t) matches analytic circle", np.max(np.abs(alpha_num - alpha_ana)) < 1e-6)
check("loop closes at T = 2*pi/delta", abs(gate.alpha(T)[0, 0]) < 1e-6)
Th = gate.geometric_phase(T)
Th_ana = c * c * (np.sin(delta * T) / delta ** 2 - T / delta)
check("geometric phase matches analytic", abs(Th[0, 1] - Th_ana) < 1e-6 * abs(Th_ana))

print("== composability: recoil MS + a cavity mode via plain Model + ==")
H_cavity = jaynes_cummings(0.3, spin="q0", mode="cavity", n_max=4)
H_combo = H1 + H_cavity
check("driven_spins(...) + jaynes_cummings(..., mode='cavity') composes and is Hermitian",
      np.allclose(np.asarray(H_combo.hamiltonian(0.4)), np.asarray(H_combo.hamiltonian(0.4)).conj().T, atol=1e-10))
check("composed system carries both subsystems", "cavity" in H_combo.subsystems and "mode" in H_combo.subsystems)

print("== quantum Rabi model: no RWA, dipole coupling ==")
H_rabi = rabi(0.2, spin="q0", mode="cavity", n_max=4)
check("rabi(...) Hermitian", np.allclose(np.asarray(H_rabi.hamiltonian(0.0)),
                                         np.asarray(H_rabi.hamiltonian(0.0)).conj().T, atol=1e-10))

print("== IonChain ==")
chain = IonChain(modes=[mode], n_ions=2)
check("IonChain.subsystems matches driven_spins' own", chain.subsystems == H1.subsystems)
H_chain = chain.drive(tones, lamb_dicke=1)
check("chain.drive(...) == driven_spins(...) directly",
      np.allclose(np.asarray(H_chain.hamiltonian(0.3)), np.asarray(H1.hamiltonian(0.3)), atol=1e-12))

print("== thermal states ==")
rho = thermal(2.0, 30)
check("thermal(nbar,...) trace 1", abs(np.trace(rho).real - 1.0) < 1e-9)
nvec = np.arange(31)
check("thermal(nbar,...) <n> == nbar", abs(np.real(np.trace(rho @ np.diag(nvec))) - 2.0) < 1e-2)
rho0 = thermal(0.0, 5)
check("thermal(0,...) == fock(0)", np.allclose(rho0, np.outer(fock(0, 5), fock(0, 5).conj())))
rho_chain = chain.thermal([1.0])
check("IonChain.thermal(...) trace 1", abs(np.trace(rho_chain).real - 1.0) < 1e-9)

print("== exact rung (lamb_dicke=None): carrier Rabi frequency vs Laguerre closed form ==")
from scipy.special import eval_genlaguerre
from scipy.signal import find_peaks
eta_x, Om_x, nu_x, nmax_x = 0.2, 1.0, 60.0, 12
mode_x = Mode(nu=nu_x, eta=eta_x, n_max=nmax_x, name="mode")
sys_exact = driven_spins([Tone(offset=0.0, amp=Om_x, phase=0.0)], ["q0"], [mode_x], lamb_dicke=None)
check("driven_spins(lamb_dicke=None) returns a System (no + / .replace())",
      not hasattr(sys_exact, "groups"))
for n_level in [0, 2]:
    Om_n_theory = Om_x * np.exp(-eta_x ** 2 / 2) * eval_genlaguerre(n_level, 0, eta_x ** 2)
    psi0 = otimes(ket("0"), fock(n_level, nmax_x))
    T_expected = 2 * np.pi / abs(Om_n_theory)
    tt = np.linspace(0, 3 * T_expected, 6000)
    with quiet():
        psis = HamiltonianEvolution(sys_exact, psi0).state_at(tt)
    idx_down = n_level
    P_down = np.abs(psis[:, idx_down]) ** 2
    peaks, _ = find_peaks(-P_down, prominence=0.3)
    spacing = np.mean(np.diff(tt[peaks]))
    Om_n_meas = 2 * np.pi / spacing
    err = abs(Om_n_meas - abs(Om_n_theory)) / abs(Om_n_theory)
    check(f"carrier Rabi freq n={n_level} matches Om*e^-eta^2/2*L_n(eta^2) (rel err {err:.1e})",
          err < 1e-3)

print("== asymmetric tones: ms_tones(delta_red=, amp_red=) ==")
delta_red = 0.65
tones_sym_explicit = ms_tones(nu, delta, Omega, theta=[0.0, 0.0], delta_red=delta)
H_sym = driven_spins(ms_tones(nu, delta, Omega, theta=[0.0, 0.0]), ["q0", "q1"], [mode], lamb_dicke=1)
H_sym_explicit = driven_spins(tones_sym_explicit, ["q0", "q1"], [mode], lamb_dicke=1)
for tt in [0.0, 0.4, 1.1]:
    check(f"delta_red=delta reduces exactly to the symmetric drive, t={tt}",
          np.allclose(np.asarray(H_sym.hamiltonian(tt)),
                      np.asarray(H_sym_explicit.hamiltonian(tt)), atol=1e-10))

tones_asym = ms_tones(nu, delta, Omega, theta=[0.0, 0.0], delta_red=delta_red)
H_asym = driven_spins(tones_asym, ["q0", "q1"], [mode], lamb_dicke=1)
H_asym_t = np.asarray(H_asym.hamiltonian(0.3))
check("genuinely asymmetric drive stays Hermitian",
      np.allclose(H_asym_t, H_asym_t.conj().T, atol=1e-10))
check("genuinely asymmetric drive differs from the symmetric one",
      not np.allclose(H_asym_t, np.asarray(H_sym.hamiltonian(0.3)), atol=1e-6))

tones_amp_asym = ms_tones(nu, delta, Omega, theta=[0.0, 0.0], amp_red=Omega * 1.5)
H_amp_asym = driven_spins(tones_amp_asym, ["q0", "q1"], [mode], lamb_dicke=1)
H_amp_t = np.asarray(H_amp_asym.hamiltonian(0.3))
check("amp_red alone stays Hermitian",
      np.allclose(H_amp_t, H_amp_t.conj().T, atol=1e-10))

print("== negative amplitude == individual-beam pi phase flip (pinned regression) ==")
tones_neg = ms_tones(nu, delta, -0.8, theta=0.0, psi=0.0)
tones_flip = ms_tones(nu, delta, 0.8, theta=np.pi, psi=0.0)
H_neg = driven_spins(tones_neg, ["q0", "q1"], [mode], lamb_dicke=1)
H_flip = driven_spins(tones_flip, ["q0", "q1"], [mode], lamb_dicke=1)
check("negative amplitude == positive amplitude + pi spin phase",
      np.allclose(np.asarray(H_neg.hamiltonian(0.37)), np.asarray(H_flip.hamiltonian(0.37)), atol=1e-10))

print("== multi-mode (Monroe Eq. 20/26/27) ==")
nu1, nu2 = 40.0, 33.0
mode0 = Mode(nu=nu1, eta=eta * np.array(b), n_max=n_max, name="mode0")
mode1 = Mode(nu=nu2, eta=0.07 * np.array(b), n_max=5, name="mode1")
tones_mm = ms_tones(nu1, delta, Omega, theta=0.0, psi=0.0)
H_mm = driven_spins(tones_mm, ["q0", "q1"], [mode0, mode1], lamb_dicke=1)
check("carrier emitted once per ion, sdf per ion+mode",
      {"carrier_q0_tone0", "carrier_q0_tone1", "sdf_q0_mode0_tone0", "sdf_q0_mode1_tone0"} <= set(H_mm.groups))
gate_mm = ms_closed_form(amplitudes=Omega, phases=phases,
                         modes=[MSMode(eta * np.array(b), delta, n_max, "mode0"),
                                MSMode(0.07 * np.array(b), nu1 + delta - nu2, 5, "mode1")])
al = gate_mm.alpha(0.5 * T)
check("alpha has shape (n_ions, n_modes)", al.shape == (2, 2))

print("== refusals (never silently wrong) ==")
try:
    driven_spins(tones, ["q0", "q1"], [mode], lamb_dicke=3)
    check("bad lamb_dicke rejected", False)
except ValueError as e:
    check("lamb_dicke not in {1,2,None} rejected", "lamb_dicke" in str(e))
try:
    driven_spins(tones, ["q0", "q1"], [mode], lamb_dicke=2, rwa=True)
    check("rwa=True + lamb_dicke=2 rejected", False)
except ValueError as e:
    check("rwa=True is only defined at lamb_dicke=1", "lamb_dicke=1" in str(e))

print("== phase-space plotting ==")
ax = plot_phase_space(gate.alpha_trajectory(ts))
check("plot_phase_space on closed-form alpha", ax is not None)
Sphi = sigma_y
_, evecs = np.linalg.eigh(Sphi)
plus = evecs[:, 1]
psi_branch = otimes(plus, plus, fock(0, n_max))
with quiet():
    ev_b = HamiltonianEvolution(H_rwa, psi_branch)
    a_meas = expectation_alpha(ev_b, np.linspace(0, T, 60))
alpha_2 = gate.alpha_trajectory(np.linspace(0, T, 60))[:, 0, 0] * 2
check("<a> on +branch follows sum of closed-form alphas",
      np.max(np.abs(a_meas - alpha_2)) < 1e-3)

print("== ergonomics helpers ==")
from htdse.submodules.spin_boson import explain, TONE_TABLE
from htdse.submodules.molmer_sorensen import ideal_gate

mp = Mode.from_participation(nu=nu, eta=eta, b=b, n_max=n_max, name="mode")
check("Mode.from_participation matches manual eta*b",
      np.allclose(mp.eta, eta * np.array(b)) and mp.nu == nu and mp.name == "mode")

ex = driven_spins([Tone(0.0)], ["q0"], [mode1q], lamb_dicke=None)
try:
    ex + ex
    check("exact_drive '+' raises a clear error", False)
except TypeError as e:
    check("exact_drive '+' raises a clear error", "sigma_+" in str(e))
try:
    ex.replace(x=1)
    check("exact_drive .replace() raises a clear error", False)
except AttributeError as e:
    check("exact_drive .replace() raises a clear error", "replace" in str(e))

g_ideal = ideal_gate(2, eta, delta, Omega, n_max)
g_manual = ms_closed_form([1.0, 1.0], eta, delta, Omega, [0.0, 0.0], n_max)
check("ideal_gate == ms_closed_form with explicit defaults",
      np.allclose(np.asarray(g_ideal.unitary(0.3 * T)), np.asarray(g_manual.unitary(0.3 * T))))

check("explain() runs and TONE_TABLE is non-empty", len(TONE_TABLE) > 0)
explain()

print(f"\nALL {len(PASS)} SPIN-BOSON/MS CHECKS PASSED")
