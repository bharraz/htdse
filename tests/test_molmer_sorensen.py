"""MS suite verification: `python tests/test_molmer_sorensen.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib
matplotlib.use("Agg")

from htdse import (Operator, UnitaryEvolution, HamiltonianEvolution, quiet,
                   process_fidelity, fidelity, otimes, ket)
from htdse.submodules.harmonic_oscillator import fock
from htdse.submodules.molmer_sorensen import (MSMagnus, ms_lamb_dicke1, ms_lamb_dicke2,
                                              plot_phase_space, expectation_alpha)
from htdse.submodules.spin import pauli_term

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

# common gate parameters: 2 ions, equal participation
b = [1.0, 1.0]
eta, delta, n_max = 0.1, 0.5, 8
Omega = 0.8
T = 2 * np.pi / delta          # one loop
phases = [0.0, 0.0]

print("== Magnus closed form vs ODE-solved RWA spin-dependent force ==")
mag = MSMagnus(b, eta, delta, Omega, phases, n_max)
H_rwa = ms_lamb_dicke1(b, eta, delta, Omega, phases, n_max, rwa=True)
# compare on LOW-FOCK states (see MSMagnus truncation caveat: the closed form
# is the infinite-dimensional result; the ODE propagates the truncated model --
# they only agree away from the Fock edge)
rng = np.random.default_rng(3)
lowf = rng.normal(size=n_max + 1) + 1j * rng.normal(size=n_max + 1)
lowf[3:] = 0.0
lowf /= np.linalg.norm(lowf)
test_states = [otimes(ket("00"), fock(0, n_max)),
               otimes(ket("01"), fock(1, n_max)),
               otimes((ket("00") + ket("11")) / np.sqrt(2), lowf)]
with quiet():
    for tt in [0.3 * T, 0.7 * T, T]:
        U_mag = np.asarray(mag.unitary(tt))
        ev_ode = UnitaryEvolution(H_rwa, dim=mag.dim)
        U_ode = np.asarray(ev_ode.unitary_at(tt))
        worst = min(fidelity(U_mag @ p, U_ode @ p) for p in test_states)
        check(f"Magnus == ODE on low-Fock states at t={tt/T:.1f}T (worst F={worst:.2e})",
              worst > 1 - 1e-7)

print("== analytic ingredients ==")
# alpha(t) analytic for constant Omega: f = c e^{-i d t}, c = -eta b Om/2
c = -eta * 1.0 * Omega / 2
ts = np.linspace(0, T, 400)
alpha_num = mag.alpha_trajectory(ts)[:, 0, 0]      # (n_times, n_ions, n_modes)
alpha_ana = (c / delta) * (np.exp(-1j * delta * ts) - 1)   # -i int c e^{-i d t'}
check("alpha(t) matches analytic circle", np.max(np.abs(alpha_num - alpha_ana)) < 1e-6)
check("loop closes at T = 2*pi/delta", abs(mag.alpha(T)[0, 0]) < 1e-6)
# Theta analytic: Theta_jk(t) = c_j c_k (sin(dt)/d^2 - t/d)
Th = mag.geometric_phase(T)
Th_ana = c * c * (np.sin(delta * T) / delta ** 2 - T / delta)
check("geometric phase matches analytic", abs(Th[0, 1] - Th_ana) < 1e-6 * abs(Th_ana))

print("== ideal gate physics at loop closure ==")
# at t=T: alpha=0, U = exp(i sum Theta_jk S_j S_k); entangling angle = 2*Theta_01
chi = 2 * Th_ana
psi0 = Operator(otimes(ket("00"), fock(0, n_max)))
with quiet():
    psiT = np.asarray(UnitaryEvolution(H_rwa, dim=mag.dim).unitary_at(T)) @ np.asarray(psi0)
# expected: exp(i chi S0 S1)|00,0>; S = sigma_{pi/2} = sigma_y here (phi=0)
from htdse.submodules.spin import sigma_y, I2
S0S1 = np.kron(np.kron(sigma_y, sigma_y), np.eye(n_max + 1))
from scipy.linalg import expm
psi_expected = expm(1j * chi * S0S1) @ (np.exp(1j * 2 * Th[0, 0]) * np.asarray(psi0))
check("gate = geometric-phase gate at closure", abs(np.vdot(psi_expected, psiT)) ** 2 > 1 - 1e-6)

print("== pre-RWA Lamb-Dicke builders ==")
nu = 40.0   # trap frequency well above Omega -> RWA should be decent
H1 = ms_lamb_dicke1(b, eta, delta, Omega, phases, n_max, nu=nu)
H2 = ms_lamb_dicke2(b, eta, delta, Omega, phases, n_max, nu=nu)
check("groups per ion, swappable",
      {"carrier_q0", "sdf_q0", "carrier_q1", "sdf_q1"} <= set(H1.groups)
      and "ld2_q0" in H2.groups)
check("H1(t0) Hermitian", np.max(np.abs(np.asarray(H1.hamiltonian(0.3))
                                        - np.asarray(H1.hamiltonian(0.3)).conj().T)) < 1e-12)
check("H2(t0) Hermitian", np.max(np.abs(np.asarray(H2.hamiltonian(0.3))
                                        - np.asarray(H2.hamiltonian(0.3)).conj().T)) < 1e-12)
with quiet():
    psi_rwa = HamiltonianEvolution(H_rwa, psi0).state_at(T)
    psi_1 = HamiltonianEvolution(H1, psi0, rtol=1e-9, atol=1e-11).state_at(T)
F1 = fidelity(psi_rwa, psi_1)
print(f"    1 - |<rwa|pre-RWA LD1>|^2 = {1-F1:.3e}  (nu >> Omega: carrier error tiny)")
check("pre-RWA LD1 ~ RWA at nu >> Omega", F1 > 0.999)
# make the corrections RESOLVABLE: lower trap frequency, bigger eta
nu_lo, eta_big = 8.0, 0.3
Hr_lo = ms_lamb_dicke1(b, eta_big, delta, Omega, phases, n_max, rwa=True)
H1_lo = ms_lamb_dicke1(b, eta_big, delta, Omega, phases, n_max, nu=nu_lo)
H2_lo = ms_lamb_dicke2(b, eta_big, delta, Omega, phases, n_max, nu=nu_lo)
with quiet():
    p_rwa = HamiltonianEvolution(Hr_lo, psi0).state_at(T)
    p1 = HamiltonianEvolution(H1_lo, psi0, rtol=1e-9, atol=1e-11).state_at(T)
    p2 = HamiltonianEvolution(H2_lo, psi0, rtol=1e-9, atol=1e-11).state_at(T)
F_carrier = fidelity(p_rwa, p1)
F_ld2 = fidelity(p1, p2)
print(f"    nu={nu_lo}, eta={eta_big}: 1-F(rwa,LD1)={1-F_carrier:.3e}  1-F(LD1,LD2)={1-F_ld2:.3e}")
check("carrier/counter-rotating error resolvable at low nu",
      1e-6 < 1 - F_carrier < 0.5)
check("eta^2 terms produce a distinct, small correction",
      1e-9 < 1 - F_ld2 < 0.1)
psi_1 = p1  # reuse low-nu case for the error-injection checks below
H1 = H1_lo

print("== composability: sigma_z error injection + drive swap ==")
H_err = H1 + pauli_term("Z0", coeff=0.05) + pauli_term("Z1", coeff=0.05)
with quiet():
    psi_err = HamiltonianEvolution(H_err, psi0, rtol=1e-9, atol=1e-11).state_at(T)
check("sigma_z error composes and matters", fidelity(psi_1, psi_err) < 1 - 1e-4)
H_swap = H1.replace(carrier_q0=ms_lamb_dicke1([1.0], eta, delta, Omega * 1.1, [0.0],
                                              n_max, nu=nu).group("carrier_q0"))
check("per-ion group swap works", np.isfinite(np.asarray(H_swap.hamiltonian(0.1))).all())

print("== phase-space plotting ==")
ax = plot_phase_space(mag.alpha_trajectory(ts))   # (n_times, n_ions, n_modes)
check("plot_phase_space on Magnus alpha", ax is not None)
# measured <a> on a sigma_Phi eigenstate follows the Magnus alpha
Sphi = sigma_y  # Phi = pi/2 for phi = 0
_, evecs = np.linalg.eigh(Sphi)
plus = evecs[:, 1]  # +1 eigenvector
psi_branch = Operator(otimes(plus, plus, fock(0, n_max)))
with quiet():
    ev_b = HamiltonianEvolution(H_rwa, psi_branch)
    a_meas = expectation_alpha(ev_b, np.linspace(0, T, 60))
alpha_2 = mag.alpha_trajectory(np.linspace(0, T, 60))[:, 0, 0] * 2  # both ions push: alpha_0+alpha_1
check("<a> on +branch follows sum of Magnus alphas",
      np.max(np.abs(a_meas - alpha_2)) < 1e-3)
ax2 = plot_phase_space(a_meas)
check("plot_phase_space on measured <a>", ax2 is not None)

print("== motion phase psi (Monroe Eq. 26/27) ==")
# alpha carries e^{-i psi} (Eq. 26); the entangling phase does NOT (Eq. 27).
# This is the whole basis of motion-phase scheduling: psi steers the phase-space
# trajectory without touching the gate.
psi_val = 0.7
mag_psi = MSMagnus(b, eta, delta, Omega, phases, n_max, motion_phases=psi_val)
a0 = mag.alpha(0.4 * T)                      # psi = 0
a1 = mag_psi.alpha(0.4 * T)                  # psi = 0.7
check("alpha rotates by e^{-i psi}",
      np.allclose(a1, a0 * np.exp(-1j * psi_val), atol=1e-9))
check("|alpha| is psi-independent", np.allclose(np.abs(a1), np.abs(a0), atol=1e-9))
check("entangling angle is psi-INDEPENDENT (the point of psi)",
      np.allclose(mag_psi.geometric_phase(0.4 * T), mag.geometric_phase(0.4 * T),
                  atol=1e-9))
# analytic alpha with psi, constant Omega (Monroe Eq. 26 up to this module's
# overall sign convention -- see the module docstring)
tau = 0.63 * T
a_ana = (eta * Omega * np.exp(-1j * psi_val) / (2 * delta)) * (1 - np.exp(-1j * delta * tau))
check("alpha(psi) matches the Eq.26 closed form",
      abs(mag_psi.alpha(tau)[0, 0] - a_ana) < 1e-6)
# entangling angle closed form: -(1/2) Om^2 sum_m (eta_m^2/d_m)(T - sin(d_m T)/d_m)
ang = mag.entangling_angle(0.63 * T)[0, 1]
ang_ana = -0.5 * Omega ** 2 * (eta ** 2 / delta) * (tau - np.sin(delta * tau) / delta)
check("entangling angle matches the closed form", abs(ang - ang_ana) < 1e-6 * abs(ang_ana))
# psi enters the pre-RWA builder as cos(mu t + psi) -- and its RWA limit must
# reproduce the Magnus gate WITH the same psi
H_rwa_psi = ms_lamb_dicke1(b, eta, delta, Omega, phases, n_max, rwa=True,
                           motion_phases=psi_val)
with quiet():
    U_ode = np.asarray(UnitaryEvolution(H_rwa_psi, dim=mag_psi.dim).unitary_at(0.7 * T))
U_mag = np.asarray(mag_psi.unitary(0.7 * T))
worst = min(fidelity(U_mag @ p, U_ode @ p) for p in test_states)
check(f"psi: Magnus == ODE'd RWA builder (worst F={worst:.2e})", worst > 1 - 1e-7)

print("== multi-mode (Monroe Eq. 20/26/27) ==")
from htdse.submodules.molmer_sorensen import MSMode
nu1, nu2 = 40.0, 33.0
mu = nu1 + delta                       # ONE bichromatic beat drives every mode
m1 = MSMode(eta=eta * np.array(b), detune=mu - nu1, n_max=n_max, nu=nu1, name="mode0")
m2 = MSMode(eta=0.07 * np.array(b), detune=mu - nu2, n_max=5, nu=nu2, name="mode1")

# a spectator mode with zero coupling must reproduce the single-mode gate exactly
m2_off = MSMode(eta=0.0, detune=mu - nu2, n_max=3, nu=nu2, name="mode1")
mag_off = MSMagnus(amplitudes=Omega, phases=phases, modes=[m1, m2_off])
check("uncoupled spectator mode leaves Theta unchanged",
      np.allclose(mag_off.geometric_phase(T), mag.geometric_phase(T), atol=1e-9))
check("uncoupled spectator mode has zero alpha",
      np.allclose(mag_off.alpha(T)[:, 1], 0, atol=1e-12))

mag_mm = MSMagnus(amplitudes=Omega, phases=phases, modes=[m1, m2])
al = mag_mm.alpha(0.5 * T)
check("alpha has shape (n_ions, n_modes)", al.shape == (2, 2))
check("multi-mode dim = 2^N * prod(n_max+1)", mag_mm.dim == 4 * (n_max + 1) * 6)
# Theta SUMS over modes: build each mode alone and add
th_1 = MSMagnus(amplitudes=Omega, phases=phases, modes=[m1]).geometric_phase(0.5 * T)
th_2 = MSMagnus(amplitudes=Omega, phases=phases, modes=[m2]).geometric_phase(0.5 * T)
check("Theta sums over modes (Eq. 27)",
      np.allclose(mag_mm.geometric_phase(0.5 * T), th_1 + th_2, atol=1e-8))
# per-mode alpha matches that mode solved alone
check("alpha per mode matches the single-mode solve",
      np.allclose(al[:, 1:2], MSMagnus(amplitudes=Omega, phases=phases,
                                       modes=[m2]).alpha(0.5 * T), atol=1e-9))

# multi-mode pre-RWA builder: one carrier per ion (NOT one per mode), and its
# RWA limit reproduces the multi-mode Magnus gate
H_mm = ms_lamb_dicke1(amplitudes=Omega, phases=phases, modes=[m1, m2])
check("carrier emitted once per ion, sdf per ion+mode",
      {"carrier_q0", "carrier_q1", "sdf_q0_mode0", "sdf_q0_mode1",
       "sdf_q1_mode0", "sdf_q1_mode1"} == set(H_mm.groups))
H_mm_rwa = ms_lamb_dicke1(amplitudes=Omega, phases=phases, modes=[m1, m2], rwa=True)
lowf2 = otimes(ket("00"), fock(0, n_max), fock(0, 5))
with quiet():
    U_ode = np.asarray(UnitaryEvolution(H_mm_rwa, dim=mag_mm.dim).unitary_at(0.5 * T))
U_mag = np.asarray(mag_mm.unitary(0.5 * T))
F_mm = fidelity(U_mag @ lowf2, U_ode @ lowf2)
check(f"multi-mode Magnus == ODE'd RWA builder (F={F_mm:.2e})", F_mm > 1 - 1e-7)

print("== refusals (never silently wrong) ==")
try:
    ms_lamb_dicke1(amplitudes=Omega, phases=phases,
                   modes=[m1, MSMode(eta=0.05, detune=1.0, n_max=4, nu=99.0, name="bad")])
    check("inconsistent beat rejected", False)
except ValueError as e:
    check("inconsistent beat (nu_m + delta_m != mu) rejected", "beat" in str(e))
try:
    ms_lamb_dicke2(amplitudes=Omega, phases=phases, modes=[m1, m2])
    check("multi-mode eta^2 rejected", False)
except NotImplementedError as e:
    check("multi-mode eta^2 rejected (cross-mode terms)", "cross-mode" in str(e))
try:
    MSMagnus(b, eta, delta, Omega, phases, n_max, modes=[m1])
    check("both mode forms rejected", False)
except ValueError:
    check("passing both `modes=` and the shorthand is rejected", True)

print("== negative amplitude == individual-beam pi phase flip ==")
# Omega_j(t) is a plain SIGNED real multiplier of sigma_x/sigma_y everywhere
# (carrier, spin-dependent force, eta^2) -- never abs()'d -- so on a
# counter-propagating Raman setup where only the individually-addressing beam
# carries amplitude/phase modulation, a negative sample IS a pi phase flip of
# that beam: Omega*sigma_theta == (-Omega)*sigma_{theta+pi}. Pinned here so
# this can't silently regress; nothing in htdse special-cases the sign.
mag_neg = MSMagnus(b, eta, delta, -0.8, phases=[0.0, 0.0], n_max=n_max)
mag_flip = MSMagnus(b, eta, delta, 0.8, phases=[np.pi, np.pi], n_max=n_max)
check("MSMagnus: negative amplitude == positive amplitude + pi spin phase",
      np.allclose(np.asarray(mag_neg.unitary(0.3 * T)),
                  np.asarray(mag_flip.unitary(0.3 * T)), atol=1e-10))
H_neg = ms_lamb_dicke1(b, eta, delta, -0.8, [0.0, 0.0], n_max, nu=40.0)
H_flip = ms_lamb_dicke1(b, eta, delta, 0.8, [np.pi, np.pi], n_max, nu=40.0)
check("ms_lamb_dicke1: same equivalence holds on the real (pre-RWA) H(t)",
      np.allclose(np.asarray(H_neg.hamiltonian(0.37)),
                  np.asarray(H_flip.hamiltonian(0.37)), atol=1e-10))

print("== asymmetric bichromatic tones (detune_red / amplitude_red) ==")
# detune_red/amplitude_red equal to blue's must reduce EXACTLY to the plain
# symmetric drive -- this is the actual claim the whole feature rests on.
H_ref = ms_lamb_dicke1(b, eta, delta, Omega, [0.3, 0.7], n_max, nu=nu,
                       motion_phases=0.4)
H_asym_eq = ms_lamb_dicke1(b, eta, delta, Omega, [0.3, 0.7], n_max, nu=nu,
                           motion_phases=0.4, detune_red=delta)
for tt in [0.0, 0.6 * T, 1.3 * T]:
    check(f"asymmetric == symmetric when detune_red==detune, t={tt:.2f}",
          np.allclose(np.asarray(H_ref.hamiltonian(tt)),
                      np.asarray(H_asym_eq.hamiltonian(tt)), atol=1e-9))
H2_ref = ms_lamb_dicke2(b, eta, delta, Omega, [0.3, 0.7], n_max, nu=nu)
H2_asym_eq = ms_lamb_dicke2(b, eta, delta, Omega, [0.3, 0.7], n_max, nu=nu,
                            detune_red=delta)
check("order-2 asymmetric == symmetric when detune_red==detune",
      np.allclose(np.asarray(H2_ref.hamiltonian(0.4 * T)),
                  np.asarray(H2_asym_eq.hamiltonian(0.4 * T)), atol=1e-9))

# genuinely asymmetric: still Hermitian, still norm-preserving under a real solve
H_true_asym = ms_lamb_dicke1(b, eta, delta, Omega, [0.0, 0.0], n_max, nu=nu,
                             detune_red=delta * 1.3)
check("asymmetric H(t) stays Hermitian",
      all(np.allclose(np.asarray(H_true_asym.hamiltonian(tt)),
                      np.asarray(H_true_asym.hamiltonian(tt)).conj().T, atol=1e-10)
          for tt in [0.0, 0.5 * T, 1.1 * T]))
with quiet():
    psiT = HamiltonianEvolution(H_true_asym, psi0).state_at(0.8 * T)
check("asymmetric drive: norm preserved under real evolution",
      abs(np.vdot(psiT, psiT) - 1) < 1e-6)

# amplitude_red independently of detune_red
H_amp_asym = ms_lamb_dicke1(b, eta, delta, Omega, [0.0, 0.0], n_max, nu=nu,
                            amplitude_red=Omega * 1.5)
check("amplitude_red alone stays Hermitian",
      np.allclose(np.asarray(H_amp_asym.hamiltonian(0.3 * T)),
                  np.asarray(H_amp_asym.hamiltonian(0.3 * T)).conj().T, atol=1e-10))

try:
    ms_lamb_dicke1(b, eta, delta, Omega, [0.0, 0.0], n_max, rwa=True, detune_red=delta * 1.1)
    check("rwa=True + asymmetric rejected", False)
except ValueError as e:
    check("rwa=True + asymmetric rejected", "asymmetric" in str(e))

print(f"\nALL {len(PASS)} MS CHECKS PASSED")
