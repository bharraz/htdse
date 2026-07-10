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
alpha_num = mag.alpha_trajectory(ts)[:, 0]
alpha_ana = (c / delta) * (np.exp(-1j * delta * ts) - 1)   # -i int c e^{-i d t'}
check("alpha(t) matches analytic circle", np.max(np.abs(alpha_num - alpha_ana)) < 1e-6)
check("loop closes at T = 2*pi/delta", abs(mag.alpha(T)[0]) < 1e-6)
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
ax = plot_phase_space(mag.alpha_trajectory(ts), labels=["ion 0", "ion 1"])
check("plot_phase_space on Magnus alpha", ax is not None)
# measured <a> on a sigma_Phi eigenstate follows the Magnus alpha
Sphi = sigma_y  # Phi = pi/2 for phi = 0
_, evecs = np.linalg.eigh(Sphi)
plus = evecs[:, 1]  # +1 eigenvector
psi_branch = Operator(otimes(plus, plus, fock(0, n_max)))
with quiet():
    ev_b = HamiltonianEvolution(H_rwa, psi_branch)
    a_meas = expectation_alpha(ev_b, np.linspace(0, T, 60))
alpha_2 = mag.alpha_trajectory(np.linspace(0, T, 60))[:, 0] * 2  # both ions push: alpha_0+alpha_1
check("<a> on +branch follows sum of Magnus alphas",
      np.max(np.abs(a_meas - alpha_2)) < 1e-3)
ax2 = plot_phase_space(a_meas)
check("plot_phase_space on measured <a>", ax2 is not None)

print(f"\nALL {len(PASS)} MS CHECKS PASSED")
