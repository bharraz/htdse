"""Trap (sideband/thermal) suite verification: `python tests/test_trap.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htdse.submodules.trap import (lamb_dicke, sideband, sideband_series,
                                   thermal_sideband, thermal_population_series)
from htdse.submodules.harmonic_oscillator import annihilation, creation

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

print("== lamb_dicke: kwarg exclusivity ==")
for kwargs in [{}, dict(fm=0.1), dict(omega_m=0.1), dict(kp=0.1), dict(lambda_p=0.1), dict(nu_p=0.1),
               dict(fm=0.1, omega_m=0.1, kp=0.1), dict(fm=0.1, kp=0.1, lambda_p=0.1)]:
    raised = False
    try:
        lamb_dicke(0.1, **kwargs)
    except ValueError:
        raised = True
    check(f"lamb_dicke raises for kwargs={kwargs}", raised)

print("== lamb_dicke: AMO.jl regression fixture ==")
m, lam, f = 2.84e-25, 435e-9, 124e3
eta_expected = 0.22297375295764688
for freq_kw in (dict(fm=f), dict(omega_m=2 * np.pi * f)):
    for k_kw in (dict(kp=2 * np.pi / lam), dict(lambda_p=lam), dict(nu_p=299792458 / lam)):
        eta = lamb_dicke(m, **freq_kw, **k_kw)
        check(f"lamb_dicke({freq_kw}, {k_kw}) = {eta_expected}", np.isclose(eta, eta_expected))

print("== sideband: eta=0 and negative-index edge cases ==")
for n1 in range(5):
    for n2 in range(5):
        check(f"sideband({n1},{n2},0) == delta", sideband(n1, n2, 0) == complex(n1 == n2))
check("sideband(0,-1,0.2) == 0", sideband(0, -1, 0.2) == 0.0)
check("sideband(-1,0,0.3) == 0", sideband(-1, 0, 0.3) == 0.0)

print("== sideband: cross-check against dense matrix exponential of i*eta*(a+adag) ==")
def exp_eta(n_dim, eta):
    a = annihilation(n_dim - 1)
    adag = creation(n_dim - 1)
    from scipy.linalg import expm
    return expm(1j * eta * (a + adag))

for eta in (0.1, 0.5, 1.2, 2.5):
    n_max_test = 25
    M = exp_eta(n_max_test * 2 + 10, eta)
    max_err = 0.0
    for n1 in range(n_max_test + 1):
        for n2 in range(n_max_test + 1):
            ele = M[n1, n2]
            got = sideband(n1, n2, eta)
            max_err = max(max_err, abs(got - ele))
            got_mag = sideband(n1, n2, eta, phase=False)
            max_err = max(max_err, abs(abs(got_mag) - abs(ele)))
    check(f"sideband matches matrix-exp ground truth, eta={eta}: max_err={max_err:.2e}",
          max_err < 1e-9)

print("== sideband_series matches sideband term-by-term ==")
for delta_n in (-3, -1, 0, 2, 4):
    eta = 0.4
    series_vals = []
    it = sideband_series(delta_n, eta, n_max=10)
    for v in it:
        series_vals.append(v)
    direct_vals = [sideband(n, n + abs(delta_n), eta) for n in range(11)]
    check(f"sideband_series(delta_n={delta_n}) matches direct calls",
          np.allclose(series_vals, direct_vals))

print("== thermal_sideband: cross-check against brute-force nested sum ==")
def brute_force_thermal_sideband(nbars, delta_ns, etas, t, n_terms=60):
    nbars, delta_ns, etas = np.atleast_1d(nbars), np.atleast_1d(delta_ns), np.atleast_1d(etas)
    n_modes = len(nbars)
    total = 0.0
    for combo in np.ndindex(*([n_terms] * n_modes)):
        weight = 1.0
        omega = 1.0
        for nbar, delta_n, eta, n in zip(nbars, delta_ns, etas, combo):
            p0 = 1.0 / (nbar + 1.0) if nbar > 0 else (1.0 if n == 0 else 0.0)
            alpha = nbar / (nbar + 1.0) if nbar > 0 else 0.0
            weight *= p0 * alpha ** n
            omega *= sideband(n, n + delta_n, eta, phase=False)
        total += weight * np.sin(t * omega) ** 2
    return total

for nbars, delta_ns, etas in [((2.0,), (1,), (0.2,)),
                              ((1.2,), (-1,), (0.1,)),
                              ((1.2, 1.0), (1, -1), (0.1, 0.5))]:
    for t in (0.5, 1.7, 3.1):
        mine = thermal_sideband(nbars, delta_ns, etas, t)
        ref = brute_force_thermal_sideband(nbars, delta_ns, etas, t)
        check(f"thermal_sideband(nbars={nbars},delta_ns={delta_ns},etas={etas},t={t}) "
              f"~ brute force ({mine:.6f} vs {ref:.6f})", np.isclose(mine, ref, atol=1e-3))

print("== thermal_sideband: detailed-balance identity ==")
nbar, eta, t = 0.7, 0.15, 2.3
for delta_n in (1, 2, 3):
    blue = thermal_sideband(nbar, delta_n, eta, t)
    red = thermal_sideband(nbar, -delta_n, eta, t)
    expected_red = blue * (nbar / (nbar + 1)) ** delta_n
    check(f"thermal_sideband(nbar={nbar}, -{delta_n}) = "
          f"thermal_sideband(nbar, {delta_n}) * (nbar/(nbar+1))^{delta_n}",
          np.isclose(red, expected_red, atol=1e-3))

print("== thermal_population_series: raw formula, NOT harmonic_oscillator.thermal ==")
for nbar in (0.0, 0.5, 3.0, 10.0):
    n = np.arange(6)
    got = thermal_population_series(nbar, 5)
    true_raw = nbar ** n / (1 + nbar) ** (n + 1)
    check(f"thermal_population_series(nbar={nbar}) matches raw Bose-Einstein formula "
          f"(not the truncated/renormalized harmonic_oscillator.thermal)",
          np.allclose(got, true_raw))
check("thermal_population_series sums close to 1 when n_max >> nbar (untruncated regime)",
      np.isclose(thermal_population_series(1.0, 40).sum(), 1.0, atol=1e-10))

print("== sideband: rejects non-integer Fock indices ==")
raised = False
try:
    sideband(3.5, 5.5, 0.3)
except ValueError:
    raised = True
check("sideband raises ValueError for non-integer n1/n2", raised)
check("sideband still accepts numpy integer types",
      np.isclose(sideband(np.int64(2), np.int64(3), 0.3, phase=False),
                sideband(2, 3, 0.3, phase=False)))

print(f"\n{len(PASS)} checks passed.")
