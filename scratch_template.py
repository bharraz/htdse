"""Copy this file to start a new idea. Four sections: KNOBS, BUILD, RUN, READ.
Delete what you don't need; the point is not re-deriving the import list and
the "how do I even look at the answer" boilerplate every time.

See GUIDE.md for the reference this is built from.
"""
import numpy as np
import htdse as ht
from htdse.submodules.spin import sigma_x, sigma_y, sigma_z, sigma_plus

# ---------------------------------------------------------------------------
# KNOBS
# ---------------------------------------------------------------------------
w0 = 1.0            # bare splitting
Omega = 0.2          # drive amplitude
T = np.pi / Omega    # e.g. a pi-pulse time

# ---------------------------------------------------------------------------
# BUILD -- a Model from named terms (see GUIDE.md Step 1, "compose the target")
# ---------------------------------------------------------------------------
H = (ht.term(0.5 * w0 * sigma_z, on="q", name="atom")
   + ht.term(0.5 * Omega * sigma_x, on="q", name="drive"))

ht.show(H)   # sanity check before running anything: is this the H you meant?

# ---------------------------------------------------------------------------
# RUN -- pick the class matching your equation of motion (see GUIDE.md Step 3,
# "evolve"): HamiltonianEvolution / UnitaryEvolution /
# DensityMatrixEvolution / LindbladEvolution
# ---------------------------------------------------------------------------
psi0 = ht.ket("0")
ts = np.linspace(0, T, 200)
with ht.quiet():
    ev = ht.HamiltonianEvolution(H, psi0)
    psis = ev.state_at(ts)

# ---------------------------------------------------------------------------
# READ -- amplitudes, populations, and (if the answer is a gate/propagator,
# not just a state) the effective Hamiltonian it implements
# ---------------------------------------------------------------------------
amp_1 = ht.bra("1") @ psis[-1]
print(f"<1|psi(T)> = {amp_1:.4f}   population = {abs(amp_1)**2:.4f}")

ht.plot_populations(ts, psis)

# For a gate/propagator instead of one state -- project out a subsystem you
# expect to return to its start, check it actually did, then read off the
# effective Hamiltonian on what's left:
#
#   U = ht.UnitaryEvolution(H, dim=...).unitary_at(T)
#   M = ht.project(U, H.subsystems, on="mode", state=0)
#   assert ht.closure(M) > 0.99                    # motion actually returned?
#   H_eff = ht.generator(M, T)
#   print(ht.paulis(H_eff))
#   print("trust this at:", ht.max_eigenphase(H_eff, T), "(want << 1)")
