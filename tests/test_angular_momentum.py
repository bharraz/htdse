"""Angular momentum suite verification: `python tests/test_angular_momentum.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htdse.submodules.angular_momentum import clebsch_gordan, wigner_6j

PASS = []
def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name); print(f"  ok: {name}")

print("== Clebsch-Gordan: known textbook values ==")
# two spin-1/2's coupling to a triplet/singlet
check("CG(1/2,1/2; 1/2,-1/2 | 1,0) = 1/sqrt(2)",
      np.isclose(clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0), 1 / np.sqrt(2)))
check("CG(1/2,1/2; 1/2,1/2 | 1,1) = 1",
      np.isclose(clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 1), 1.0))
check("CG(1/2,1/2; 1/2,-1/2 | 0,0) = 1/sqrt(2)",
      np.isclose(clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0, 0), 1 / np.sqrt(2)))
# spin-1 + spin-1/2 -> j=3/2
check("CG(1,1; 1/2,-1/2 | 3/2,1/2) = 1/sqrt(3)",
      np.isclose(clebsch_gordan(1, 1, 0.5, -0.5, 1.5, 0.5), 1 / np.sqrt(3)))
check("CG(1,0; 1/2,1/2 | 3/2,1/2) = sqrt(2/3)",
      np.isclose(clebsch_gordan(1, 0, 0.5, 0.5, 1.5, 0.5), np.sqrt(2 / 3)))

print("== Clebsch-Gordan: forbidden inputs return 0 ==")
check("m1+m2 != m3 -> 0", clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 0) == 0.0)
check("triangle inequality violated -> 0", clebsch_gordan(0.5, 0.5, 0.5, -0.5, 3, 0) == 0.0)

print("== Clebsch-Gordan: malformed inputs raise ==")
raised = False
try:
    clebsch_gordan(0.5, 1.0, 0.5, -0.5, 1, 0)  # |m1| > j1
except ValueError:
    raised = True
check("raises ValueError for |m| > j", raised)

raised = False
try:
    clebsch_gordan(0.3, 0.0, 0.5, -0.5, 1, 0)  # non-half-integer j
except ValueError:
    raised = True
check("raises ValueError for non-half-integer j", raised)

print("== Clebsch-Gordan: orthogonality (sum over m1 of CG^2 == 1) ==")
for (j1, j2, j3, m3) in [(0.5, 0.5, 1, 0), (1, 0.5, 1.5, 0.5), (1, 1, 2, 1), (1.5, 1, 2.5, -0.5)]:
    m1s = np.arange(-j1, j1 + 0.5, 1.0)
    total = sum(clebsch_gordan(j1, m1, j2, m3 - m1, j3, m3) ** 2
                for m1 in m1s if abs(m3 - m1) <= j2 + 1e-8)
    check(f"orthogonality j1={j1},j2={j2},j3={j3},m3={m3}: sum CG^2 = {total:.6f} ~ 1",
          np.isclose(total, 1.0, atol=1e-8))

print("== Wigner 6j: known textbook values (cross-checked against sympy.physics.wigner) ==")
check("6j{1,1,1;1,1,1} = 1/6", np.isclose(wigner_6j(1, 1, 1, 1, 1, 1), 1 / 6))
check("6j{1/2,1/2,1;1/2,1/2,1} = 1/6", np.isclose(wigner_6j(0.5, 0.5, 1, 0.5, 0.5, 1), 1 / 6))
check("6j{1,1,0;1,1,1} = -1/3", np.isclose(wigner_6j(1, 1, 0, 1, 1, 1), -1 / 3))
check("6j{1/2,1/2,0;1/2,1/2,1} = 1/2",
      np.isclose(wigner_6j(0.5, 0.5, 0, 0.5, 0.5, 1), 0.5))

print("== Wigner 6j: forbidden triads return 0 ==")
check("triangle violated in (j1,j2,j3) -> 0", wigner_6j(1, 1, 5, 1, 1, 1) == 0.0)

print("== Wigner 6j: malformed inputs raise ==")
raised = False
try:
    wigner_6j(-1, 1, 1, 1, 1, 1)
except ValueError:
    raised = True
check("raises ValueError for negative j", raised)

print("== Wigner 6j: symmetry relations ==")
base = wigner_6j(1, 1, 2, 1, 1, 1)
colswap13 = wigner_6j(2, 1, 1, 1, 1, 1)   # swap columns 1 and 3
rowswap23 = wigner_6j(1, 1, 1, 1, 1, 2)   # swap rows within columns 2 and 3
check("6j invariant under column permutation", np.isclose(base, colswap13))
check("6j invariant under swapping two columns' rows", np.isclose(base, rowswap23))

print(f"\n{len(PASS)} checks passed.")
