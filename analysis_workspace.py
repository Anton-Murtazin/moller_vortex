"""Working file for exploratory calculations.

Run this file from the project root, or open it in VS Code and execute cells with
Python Interactive mode. It imports the whole project namespace for convenient
research work.
"""

import numpy as np
import moller_vortex as mv
from moller_vortex import *


# %% Basic setup
packet1 = mv.LGPacket(ell=1, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
packet2 = mv.LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)

N1 = mv.normalization_constant(packet1)
N2 = mv.normalization_constant(packet2)

k3 = np.array([0.80, 0.10, 19.70])
k4 = np.array([-0.55, -0.08, -19.60])
impact_b = np.array([0.30, 0.00])

S = mv.S_impulse_closed_form(
    k3,
    k4,
    packet1,
    packet2,
    lam1=0.5,
    lam2=0.5,
    lam3=0.5,
    lam4=0.5,
    impact_b=impact_b,
    N1=N1,
    N2=N2,
)

print("N1 =", N1)
print("N2 =", N2)
print("S  =", S)


# %% Example scan over ell1
ell_values = range(-3, 4)
scan = []

for ell in ell_values:
    p1 = mv.LGPacket(
        ell=ell,
        sigma_perp=packet1.sigma_perp,
        sigma_par=packet1.sigma_par,
        kbar_z=packet1.kbar_z,
    )
    Np1 = mv.normalization_constant(p1)
    S_ell = mv.S_impulse_closed_form(
        k3,
        k4,
        p1,
        packet2,
        lam1=0.5,
        lam2=0.5,
        lam3=0.5,
        lam4=0.5,
        impact_b=impact_b,
        N1=Np1,
        N2=N2,
    )
    scan.append((ell, S_ell, abs(S_ell)))

for ell, S_ell, abs_S in scan:
    print(f"ell1={ell:2d}  S={S_ell}  |S|={abs_S:.6e}")
