"""Numerical constants and unit convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FLOAT_DTYPE = np.float64
COMPLEX_DTYPE = np.complex128

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

PI = np.pi

# Natural units: hbar = c = 1.
# Energies, momenta and masses are measured in MeV.
# Lengths and impact parameters are measured in MeV^{-1}.
ALPHA_EM = 1.0 / 137.035999084
ELECTRON_CHARGE = np.sqrt(4.0 * PI * ALPHA_EM)
ELECTRON_MASS = 0.51099895000

HBARC_MEV_NM = 1.97463e-4 #MeV * nm
NM_TO_MEV_INV = 1.0 / HBARC_MEV_NM 

def spatial_width_nm_to_momentum_mev(width_nm: float) -> float:
    return HBARC_MEV_NM / width_nm
