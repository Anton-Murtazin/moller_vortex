"""Physical constants and user-configurable calculation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .numerics import Axis


# Change this line to switch the real dtype throughout the package.
REAL_DTYPE = np.float64
COMPLEX_DTYPE = np.result_type(REAL_DTYPE, np.complex64).type

PI = REAL_DTYPE(np.pi)
ALPHA_EM = REAL_DTYPE(1.0 / 137.035999084)
ELECTRON_CHARGE = REAL_DTYPE(np.sqrt(4.0 * PI * ALPHA_EM))
ELECTRON_MASS = REAL_DTYPE(0.51099895000)  # MeV
HBARC_MEV_NM = REAL_DTYPE(1.973269804e-4)  # MeV nm


@dataclass(frozen=True)
class NormalizationGrid:
    """Two-dimensional finite domain for the normalization integral."""

    k_perp: Axis
    k_z: Axis


@dataclass(frozen=True)
class ProbabilityGrid:
    """Axes for differential probability and the mean total momentum."""

    k3_perp: Axis
    k3_phi: Axis
    total_k_perp: Axis | None = None
    total_k_phi: Axis | None = None


def spatial_width_nm_to_momentum_mev(width_nm: float) -> np.floating:
    """Convert a Gaussian coordinate width in nm to momentum width in MeV."""
    return HBARC_MEV_NM / width_nm
