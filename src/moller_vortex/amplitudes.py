"""Massive spinor factors used by S_time."""

from __future__ import annotations

import numpy as np

from .constants import ELECTRON_MASS
from .kinematics import helicity, kron_delta


def massive_ab(
    E1,
    E2,
    E3,
    E4,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
):
    """Return the massive paraxial A and B spinor numerators.

    The expression keeps the electron mass in the spinor products and matches
    the A/B definitions used in the rearranged Moller matrix element.
    Energies may be scalars or broadcastable NumPy arrays.
    """
    lam1 = helicity(lam1)
    lam2 = helicity(lam2)
    lam3 = helicity(lam3)
    lam4 = helicity(lam4)

    E1 = np.asarray(E1)
    E2 = np.asarray(E2)
    E3 = np.asarray(E3)
    E4 = np.asarray(E4)

    E1p = E1 + m
    E2p = E2 + m
    E3p = E3 + m
    E4p = E4 + m
    E1m = E1 - m
    E2m = E2 - m
    E3m = E3 - m
    E4m = E4 - m

    A = (
        np.sqrt(E1p * E2p * E3p * E4p)
        + 4.0 * lam2 * lam4 * np.sqrt(E1p * E2m * E3p * E4m)
        + 4.0 * lam1 * lam3 * np.sqrt(E1m * E2p * E3m * E4p)
        + 16.0
        * lam1
        * lam2
        * lam3
        * lam4
        * np.sqrt(E1m * E2m * E3m * E4m)
    )
    B = (
        lam3 * lam4 * np.sqrt(E1p * E2p * E3m * E4m)
        + lam2 * lam3 * np.sqrt(E1p * E2m * E3m * E4p)
        + lam1 * lam4 * np.sqrt(E1m * E2p * E3p * E4m)
        + lam1 * lam2 * np.sqrt(E1m * E2m * E3p * E4p)
    )
    return A, B


def _helicity_direct(lam1: float, lam2: float, lam3: float, lam4: float) -> int:
    """Return the direct paraxial spinor-overlap Kronecker product."""
    return kron_delta(helicity(lam3), helicity(lam1)) * kron_delta(
        helicity(lam4), helicity(lam2)
    )


def _helicity_exchange(lam1: float, lam2: float, lam3: float, lam4: float) -> float:
    """Return the exchange paraxial spinor-overlap factor."""
    lam1 = helicity(lam1)
    lam2 = helicity(lam2)
    lam3 = helicity(lam3)
    lam4 = helicity(lam4)
    return (
        (2.0 * lam2)
        * (2.0 * lam4)
        * kron_delta(lam3, -lam2)
        * kron_delta(lam4, -lam1)
    )


def massive_spinor_scale(
    E1,
    E2,
    E3,
    E4,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
):
    """Return the massive t-channel numerator relative to the ultrarelativistic one.

    This helper returns the spinor numerator ratio

        [(A + 4B) direct - 8B exchange] / [8 sqrt(E1 E2 E3 E4)].

    It does not include any denominator.  The packet S-matrix uses the same
    algebra with the invariant denominator inside the massive S_time integrand.
    """
    direct = _helicity_direct(lam1, lam2, lam3, lam4)
    exchange = _helicity_exchange(lam1, lam2, lam3, lam4)
    if direct == 0 and exchange == 0:
        return 0.0

    A, B = massive_ab(E1, E2, E3, E4, lam1, lam2, lam3, lam4, m=m)
    numerator = (A + 4.0 * B) * direct - 8.0 * B * exchange
    old_numerator = 8.0 * np.sqrt(E1 * E2 * E3 * E4)
    return numerator / old_numerator
