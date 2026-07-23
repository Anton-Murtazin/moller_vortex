"""Analytic ultrarelativistic S-impulse."""

from __future__ import annotations

import numpy as np

from .constants import ELECTRON_CHARGE, ELECTRON_MASS
from .kinematics import vec2, vec3
from .packets import LGPacket
from .smatrix_common import _impulse_prefactor, _impulse_prefactor_grid
from .transverse import transverse_integral, transverse_integral_grid

def S_impulse(
    k3,
    k4,
    packet1: LGPacket,
    packet2: LGPacket,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    impact_b=(0.0, 0.0),
    N1: float | None = None,
    N2: float | None = None,
    return_details: bool = False,
) -> complex | tuple[complex, dict]:
    """Compute the impulse S-matrix with analytic transverse integral.

    Parameters
    ----------
    k3, k4:
        Final three-momenta.
    packet1, packet2:
        Incoming wave packets.
    lam1, lam2, lam3, lam4:
        Helicity labels.
    m, e_charge:
        Particle mass and electric charge.
    impact_b:
        Transverse impact parameter.
    N1, N2:
        Optional precomputed packet normalizations.
    return_details:
        If True, return ``(S, details)``.

    Returns
    -------
    complex or tuple[complex, dict]
        S-matrix value, optionally with diagnostic details.
    """
    common_factor, details = _impulse_prefactor(
        k3,
        k4,
        packet1,
        packet2,
        lam1,
        lam2,
        lam3,
        lam4,
        m=m,
        e_charge=e_charge,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
    )

    if "reason" in details:
        return (common_factor, details) if return_details else common_factor

    k3_perp = vec3(k3)[:2]
    b = vec2(impact_b)

    Iperp = transverse_integral(
        packet1.ell,
        packet2.ell,
        k3_perp,
        details["K_perp"],
        b,
        details["alpha"],
        details["beta"],
        details["gamma"],
    )

    S = common_factor * Iperp

    if return_details:
        details.update(dict(Iperp=Iperp))
        return S, details

    return S


def S_impulse_grid(
    k3x,
    k3y,
    k3z,
    k4x,
    k4y,
    k4z,
    packet1: LGPacket,
    packet2: LGPacket,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    impact_b=(0.0, 0.0),
    N1: float | None = None,
    N2: float | None = None,
):
    """Vectorized impulse S matrix for broadcastable momenta."""
    common_factor, details = _impulse_prefactor_grid(
        k3x,
        k3y,
        k3z,
        k4x,
        k4y,
        k4z,
        packet1,
        packet2,
        lam1,
        lam2,
        lam3,
        lam4,
        m=m,
        e_charge=e_charge,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
    )
    if "reason" in details:
        return common_factor

    Iperp = transverse_integral_grid(
        packet1.ell,
        packet2.ell,
        k3x,
        k3y,
        details["K_perp"],
        details["impact_b"],
        details["alpha"],
        details["beta"],
        details["gamma"],
    )
    return common_factor * Iperp
