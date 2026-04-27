"""Plane-wave Møller amplitudes in impulse approximation."""

from __future__ import annotations

import numpy as np

from .constants import ELECTRON_CHARGE, ELECTRON_MASS
from .kinematics import energy, helicity, kron_delta, vec3


def moller_amplitude_impulse(
    k1,
    k2,
    k3,
    k4,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
) -> complex:
    """Ultrarelativistic paraxial impulse approximation to Møller scattering."""
    k1 = vec3(k1)
    k2 = vec3(k2)
    k3 = vec3(k3)
    k4 = vec3(k4)

    helicity_conserving = kron_delta(helicity(lam3), helicity(lam1)) and kron_delta(
        helicity(lam4), helicity(lam2)
    )
    if not helicity_conserving:
        return 0.0 + 0.0j

    q_perp = k3[:2] - k1[:2]
    q_perp_sq = np.dot(q_perp, q_perp)
    if q_perp_sq == 0.0:
        raise ZeroDivisionError("Impulse transverse denominator is exactly zero.")

    E1 = energy(k1, m)
    E2 = energy(k2, m)
    E3 = energy(k3, m)
    E4 = energy(k4, m)

    return -8.0 * e_charge ** 2 * np.sqrt(E1 * E2 * E3 * E4) / q_perp_sq
