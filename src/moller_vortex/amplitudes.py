"""Plane-wave Moller amplitudes in paraxial approximations."""

from __future__ import annotations

import numpy as np

from .constants import ELECTRON_CHARGE, ELECTRON_MASS
from .kinematics import energy, helicity, kron_delta, vec3


def moller_massive_ab(
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


def moller_paraxial_massive_t_scale(
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
    """Return the t-channel massive-paraxial numerator relative to the old one.

    This helper returns the spinor numerator ratio

        [(A + 4B) direct - 8B exchange] / [8 sqrt(E1 E2 E3 E4)].

    It does not include any denominator.  The packet S-matrix uses the same
    algebra with the invariant denominator inside the exact-time integrand for
    ``matrix_element="paraxial_massive"``.
    """
    direct = _helicity_direct(lam1, lam2, lam3, lam4)
    exchange = _helicity_exchange(lam1, lam2, lam3, lam4)
    if direct == 0 and exchange == 0:
        return 0.0

    A, B = moller_massive_ab(E1, E2, E3, E4, lam1, lam2, lam3, lam4, m=m)
    numerator = (A + 4.0 * B) * direct - 8.0 * B * exchange
    old_numerator = 8.0 * np.sqrt(E1 * E2 * E3 * E4)
    return numerator / old_numerator


def _minkowski_transfer_sq(k_a, k_b, m: float) -> float:
    """Return ``(p_a-p_b)^2`` for on-shell three-momenta."""
    k_a = vec3(k_a)
    k_b = vec3(k_b)
    dE = energy(k_a, m) - energy(k_b, m)
    dk = k_a - k_b
    return dE * dE - np.dot(dk, dk)


def moller_amplitude_paraxial_massive_t(
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
    """Compute the t-channel-dominated massive paraxial Moller amplitude."""
    k1 = vec3(k1)
    k2 = vec3(k2)
    k3 = vec3(k3)
    k4 = vec3(k4)

    t = _minkowski_transfer_sq(k1, k3, m)
    if t == 0.0:
        raise ZeroDivisionError("The t-channel Moller denominator is zero.")

    E1 = energy(k1, m)
    E2 = energy(k2, m)
    E3 = energy(k3, m)
    E4 = energy(k4, m)
    A, B = moller_massive_ab(E1, E2, E3, E4, lam1, lam2, lam3, lam4, m=m)
    direct = _helicity_direct(lam1, lam2, lam3, lam4)
    exchange = _helicity_exchange(lam1, lam2, lam3, lam4)
    numerator = (A + 4.0 * B) * direct - 8.0 * B * exchange
    return e_charge ** 2 * numerator / t


def moller_amplitude_paraxial_massive(
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
    """Compute the rearranged massive paraxial Moller amplitude with t and u."""
    k1 = vec3(k1)
    k2 = vec3(k2)
    k3 = vec3(k3)
    k4 = vec3(k4)

    t = _minkowski_transfer_sq(k1, k3, m)
    u = _minkowski_transfer_sq(k1, k4, m)
    if t == 0.0 or u == 0.0:
        raise ZeroDivisionError("A Moller denominator is zero.")

    E1 = energy(k1, m)
    E2 = energy(k2, m)
    E3 = energy(k3, m)
    E4 = energy(k4, m)

    A_t, B_t = moller_massive_ab(E1, E2, E3, E4, lam1, lam2, lam3, lam4, m=m)
    A_u, B_u = moller_massive_ab(E1, E2, E4, E3, lam1, lam2, lam4, lam3, m=m)

    direct = _helicity_direct(lam1, lam2, lam3, lam4)
    exchange = _helicity_exchange(lam1, lam2, lam3, lam4)
    direct_coefficient = (A_t + 4.0 * B_t) / t + 8.0 * B_u / u
    exchange_coefficient = 8.0 * B_t / t + (A_u + 4.0 * B_u) / u
    return e_charge ** 2 * (
        direct_coefficient * direct - exchange_coefficient * exchange
    )


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
    """Compute the ultrarelativistic paraxial Moller amplitude.

    Parameters
    ----------
    k1, k2:
        Incoming plane-wave momenta.
    k3, k4:
        Outgoing plane-wave momenta.
    lam1, lam2, lam3, lam4:
        Helicity labels.
    m:
        Electron mass.
    e_charge:
        Electric charge.

    Returns
    -------
    complex
        Plane-wave Moller amplitude in the impulse approximation. Non-
        helicity-conserving channels return zero.
    """
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
