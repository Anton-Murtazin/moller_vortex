"""Impulse S-matrix assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import ive

from .amplitudes import moller_paraxial_massive_t_scale
from .constants import (
    COMPLEX_DTYPE,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    FLOAT_DTYPE,
    PI,
    complex_zero,
    real_array,
)
from .kinematics import energy, helicity, kron_delta, vec2, vec3
from .packets import LGPacket, central_energy, resolve_normalizations
from .transverse import (
    transverse_integral_explicit,
    transverse_integral_explicit_grid,
)
from .quadrature import ExactTimeQuadrature, exact_time_nodes, nodes_and_weights


def _helicity_conserving(lam1: float, lam2: float, lam3: float, lam4: float) -> bool:
    """Return whether the helicity channel is conserved.

    Parameters
    ----------
    lam1, lam2, lam3, lam4:
        Incoming and outgoing helicity labels.

    Returns
    -------
    bool
        True when ``lam3 == lam1`` and ``lam4 == lam2``.
    """
    return bool(
        kron_delta(helicity(lam3), helicity(lam1))
        and kron_delta(helicity(lam4), helicity(lam2))
    )


def _packet_denominator(packet1: LGPacket, packet2: LGPacket) -> float:
    """Return sigma and factorial factors from the two LG packets.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.

    Returns
    -------
    float
        Product of transverse-width powers and factorial square root.
    """
    L1 = abs(packet1.ell)
    L2 = abs(packet2.ell)
    log_denominator = (
        L1 * math.log(packet1.sigma_perp)
        + L2 * math.log(packet2.sigma_perp)
        + 0.5 * (math.lgamma(L1 + 1) + math.lgamma(L2 + 1))
    )
    return math.exp(log_denominator)


def _mode(value: str, allowed: tuple[str, ...], name: str) -> str:
    """Validate a public string selector.

    Parameters
    ----------
    value:
        User-provided selector.
    allowed:
        Allowed selector values.
    name:
        Selector name for error messages.

    Returns
    -------
    str
        The validated selector.
    """
    if value not in allowed:
        allowed_text = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{name} must be one of {allowed_text}.")
    return value


def _matrix_element_scale(
    matrix_element: str,
    E1,
    E2,
    E3,
    E4,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    m: float,
):
    """Return the t-channel numerator scale relative to the old UR element."""
    matrix_element = _mode(
        matrix_element,
        ("ultrarelativistic", "paraxial_massive"),
        "matrix_element",
    )
    if matrix_element == "ultrarelativistic":
        return 1.0 if _helicity_conserving(lam1, lam2, lam3, lam4) else 0.0

    return moller_paraxial_massive_t_scale(
        E1,
        E2,
        E3,
        E4,
        lam1,
        lam2,
        lam3,
        lam4,
        m=m,
    )


@dataclass(frozen=True)
class FirstOrderLongitudinal:
    """Longitudinal parameters entering the first-order time block.

    Parameters
    ----------
    gamma1, gamma2:
        Lorentz factors for the incoming packet central energies.
    delta_kz:
        Longitudinal total-momentum mismatch.
    Omega, a, b, c, d:
        Scalars appearing in the first-order time-kernel formulas.

    Returns
    -------
    FirstOrderLongitudinal
        Immutable container for the longitudinal part of the first-order
        correction.
    """

    gamma1: float
    gamma2: float
    delta_kz: float
    Omega: float
    a: float
    b: float
    c: float
    d: float


def impulse_parameters(
    packet1: LGPacket,
    packet2: LGPacket,
    k3,
    k4,
    impact_b,
    m: float = ELECTRON_MASS,
) -> dict:
    """Return scalar parameters used by S-matrix formulas.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    k3, k4:
        Final three-momenta.
    impact_b:
        Transverse impact parameter.
    m:
        Particle mass.

    Returns
    -------
    dict
        Kinematic quantities, transverse Gaussian coefficients and
        longitudinal impulse parameters.

    Only ``Xi0`` is algebraically shortened.  The transverse parameters
    ``alpha``, ``beta`` and ``gamma`` are intentionally kept in the original
    derivation form.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()

    k3 = vec3(k3)
    k4 = vec3(k4)
    b = vec2(impact_b)

    K_vec = k3 + k4
    K_perp = K_vec[:2]
    Kz = K_vec[2]

    K_perp_sq = np.dot(K_perp, K_perp)

    E3 = energy(k3, m)
    E4 = energy(k4, m)
    E_K = E3 + E4

    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)

    v1 = packet1.kbar_z / eps1
    v2 = packet2.kbar_z / eps2

    if v1 == v2:
        raise ValueError("The closed longitudinal impulse integral requires v1 != v2.")

    DeltaKz = Kz - packet1.kbar_z - packet2.kbar_z

    s1p = packet1.sigma_perp
    s1z = packet1.sigma_par
    s2p = packet2.sigma_perp
    s2z = packet2.sigma_par

    # Algebraically reduced Xi0 only.  The transverse parameters below are kept
    # in the original derivation form so that no additional cancellation between
    # exp(Xi0) and the transverse integral is imposed here.
    Xi0 = (
        0.5 * K_perp_sq * (1.0 / (s2z * s2z) - 1.0 / (s2p * s2p))
        + 1j * np.dot(b, K_perp)
    )

    A_long = FLOAT_DTYPE(0.0)

    Omega_long = eps1 + eps2 - E_K + v2 * DeltaKz

    alpha = -(
        + 1.0 / (s2z * s2z)
        - 1.0 / (s1p * s1p)
        - 1.0 / (s2p * s2p)
    )

    beta = 1.0 / (s2p * s2p) - 1.0 / (s2z * s2z)
    gamma = 1.0 / (s2z * s2z)

    return dict(
        K_vec=K_vec,
        K_perp=K_perp,
        Kz=Kz,
        E3=E3,
        E4=E4,
        E_K=E_K,
        eps1=eps1,
        eps2=eps2,
        v1=v1,
        v2=v2,
        DeltaKz=DeltaKz,
        Xi0=Xi0,
        A_long=A_long,
        Omega_long=Omega_long,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def S_impulse_common_factor(
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
    matrix_element: str = "ultrarelativistic",
) -> tuple[complex, dict]:
    """Return the common prefactor outside the transverse integral.

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

    Returns
    -------
    tuple[complex, dict]
        Common complex factor and diagnostic details.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    if matrix_element == "paraxial_massive":
        raise ValueError(
            "matrix_element='paraxial_massive' is implemented for "
            "S_exact_time with denominator_mode='minkowski'. The closed and "
            "first-order impulse formulas use the ultrarelativistic "
            "transverse t-channel denominator."
        )

    b = vec2(impact_b)
    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )

    pars = impulse_parameters(packet1, packet2, k3, k4, b, m)
    matrix_scale = _matrix_element_scale(
        matrix_element,
        pars["eps1"],
        pars["eps2"],
        pars["E3"],
        pars["E4"],
        lam1,
        lam2,
        lam3,
        lam4,
        m,
    )
    if np.all(matrix_scale == 0.0):
        return complex_zero(), {"reason": "matrix element helicity factor is zero"}

    # This common factor is before the transverse integral; the transverse
    # routines already include the angular 2*pi from d^2q.
    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(pars["E3"] * pars["E4"] / (pars["eps1"] * pars["eps2"]))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
        * matrix_scale
    )

    longitudinal_factor = (
        1.0
        / abs(pars["v1"] - pars["v2"])
        * np.exp(pars["Omega_long"] * pars["A_long"] / (pars["v1"] - pars["v2"]))
    )

    common_factor = prefactor * np.exp(pars["Xi0"]) * longitudinal_factor

    details = dict(pars)
    details.update(
        dict(
            N1=N1,
            N2=N2,
            impact_b=b,
            prefactor=prefactor,
            longitudinal_factor=longitudinal_factor,
            common_factor=common_factor,
            matrix_element=matrix_element,
            matrix_element_scale=matrix_scale,
        )
    )

    return common_factor, details


def _impulse_parameters_grid(
    k3x,
    k3y,
    k3z,
    k4x,
    k4y,
    k4z,
    packet1: LGPacket,
    packet2: LGPacket,
    impact_b=(0.0, 0.0),
    m: float = ELECTRON_MASS,
) -> dict:
    """Return vectorized impulse parameters for broadcastable momenta."""
    b = vec2(impact_b)
    Kx = k3x + k4x
    Ky = k3y + k4y
    Kz = k3z + k4z
    K_perp_sq = Kx * Kx + Ky * Ky

    E3 = np.sqrt(m * m + k3x * k3x + k3y * k3y + k3z * k3z)
    E4 = np.sqrt(m * m + k4x * k4x + k4y * k4y + k4z * k4z)
    E_K = E3 + E4

    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)
    v1 = packet1.kbar_z / eps1
    v2 = packet2.kbar_z / eps2
    if v1 == v2:
        raise ValueError("The closed longitudinal impulse integral requires v1 != v2.")

    DeltaKz = Kz - packet1.kbar_z - packet2.kbar_z

    s1p = packet1.sigma_perp
    s2p = packet2.sigma_perp
    s2z = packet2.sigma_par

    Xi0 = (
        0.5 * K_perp_sq * (1.0 / (s2z * s2z) - 1.0 / (s2p * s2p))
        + 1j * (b[0] * Kx + b[1] * Ky)
    )
    A_long = FLOAT_DTYPE(0.0)
    Omega_long = eps1 + eps2 - E_K + v2 * DeltaKz

    alpha = -(+1.0 / (s2z * s2z) - 1.0 / (s1p * s1p) - 1.0 / (s2p * s2p))
    beta = 1.0 / (s2p * s2p) - 1.0 / (s2z * s2z)
    gamma = 1.0 / (s2z * s2z)

    return dict(
        K_perp=(Kx, Ky),
        Kz=Kz,
        impact_b=b,
        E3=E3,
        E4=E4,
        E_K=E_K,
        eps1=eps1,
        eps2=eps2,
        v1=v1,
        v2=v2,
        DeltaKz=DeltaKz,
        Xi0=Xi0,
        A_long=A_long,
        Omega_long=Omega_long,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def _S_impulse_common_factor_grid(
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
    matrix_element: str = "ultrarelativistic",
) -> tuple[np.ndarray, dict]:
    """Return the vectorized impulse prefactor outside transverse integrals."""
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    if matrix_element == "paraxial_massive":
        raise ValueError(
            "matrix_element='paraxial_massive' is implemented for "
            "S_exact_time_grid with denominator_mode='minkowski'. The closed "
            "and first-order impulse formulas use the ultrarelativistic "
            "transverse t-channel denominator."
        )
    b = vec2(impact_b)
    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )

    pars = _impulse_parameters_grid(
        k3x,
        k3y,
        k3z,
        k4x,
        k4y,
        k4z,
        packet1,
        packet2,
        impact_b=b,
        m=m,
    )
    matrix_scale = _matrix_element_scale(
        matrix_element,
        pars["eps1"],
        pars["eps2"],
        pars["E3"],
        pars["E4"],
        lam1,
        lam2,
        lam3,
        lam4,
        m,
    )
    if np.all(matrix_scale == 0.0):
        shape = np.broadcast(k3x, k3y, k3z, k4x, k4y, k4z).shape
        return np.zeros(shape, dtype=COMPLEX_DTYPE), {
            "reason": "matrix element helicity factor is zero"
        }

    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(pars["E3"] * pars["E4"] / (pars["eps1"] * pars["eps2"]))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
        * matrix_scale
    )
    longitudinal_factor = (
        1.0
        / abs(pars["v1"] - pars["v2"])
        * np.exp(pars["Omega_long"] * pars["A_long"] / (pars["v1"] - pars["v2"]))
    )
    common_factor = prefactor * np.exp(pars["Xi0"]) * longitudinal_factor

    details = dict(pars)
    details.update(
        dict(
            N1=N1,
            N2=N2,
            prefactor=prefactor,
            longitudinal_factor=longitudinal_factor,
            common_factor=common_factor,
            matrix_element=matrix_element,
            matrix_element_scale=matrix_scale,
        )
    )
    return common_factor, details


def S_impulse_closed_form(
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
    matrix_element: str = "ultrarelativistic",
    return_details: bool = False,
) -> complex | tuple[complex, dict]:
    """Compute the closed impulse S-matrix with analytic transverse integral.

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
    common_factor, details = S_impulse_common_factor(
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
        matrix_element=matrix_element,
    )

    if "reason" in details:
        return (common_factor, details) if return_details else common_factor

    k3_perp = vec3(k3)[:2]
    b = vec2(impact_b)

    Iperp, case = transverse_integral_explicit(
        packet1.ell,
        packet2.ell,
        k3_perp,
        details["K_perp"],
        b,
        details["alpha"],
        details["beta"],
        details["gamma"],
        return_case=True,
    )

    S = common_factor * Iperp

    if return_details:
        details.update(dict(Iperp=Iperp, transverse_case=case))
        return S, details

    return S


def S_impulse_closed_grid(
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
    matrix_element: str = "ultrarelativistic",
):
    """Vectorized closed impulse S matrix for broadcastable momenta."""
    common_factor, details = _S_impulse_common_factor_grid(
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
        matrix_element=matrix_element,
    )
    if "reason" in details:
        return common_factor

    Iperp = transverse_integral_explicit_grid(
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


def _first_order_longitudinal_parameters(
    packet1: LGPacket,
    packet2: LGPacket,
    details: dict,
    m: float,
) -> FirstOrderLongitudinal:
    """Return the longitudinal scalars used by ``S_impulse_first_order``.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    details:
        Dictionary produced by ``S_impulse_common_factor``.
    m:
        Particle mass.

    Returns
    -------
    FirstOrderLongitudinal
        Grouped longitudinal parameters for the first-order time block.
    """
    eps1 = details["eps1"]
    eps2 = details["eps2"]
    gamma1 = eps1 / m
    gamma2 = eps2 / m
    delta_kz = details["DeltaKz"]

    Omega = (
        details["Omega_long"]
        + delta_kz * delta_kz / (2.0 * eps2 * gamma2 * gamma2)
    )

    a_long = 0.5 * (
        1.0 / (eps1 * gamma1 * gamma1)
        + 1.0 / (eps2 * gamma2 * gamma2)
    )

    b_long = (
        1.0 / (2.0 * packet1.sigma_par * packet1.sigma_par * gamma1 * gamma1)
        + 1.0 / (2.0 * packet2.sigma_par * packet2.sigma_par * gamma2 * gamma2)
    )

    c_long = (
        details["v1"]
        - details["v2"]
        - delta_kz / (eps2 * gamma2 * gamma2)
    )

    d_long = (
        details["A_long"]
        + delta_kz / (packet2.sigma_par * packet2.sigma_par * gamma2 * gamma2)
    )

    return FirstOrderLongitudinal(
        gamma1=gamma1,
        gamma2=gamma2,
        delta_kz=delta_kz,
        Omega=Omega,
        a=a_long,
        b=b_long,
        c=c_long,
        d=d_long,
    )


def _automatic_time_step(
    alpha0: complex,
    gamma0: complex,
    eps1: float,
    eps2: float,
    time_step_scale: float,
) -> float:
    """Choose the finite-difference step from transverse-coefficient scales.

    Parameters
    ----------
    alpha0, gamma0:
        Time-zero transverse Gaussian coefficients.
    eps1, eps2:
        Central energies of incoming packets.
    time_step_scale:
        Dimensionless multiplier for the natural time scale.

    Returns
    -------
    float
        Finite-difference step.
    """
    A0 = alpha0 + gamma0
    A_dot_abs = abs(1.0 / eps1 + 1.0 / eps2)
    gamma_dot_abs = abs(1.0 / eps2)

    time_scale_A = abs(A0) / A_dot_abs
    if gamma0 == 0:
        time_scale_gamma = np.inf
    else:
        time_scale_gamma = abs(gamma0) / gamma_dot_abs

    return time_step_scale * min(time_scale_A, time_scale_gamma)


def _five_point_time_derivatives(values: tuple[complex, complex, complex, complex, complex], h: float):
    """Return ``I0``, ``I1`` and ``I2`` from five time samples.

    Parameters
    ----------
    values:
        Transverse integral samples at ``-2h``, ``-h``, ``0``, ``h``, ``2h``.
    h:
        Time step.

    Returns
    -------
    tuple[complex, complex, complex]
        Integral value and first two time derivatives at zero.
    """
    I_m2, I_m1, I0, I_p1, I_p2 = values

    I1 = (
        -I_p2
        + 8.0 * I_p1
        - 8.0 * I_m1
        + I_m2
    ) / (12.0 * h)

    I2 = (
        -I_p2
        + 16.0 * I_p1
        - 30.0 * I0
        + 16.0 * I_m1
        - I_m2
    ) / (12.0 * h * h)

    return I0, I1, I2


def _first_order_time_block(
    longitudinal: FirstOrderLongitudinal,
    I0: complex,
    I1: complex,
    I2: complex,
) -> tuple[complex, dict]:
    """Assemble the resummed first-order time block.

    Parameters
    ----------
    longitudinal:
        Longitudinal first-order parameters.
    I0, I1, I2:
        Transverse integral and its first two time derivatives at zero.

    Returns
    -------
    tuple[complex, dict]
        Time block without the scalar longitudinal exponential and
        intermediate diagnostic values.  The caller combines
        ``longitudinal_exponent`` with the remaining Gaussian exponent before
        exponentiating; this avoids overflow from cancelling exponentials.
    """
    Omega = longitudinal.Omega
    a_long = longitudinal.a
    b_long = longitudinal.b
    c_long = longitudinal.c
    d_long = longitudinal.d
    c2 = c_long * c_long

    longitudinal_exponent = (
        -b_long * Omega * Omega / c2
        -d_long * Omega / c_long
    )

    longitudinal_factor = 2.0 * np.sqrt(np.pi) / c_long

    q_time = (
        2.0 * b_long * Omega / c2
        + d_long / c_long
    )

    transverse_time_bracket = (
        I0
        + 1j * q_time * I1
        + (
            b_long / c2
            - 0.5 * q_time * q_time
        )
        * I2
        + 2.0 * a_long * Omega * I0 / c2
        - 2.0j * a_long * I1 / c2
    )

    time_block = longitudinal_factor * transverse_time_bracket
    return time_block, dict(
        longitudinal_exponent=longitudinal_exponent,
        longitudinal_factor=longitudinal_factor,
        q_time=q_time,
        transverse_time_bracket=transverse_time_bracket,
    )


def S_impulse_first_order(
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
    time_step: float | None = None,
    time_step_scale: float = 1.0e-4,
    matrix_element: str = "ultrarelativistic",
    return_details: bool = False,
) -> complex | tuple[complex, dict]:
    """S-matrix beyond strict impulse approximation.

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
        First-order S-matrix value, optionally with diagnostic details.
    """

    common_factor, details = S_impulse_common_factor(
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
        matrix_element=matrix_element,
    )

    if "reason" in details:
        return (common_factor, details) if return_details else common_factor

    k3 = vec3(k3)
    b_perp = vec2(impact_b)

    k3_perp = k3[:2]
    K_perp = details["K_perp"]

    longitudinal = _first_order_longitudinal_parameters(packet1, packet2, details, m)

    alpha0 = details["alpha"]
    beta0 = details["beta"]
    gamma0 = details["gamma"]

    def transverse_integral_at_time(t):
        """Evaluate the transverse integral with time-shifted widths.

        Parameters
        ----------
        t:
            Time offset used in the finite-difference stencil.

        Returns
        -------
        complex
            Analytic transverse integral at the shifted time.
        """
        alpha_t = alpha0 - 1j * t / details["eps1"]
        gamma_t = gamma0 - 1j * t / details["eps2"]

        return transverse_integral_explicit(
            packet1.ell,
            packet2.ell,
            k3_perp,
            K_perp,
            b_perp,
            alpha_t,
            beta0,
            gamma_t,
        )

    if time_step is None:
        time_step = _automatic_time_step(
            alpha0,
            gamma0,
            details["eps1"],
            details["eps2"],
            time_step_scale,
        )

    samples = tuple(
        transverse_integral_at_time(multiplier * time_step)
        for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    I0, I1, I2 = _five_point_time_derivatives(samples, time_step)
    time_block, time_details = _first_order_time_block(longitudinal, I0, I1, I2)

    xi0_localization = (
        -longitudinal.delta_kz * longitudinal.delta_kz
        / (
            2.0
            * packet2.sigma_par
            * packet2.sigma_par
            * longitudinal.gamma2
            * longitudinal.gamma2
        )
    )

    combined_exponent = (
        details["Xi0"]
        + xi0_localization
        + time_details["longitudinal_exponent"]
    )
    base_factor = details["prefactor"] / (2.0 * np.sqrt(np.pi)) * np.exp(
        combined_exponent
    )

    S = base_factor * time_block

    if return_details:
        details.update(
            dict(
                gamma1_first=longitudinal.gamma1,
                gamma2_first=longitudinal.gamma2,
                Omega_first=longitudinal.Omega,
                a_long_first=longitudinal.a,
                b_long_first=longitudinal.b,
                c_long_first=longitudinal.c,
                d_long_first=longitudinal.d,
                time_step=time_step,
                I0=I0,
                I1=I1,
                I2=I2,
                time_block=time_block,
                combined_exponent=combined_exponent,
                base_factor=base_factor,
                S_first_order=S,
                **time_details,
            )
        )
        return S, details

    return S


def S_impulse_first_order_grid(
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
    time_step: float | None = None,
    time_step_scale: float = 1.0e-4,
    matrix_element: str = "ultrarelativistic",
):
    """Vectorized first-order impulse S matrix for broadcastable momenta."""
    common_factor, details = _S_impulse_common_factor_grid(
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
        matrix_element=matrix_element,
    )
    if "reason" in details:
        return common_factor

    longitudinal = _first_order_longitudinal_parameters(packet1, packet2, details, m)

    alpha0 = details["alpha"]
    beta0 = details["beta"]
    gamma0 = details["gamma"]

    if time_step is None:
        time_step = _automatic_time_step(
            alpha0,
            gamma0,
            details["eps1"],
            details["eps2"],
            time_step_scale,
        )

    samples = tuple(
        transverse_integral_explicit_grid(
            packet1.ell,
            packet2.ell,
            k3x,
            k3y,
            details["K_perp"],
            details["impact_b"],
            alpha0 - 1j * multiplier * time_step / details["eps1"],
            beta0,
            gamma0 - 1j * multiplier * time_step / details["eps2"],
        )
        for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    I0, I1, I2 = _five_point_time_derivatives(samples, time_step)
    time_block, time_details = _first_order_time_block(longitudinal, I0, I1, I2)

    xi0_localization = -longitudinal.delta_kz * longitudinal.delta_kz / (
        2.0
        * packet2.sigma_par
        * packet2.sigma_par
        * longitudinal.gamma2
        * longitudinal.gamma2
    )
    combined_exponent = (
        details["Xi0"]
        + xi0_localization
        + time_details["longitudinal_exponent"]
    )
    base_factor = details["prefactor"] / (2.0 * np.sqrt(np.pi)) * np.exp(
        combined_exponent
    )
    return base_factor * time_block


def _vortex_monomial_z(z, ell: int):
    """Return the Cartesian vortex monomial for scalar or array ``z``."""
    if ell >= 0:
        return z ** ell
    return np.conjugate(z) ** (-ell)


@lru_cache(maxsize=64)
def _cached_exact_time_theta(n_theta: int, theta_method: str):
    """Return cached theta nodes, weights and trigonometric arrays."""
    if theta_method == "analytic":
        raise ValueError("theta_method='analytic' has no theta nodes.")
    theta_nodes, theta_weights = nodes_and_weights(
        (0.0, 2.0 * PI),
        n_theta,
        method=theta_method,
        endpoint=False,
    )
    return theta_nodes, theta_weights, np.cos(theta_nodes), np.sin(theta_nodes)


@lru_cache(maxsize=64)
def _cached_exact_time_unit_kappa(n_kappa: int, kappa_method: str):
    """Return cached unit-interval kappa nodes and weights."""
    return nodes_and_weights(
        (0.0, 1.0),
        n_kappa,
        method=kappa_method,
        endpoint=False,
    )


def _exact_time_A_perp_grid(
    qx,
    qy,
    *,
    Kx,
    Ky,
    packet1_kbar_z,
    k3x,
    k3y,
    k3z,
    E3,
    k3_perp_sq,
    b_perp,
    s1p: float,
    s2p: float,
    ell1: int,
    ell2: int,
    xi=None,
    m: float = ELECTRON_MASS,
    matrix_element: str,
    denominator_mode: str,
    denominator_regulator: float,
):
    """Return the exact-time transverse integrand on a batched q-grid."""
    K_minus_qx = Kx - qx
    K_minus_qy = Ky - qy

    if matrix_element == "paraxial_massive":
        if xi is None:
            raise ValueError("xi must be provided for the massive paraxial denominator.")
        k1z = packet1_kbar_z + xi
        q_sq = qx * qx + qy * qy
        E1_internal = np.sqrt(m * m + q_sq + k1z * k1z)
        diff_x = k3x - qx
        diff_y = k3y - qy
        diff_z = k3z - k1z
        t_channel = (
            (E1_internal - E3) * (E1_internal - E3)
            - diff_x * diff_x
            - diff_y * diff_y
            - diff_z * diff_z
        )
        denominator_factor = -1.0 / (
            t_channel - denominator_regulator * denominator_regulator
        )
        matrix_denominator_factor = denominator_factor
    elif denominator_mode == "expanded":
        denominator_factor = (
            1.0
            / k3_perp_sq
            * (1.0 + 2.0 * (qx * k3x + qy * k3y) / k3_perp_sq)
        )
        matrix_denominator_factor = denominator_factor
    else:
        diff_x = k3x - qx
        diff_y = k3y - qy
        denominator_factor = 1.0 / (
            diff_x * diff_x
            + diff_y * diff_y
            + denominator_regulator * denominator_regulator
        )
        matrix_denominator_factor = denominator_factor

    q_sq = qx * qx + qy * qy
    K_minus_q_sq = K_minus_qx * K_minus_qx + K_minus_qy * K_minus_qy
    phase = b_perp[0] * K_minus_qx + b_perp[1] * K_minus_qy
    transverse_exp = np.exp(
        -0.5 * q_sq / (s1p * s1p)
        -0.5 * K_minus_q_sq / (s2p * s2p)
        + 1j * phase
    )
    z1 = qx + 1j * qy
    z2 = K_minus_qx + 1j * K_minus_qy
    vortex_factor = _vortex_monomial_z(z1, ell1) * _vortex_monomial_z(z2, ell2)
    return matrix_denominator_factor * transverse_exp * vortex_factor


def _angular_kernels_with_exponent(min_m: int, max_m: int, u, v, C_perp):
    """Return exp(C_perp) K_m(u, v) for a contiguous integer m range."""
    u = np.asarray(u, dtype=COMPLEX_DTYPE)
    v = np.asarray(v, dtype=COMPLEX_DTYPE)
    C_perp = np.asarray(C_perp, dtype=COMPLEX_DTYPE)
    u, v, C_perp = np.broadcast_arrays(u, v, C_perp)

    zero_mask = (np.abs(u) == 0.0) | (np.abs(v) == 0.0)
    bessel_mask = ~zero_mask
    kernels = {}

    if np.any(bessel_mask):
        sqrt_u = np.sqrt(u[bessel_mask])
        sqrt_v = np.sqrt(v[bessel_mask])
        z = 2.0 * sqrt_u * sqrt_v
        scale = np.exp(C_perp[bessel_mask] + np.abs(np.real(z)))
        ratio_base = sqrt_v / sqrt_u

    if np.any(zero_mask):
        u_zero = u[zero_mask]
        v_zero = v[zero_mask]
        exp_C_zero = np.exp(C_perp[zero_mask])
        u_is_zero = np.abs(u_zero) == 0.0
        v_is_zero = np.abs(v_zero) == 0.0
        both_zero = u_is_zero & v_is_zero
        u_only_zero = u_is_zero & ~v_is_zero
        v_only_zero = v_is_zero & ~u_is_zero

    for m in range(min_m, max_m + 1):
        kernel = np.empty_like(u, dtype=COMPLEX_DTYPE)
        if np.any(bessel_mask):
            kernel[bessel_mask] = scale * ive(abs(m), z) * ratio_base ** m
        if np.any(zero_mask):
            zero_kernel = np.zeros_like(u_zero, dtype=COMPLEX_DTYPE)
            if m == 0:
                zero_kernel[both_zero] = 1.0
            if m >= 0:
                zero_kernel[u_only_zero] = (
                    v_zero[u_only_zero] ** m / math.factorial(m)
                )
            if m <= 0:
                zero_kernel[v_only_zero] = (
                    u_zero[v_only_zero] ** (-m) / math.factorial(-m)
                )
            kernel[zero_mask] = exp_C_zero * zero_kernel
        kernels[m] = kernel

    return kernels


def _source_derivative_terms(base, kappa, power: int, sign: float):
    """Return binomial terms for one source derivative block."""
    return [
        math.comb(power, index)
        * base ** (power - index)
        * (sign * kappa) ** index
        for index in range(power + 1)
    ]


def _source_derivative_convolution(q0, p0, kappa, q_power: int, p_power: int):
    """Return coefficients summed over source derivatives with equal order."""
    q_terms = _source_derivative_terms(q0, kappa, q_power, +1.0)
    p_terms = _source_derivative_terms(p0, kappa, p_power, -1.0)
    coefficients = []
    for order in range(q_power + p_power + 1):
        coeff = np.zeros_like(kappa, dtype=COMPLEX_DTYPE)
        q_min = max(0, order - p_power)
        q_max = min(q_power, order)
        for q_index in range(q_min, q_max + 1):
            coeff = coeff + q_terms[q_index] * p_terms[order - q_index]
        coefficients.append(coeff)
    return coefficients


def _exact_time_D_from_terms(plus_terms, minus_terms, kernels_scaled):
    """Return exp(C_perp) D from precomputed source-derivative terms."""
    total = np.zeros_like(plus_terms[0], dtype=COMPLEX_DTYPE)
    for plus_order, plus_term in enumerate(plus_terms):
        for minus_order, minus_term in enumerate(minus_terms):
            total = total + (
                plus_term
                * minus_term
                * kernels_scaled[plus_order - minus_order]
            )
    return total


def _exact_time_angular_integral_expanded(
    kappa,
    *,
    Kx,
    Ky,
    q0x,
    q0y,
    k3x,
    k3y,
    k3_perp_sq,
    b_perp,
    s1p: float,
    s2p: float,
    ell1: int,
    ell2: int,
):
    """Return the analytic theta integral for the expanded exact-time integrand."""
    kappa = np.asarray(kappa, dtype=FLOAT_DTYPE)

    p0x = Kx - q0x
    p0y = Ky - q0y

    q0_plus = q0x + 1j * q0y
    q0_minus = q0x - 1j * q0y
    p0_plus = p0x + 1j * p0y
    p0_minus = p0x - 1j * p0y

    q0_sq = q0x * q0x + q0y * q0y
    p0_sq = p0x * p0x + p0y * p0y
    kappa_sq = kappa * kappa
    C_perp = (
        -0.5 * (q0_sq + kappa_sq) / (s1p * s1p)
        -0.5 * (p0_sq + kappa_sq) / (s2p * s2p)
        + 1j * (b_perp[0] * p0x + b_perp[1] * p0y)
    )

    alpha_x = kappa * (-q0x / (s1p * s1p) + p0x / (s2p * s2p) - 1j * b_perp[0])
    alpha_y = kappa * (-q0y / (s1p * s1p) + p0y / (s2p * s2p) - 1j * b_perp[1])
    u = 0.5 * (alpha_x - 1j * alpha_y)
    v = 0.5 * (alpha_x + 1j * alpha_y)

    L1 = abs(ell1)
    L2 = abs(ell2)
    a = L1 if ell1 >= 0 else 0
    b = 0 if ell1 >= 0 else L1
    c = L2 if ell2 >= 0 else 0
    d = 0 if ell2 >= 0 else L2

    min_m = -(b + d + 1)
    max_m = a + c + 1
    kernels_scaled = _angular_kernels_with_exponent(min_m, max_m, u, v, C_perp)

    plus_base = _source_derivative_convolution(q0_plus, p0_plus, kappa, a, c)
    plus_q = _source_derivative_convolution(q0_plus, p0_plus, kappa, a + 1, c)
    minus_base = _source_derivative_convolution(q0_minus, p0_minus, kappa, b, d)
    minus_q = _source_derivative_convolution(q0_minus, p0_minus, kappa, b + 1, d)

    D0 = _exact_time_D_from_terms(plus_base, minus_base, kernels_scaled)
    D_plus = _exact_time_D_from_terms(plus_q, minus_base, kernels_scaled)
    D_minus = _exact_time_D_from_terms(plus_base, minus_q, kernels_scaled)

    k3_plus = k3x + 1j * k3y
    k3_minus = k3x - 1j * k3y
    return 2.0 * PI * (
        D0 / k3_perp_sq
        + (k3_minus * D_plus + k3_plus * D_minus)
        / (k3_perp_sq * k3_perp_sq)
    )


def S_exact_time_grid(
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
    quadrature: ExactTimeQuadrature | None = None,
    denominator_mode: str = "expanded",
    denominator_regulator: float = 0.0,
    matrix_element: str = "ultrarelativistic",
    batch_size: int = 32,
):
    """Vectorized exact-time S matrix on broadcastable momentum grids.

    The outer probability grid is processed in batches. Numerical theta
    quadrature builds arrays of shape ``(batch, n_radial, n_theta)``; analytic
    theta integration uses ``(batch, n_radial)`` arrays. Both paths keep memory
    bounded while removing Python loops over individual final-momentum points.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    quadrature = ExactTimeQuadrature() if quadrature is None else quadrature
    denominator_mode = _mode(
        denominator_mode,
        ("expanded", "exact", "minkowski"),
        "denominator_mode",
    )
    analytic_theta = quadrature.theta_method == "analytic"
    if analytic_theta and denominator_mode != "expanded":
        raise ValueError(
            "theta_method='analytic' is implemented only for "
            "denominator_mode='expanded'."
        )
    if matrix_element == "paraxial_massive":
        if denominator_mode != "minkowski":
            raise ValueError(
                "matrix_element='paraxial_massive' requires "
                "denominator_mode='minkowski'; the transverse-only "
                "'expanded' and 'exact' denominators are ultrarelativistic."
            )
        if analytic_theta:
            raise ValueError(
                "matrix_element='paraxial_massive' requires numerical theta "
                "quadrature because (k1-k3)^2 depends on xi and theta. Use "
                "ExactTimeQuadrature(theta_method='trapezoid')."
            )
    elif denominator_mode == "minkowski":
        raise ValueError(
            "denominator_mode='minkowski' is defined only for "
            "matrix_element='paraxial_massive'."
        )
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    arrays = np.broadcast_arrays(k3x, k3y, k3z, k4x, k4y, k4z)
    shape = arrays[0].shape
    out = np.zeros(shape, dtype=COMPLEX_DTYPE)

    k3x_f, k3y_f, k3z_f, k4x_f, k4y_f, k4z_f = (
        np.ravel(np.asarray(arr, dtype=FLOAT_DTYPE)) for arr in arrays
    )
    out_f = out.ravel()
    b_perp = vec2(impact_b)

    Kx = k3x_f + k4x_f
    Ky = k3y_f + k4y_f
    Kz = k3z_f + k4z_f
    K_perp_sq = Kx * Kx + Ky * Ky
    k3_perp_sq = k3x_f * k3x_f + k3y_f * k3y_f

    if denominator_mode == "expanded" and np.any(k3_perp_sq == 0.0):
        raise ValueError("Expanded denominator requires nonzero |k3_perp|.")

    E3 = np.sqrt(m * m + k3_perp_sq + k3z_f * k3z_f)
    E4 = np.sqrt(m * m + k4x_f * k4x_f + k4y_f * k4y_f + k4z_f * k4z_f)
    E_K = E3 + E4

    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)
    matrix_scale = _matrix_element_scale(
        matrix_element,
        eps1,
        eps2,
        E3,
        E4,
        lam1,
        lam2,
        lam3,
        lam4,
        m,
    )
    if np.all(matrix_scale == 0.0):
        return out

    gamma1 = eps1 / m
    gamma2 = eps2 / m
    v1 = packet1.kbar_z / eps1
    v2 = packet2.kbar_z / eps2

    DeltaKz = Kz - packet1.kbar_z - packet2.kbar_z

    s1p = packet1.sigma_perp
    s1z = packet1.sigma_par
    s2p = packet2.sigma_perp
    s2z = packet2.sigma_par

    A_z = 0.5 * (
        1.0 / (eps1 * gamma1 * gamma1)
        + 1.0 / (eps2 * gamma2 * gamma2)
    )
    B_z = (
        eps1 / (2.0 * s1z * s1z) * 1.0 / (eps1 * gamma1 * gamma1)
        + eps2 / (2.0 * s2z * s2z) * 1.0 / (eps2 * gamma2 * gamma2)
    )
    C_z = v1 - v2 - DeltaKz / (eps2 * gamma2 * gamma2)
    D_z = (
        packet1.kbar_z / (s1z * s1z)
        - packet2.kbar_z / (s2z * s2z)
        - v1 * eps1 / (s1z * s1z)
        + v2 * eps2 / (s2z * s2z)
        + eps2 / (s2z * s2z) * DeltaKz / (eps2 * gamma2 * gamma2)
    )
    Omega_z = (
        eps1
        + eps2
        - E_K
        + v2 * DeltaKz
        + DeltaKz * DeltaKz / (2.0 * eps2 * gamma2 * gamma2)
    )
    longitudinal_exp_arg = -DeltaKz * DeltaKz / (
        2.0 * s2z * s2z * gamma2 * gamma2
    )

    eta_perp = 1.0 / eps1 + 1.0 / eps2
    q0x = eps1 / (eps1 + eps2) * Kx
    q0y = eps1 / (eps1 + eps2) * Ky

    Delta0 = C_z * C_z - 4.0 * A_z * (
        Omega_z + K_perp_sq / (2.0 * (eps1 + eps2))
    )
    active = Delta0 > 0.0
    if not np.any(active):
        return out

    sqrt_Delta0 = np.zeros_like(Delta0)
    sqrt_Delta0[active] = np.sqrt(Delta0[active])
    R = np.zeros_like(Delta0)
    R[active] = np.sqrt(Delta0[active] / (2.0 * A_z * eta_perp))

    if denominator_mode == "exact" and denominator_regulator == 0.0:
        pole_distance = np.sqrt((k3x_f - q0x) ** 2 + (k3y_f - q0y) ** 2)
        if np.any(active & (pole_distance < R)):
            raise ValueError(
                "The exact Moller denominator has a non-integrable pole inside "
                "the exact-time q-disk. Use denominator_regulator or "
                "denominator_mode='expanded'."
            )

    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )
    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(E3 * E4 / (eps1 * eps2))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
        * matrix_scale
    )

    if not analytic_theta:
        _, theta_weights, cos_theta, sin_theta = _cached_exact_time_theta(
            quadrature.n_theta,
            quadrature.theta_method,
        )
        theta_weights = theta_weights[None, None, :]
        cos_theta = cos_theta[None, None, :]
        sin_theta = sin_theta[None, None, :]

    active_indices = np.nonzero(active)[0]

    n_kappa = quadrature.n_kappa
    kappa_method = quadrature.kappa_method

    unit_nodes, unit_weights = _cached_exact_time_unit_kappa(n_kappa, kappa_method)
    unit_nodes = unit_nodes[None, :, None]
    unit_weights = unit_weights[None, :, None]

    sigma_eff = 1.0 / np.sqrt(1.0 / (s1p * s1p) + 1.0 / (s2p * s2p))
    gaussian_fraction = s1p * s1p / (s1p * s1p + s2p * s2p)
    disk_curvature = np.zeros_like(Delta0)
    disk_curvature[active] = 2.0 * A_z * eta_perp / Delta0[active]

    for start in range(0, len(active_indices), batch_size):
        idx = active_indices[start : start + batch_size]
        bshape = (len(idx), 1, 1)

        if quadrature.kappa_n_sigma is None:
            radial_lower = np.zeros(len(idx), dtype=FLOAT_DTYPE)
            radial_upper = R[idx]
        else:
            gaussian_center_x = gaussian_fraction * Kx[idx]
            gaussian_center_y = gaussian_fraction * Ky[idx]
            center_distance = np.sqrt(
                (gaussian_center_x - q0x[idx]) ** 2
                + (gaussian_center_y - q0y[idx]) ** 2
            )
            support = quadrature.kappa_n_sigma * sigma_eff
            radial_lower = np.maximum(0.0, center_distance - support)
            radial_upper = np.minimum(R[idx], center_distance + support)

        width = np.maximum(radial_upper - radial_lower, 0.0)
        kappa_nodes = radial_lower.reshape(bshape) + width.reshape(bshape) * unit_nodes
        kappa_weights = width.reshape(bshape) * unit_weights

        curvature = disk_curvature[idx].reshape(bshape)
        discriminant_factor = 1.0 - curvature * kappa_nodes * kappa_nodes
        valid = discriminant_factor > 0.0
        sqrt_factor = np.sqrt(np.where(valid, discriminant_factor, 1.0))

        sqrt_D = sqrt_Delta0[idx].reshape(bshape)
        C = C_z[idx].reshape(bshape)
        D = D_z[idx].reshape(bshape)
        longitudinal_exp = longitudinal_exp_arg[idx].reshape(bshape)
        xi_plus = (-C + sqrt_D * sqrt_factor) / (2.0 * A_z)
        xi_minus = (-C - sqrt_D * sqrt_factor) / (2.0 * A_z)
        root_weight_plus = np.exp(
            longitudinal_exp - B_z * xi_plus * xi_plus + D * xi_plus
        )
        root_weight_minus = np.exp(
            longitudinal_exp - B_z * xi_minus * xi_minus + D * xi_minus
        )
        root_weight = root_weight_plus + root_weight_minus

        if analytic_theta:
            kappa_2d = kappa_nodes[:, :, 0]
            kappa_weights_2d = kappa_weights[:, :, 0]
            sqrt_factor_2d = sqrt_factor[:, :, 0]
            root_weight_2d = root_weight[:, :, 0]
            valid_2d = valid[:, :, 0]
            row_shape = (len(idx), 1)

            angular_integral = _exact_time_angular_integral_expanded(
                kappa_2d,
                Kx=Kx[idx].reshape(row_shape),
                Ky=Ky[idx].reshape(row_shape),
                q0x=q0x[idx].reshape(row_shape),
                q0y=q0y[idx].reshape(row_shape),
                k3x=k3x_f[idx].reshape(row_shape),
                k3y=k3y_f[idx].reshape(row_shape),
                k3_perp_sq=k3_perp_sq[idx].reshape(row_shape),
                b_perp=b_perp,
                s1p=s1p,
                s2p=s2p,
                ell1=packet1.ell,
                ell2=packet2.ell,
            )
            weights = (
                kappa_weights_2d
                * kappa_2d
                / sqrt_factor_2d
                * root_weight_2d
                * valid_2d
            )
            integral = np.sum(weights * angular_integral, axis=1)
        else:
            qx = q0x[idx].reshape(bshape) + kappa_nodes * cos_theta
            qy = q0y[idx].reshape(bshape) + kappa_nodes * sin_theta
            weights = kappa_weights * theta_weights * kappa_nodes / sqrt_factor * valid

            if matrix_element == "paraxial_massive":
                A_perp_plus = _exact_time_A_perp_grid(
                    qx,
                    qy,
                    Kx=Kx[idx].reshape(bshape),
                    Ky=Ky[idx].reshape(bshape),
                    packet1_kbar_z=packet1.kbar_z,
                    k3x=k3x_f[idx].reshape(bshape),
                    k3y=k3y_f[idx].reshape(bshape),
                    k3z=k3z_f[idx].reshape(bshape),
                    E3=E3[idx].reshape(bshape),
                    k3_perp_sq=k3_perp_sq[idx].reshape(bshape),
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                    xi=xi_plus,
                    m=m,
                    matrix_element=matrix_element,
                    denominator_mode=denominator_mode,
                    denominator_regulator=denominator_regulator,
                )
                A_perp_minus = _exact_time_A_perp_grid(
                    qx,
                    qy,
                    Kx=Kx[idx].reshape(bshape),
                    Ky=Ky[idx].reshape(bshape),
                    packet1_kbar_z=packet1.kbar_z,
                    k3x=k3x_f[idx].reshape(bshape),
                    k3y=k3y_f[idx].reshape(bshape),
                    k3z=k3z_f[idx].reshape(bshape),
                    E3=E3[idx].reshape(bshape),
                    k3_perp_sq=k3_perp_sq[idx].reshape(bshape),
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                    xi=xi_minus,
                    m=m,
                    matrix_element=matrix_element,
                    denominator_mode=denominator_mode,
                    denominator_regulator=denominator_regulator,
                )
                root_integrand = (
                    root_weight_plus * A_perp_plus
                    + root_weight_minus * A_perp_minus
                )
            else:
                A_perp = _exact_time_A_perp_grid(
                    qx,
                    qy,
                    Kx=Kx[idx].reshape(bshape),
                    Ky=Ky[idx].reshape(bshape),
                    packet1_kbar_z=packet1.kbar_z,
                    k3x=k3x_f[idx].reshape(bshape),
                    k3y=k3y_f[idx].reshape(bshape),
                    k3z=k3z_f[idx].reshape(bshape),
                    E3=E3[idx].reshape(bshape),
                    k3_perp_sq=k3_perp_sq[idx].reshape(bshape),
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                    matrix_element=matrix_element,
                    denominator_mode=denominator_mode,
                    denominator_regulator=denominator_regulator,
                )
                root_integrand = root_weight * A_perp

            integral = np.sum(weights * root_integrand, axis=(1, 2))
        disk_factor = 1.0 / sqrt_Delta0[idx]
        out_f[idx] = (
            prefactor[idx]
            * disk_factor
            * integral
        )

    return out


def S_exact_time(
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
    quadrature: ExactTimeQuadrature | None = None,
    denominator_mode: str = "expanded",
    denominator_regulator: float = 0.0,
    matrix_element: str = "ultrarelativistic",
    batch_size: int | None = None,
    return_details: bool = False,
):
    """S-matrix with the time integral evaluated by the delta representation.

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
    quadrature:
        Exact-time quadrature settings. Defaults to ``ExactTimeQuadrature()``.
    denominator_mode:
        ``"expanded"`` for the ultrarelativistic first-order transverse
        denominator, ``"exact"`` for the regulated ultrarelativistic
        transverse denominator, or ``"minkowski"`` for the invariant
        ``(k1-k3)^2`` denominator used with
        ``matrix_element="paraxial_massive"``.
    denominator_regulator:
        Regulator added in ``"exact"`` and ``"minkowski"`` denominator modes.
    batch_size:
        Batch size used by the vectorized non-diagnostic path.  It is ignored
        when ``return_details=True``.
    return_details:
        If True, return ``(S, details)``.

    Returns
    -------
    complex or tuple[complex, dict]
        Exact-time S-matrix value, optionally with diagnostic details.

    The exact-time disk is integrated with the direct radial formula

        q = q0 + kappa (cos theta, sin theta).

    With ``quadrature.theta_method="analytic"``, the theta integral is
    evaluated in closed form for the ultrarelativistic
    ``denominator_mode="expanded"``. Otherwise theta is integrated by the
    quadrature rule named in ``theta_method``. The massive paraxial
    ``"minkowski"`` denominator requires numerical theta quadrature.

    A Gaussian-support cutoff can be used to avoid wasting nodes when the
    exact-time disk radius is much larger than the transverse packet widths.

    The exponent is evaluated through the algebraically combined
    ``Xi0 + Xi_perp`` form to avoid artificial cancellations.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    quadrature = ExactTimeQuadrature() if quadrature is None else quadrature
    denominator_mode = _mode(
        denominator_mode,
        ("expanded", "exact", "minkowski"),
        "denominator_mode",
    )
    analytic_theta = quadrature.theta_method == "analytic"
    if analytic_theta and denominator_mode != "expanded":
        raise ValueError(
            "theta_method='analytic' is implemented only for "
            "denominator_mode='expanded'."
        )
    if matrix_element == "paraxial_massive":
        if denominator_mode != "minkowski":
            raise ValueError(
                "matrix_element='paraxial_massive' requires "
                "denominator_mode='minkowski'; the transverse-only "
                "'expanded' and 'exact' denominators are ultrarelativistic."
            )
        if analytic_theta:
            raise ValueError(
                "matrix_element='paraxial_massive' requires numerical theta "
                "quadrature because (k1-k3)^2 depends on xi and theta. Use "
                "ExactTimeQuadrature(theta_method='trapezoid')."
            )
    elif denominator_mode == "minkowski":
        raise ValueError(
            "denominator_mode='minkowski' is defined only for "
            "matrix_element='paraxial_massive'."
        )

    if not return_details:
        k3_vec = vec3(k3)
        k4_vec = vec3(k4)
        S = S_exact_time_grid(
            k3_vec[0],
            k3_vec[1],
            k3_vec[2],
            k4_vec[0],
            k4_vec[1],
            k4_vec[2],
            packet1,
            packet2,
            lam1=lam1,
            lam2=lam2,
            lam3=lam3,
            lam4=lam4,
            m=m,
            e_charge=e_charge,
            impact_b=impact_b,
            N1=N1,
            N2=N2,
            quadrature=quadrature,
            denominator_mode=denominator_mode,
            denominator_regulator=denominator_regulator,
            matrix_element=matrix_element,
            batch_size=1 if batch_size is None else batch_size,
        )
        return S.reshape(-1)[0]

    k3 = vec3(k3)
    k4 = vec3(k4)
    b_perp = vec2(impact_b)

    K_vec = k3 + k4
    K_perp = K_vec[:2]
    Kz = K_vec[2]
    K_perp_sq = np.dot(K_perp, K_perp)

    k3_perp = k3[:2]
    k3_perp_sq = np.dot(k3_perp, k3_perp)

    E3 = energy(k3, m)
    E4 = energy(k4, m)
    E_K = E3 + E4

    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)
    matrix_scale = _matrix_element_scale(
        matrix_element,
        eps1,
        eps2,
        E3,
        E4,
        lam1,
        lam2,
        lam3,
        lam4,
        m,
    )
    if np.all(matrix_scale == 0.0):
        S_zero = complex_zero()
        details = {"reason": "matrix element helicity factor is zero"}
        return (S_zero, details) if return_details else S_zero

    gamma1 = eps1 / m
    gamma2 = eps2 / m

    v1 = packet1.kbar_z / eps1
    v2 = packet2.kbar_z / eps2

    DeltaKz = Kz - packet1.kbar_z - packet2.kbar_z

    s1p = packet1.sigma_perp
    s1z = packet1.sigma_par
    s2p = packet2.sigma_perp
    s2z = packet2.sigma_par

    A_z = 0.5 * (
        1.0 / (eps1 * gamma1 * gamma1)
        + 1.0 / (eps2 * gamma2 * gamma2)
    )
    B_z = (
        eps1 / (2.0 * s1z * s1z) * 1.0 / (eps1 * gamma1 * gamma1)
        + eps2 / (2.0 * s2z * s2z) * 1.0 / (eps2 * gamma2 * gamma2)
    )
    C_z = v1 - v2 - DeltaKz / (eps2 * gamma2 * gamma2)
    D_z = (
        packet1.kbar_z / (s1z * s1z)
        - packet2.kbar_z / (s2z * s2z)
        - v1 * eps1 / (s1z * s1z)
        + v2 * eps2 / (s2z * s2z)
        + eps2 / (s2z * s2z) * DeltaKz / (eps2 * gamma2 * gamma2)
    )
    Omega_z = (
        eps1
        + eps2
        - E_K
        + v2 * DeltaKz
        + DeltaKz * DeltaKz / (2.0 * eps2 * gamma2 * gamma2)
    )

    longitudinal_exp_arg = -DeltaKz * DeltaKz / (
        2.0 * s2z * s2z * gamma2 * gamma2
    )

    eta_perp = 1.0 / eps1 + 1.0 / eps2
    q0 = eps1 / (eps1 + eps2) * K_perp

    Delta0 = C_z * C_z - 4.0 * A_z * (
        Omega_z + K_perp_sq / (2.0 * (eps1 + eps2))
    )

    if Delta0 <= 0.0:
        S_zero = complex_zero()
        details = dict(
            A_z=A_z,
            B_z=B_z,
            C_z=C_z,
            D_z=D_z,
            Omega_z=Omega_z,
            Delta0=Delta0,
            reason="Delta0 is non-positive",
        )
        return (S_zero, details) if return_details else S_zero

    sqrt_Delta0 = np.sqrt(Delta0)
    R = np.sqrt(Delta0 / (2.0 * A_z * eta_perp))

    if denominator_mode == "exact" and denominator_regulator == 0.0:
        pole_distance = np.linalg.norm(k3_perp - q0)
        if pole_distance < R:
            raise ValueError(
                "The exact Moller denominator has a non-integrable pole inside "
                "the exact-time q-disk. Use denominator_regulator or "
                "denominator_mode='expanded'."
            )

    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )

    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(E3 * E4 / (eps1 * eps2))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
        * matrix_scale
    )

    def A_perp_grid(qx, qy, xi=None):
        K_minus_qx = K_perp[0] - qx
        K_minus_qy = K_perp[1] - qy

        if matrix_element == "paraxial_massive":
            if xi is None:
                raise ValueError(
                    "xi must be provided for the massive paraxial denominator."
                )
            k1z = packet1.kbar_z + xi
            q_sq = qx * qx + qy * qy
            E1_internal = np.sqrt(m * m + q_sq + k1z * k1z)
            diff_x = k3_perp[0] - qx
            diff_y = k3_perp[1] - qy
            diff_z = k3[2] - k1z
            t_channel = (
                (E1_internal - E3) * (E1_internal - E3)
                - diff_x * diff_x
                - diff_y * diff_y
                - diff_z * diff_z
            )
            denominator_factor = -1.0 / (
                t_channel - denominator_regulator * denominator_regulator
            )
            matrix_denominator_factor = denominator_factor
        elif denominator_mode == "expanded":
            if k3_perp_sq == 0.0:
                raise ValueError("Expanded denominator requires nonzero |k3_perp|.")
            denominator_factor = (
                1.0 / k3_perp_sq
                * (1.0 + 2.0 * (qx * k3_perp[0] + qy * k3_perp[1]) / k3_perp_sq)
            )
            matrix_denominator_factor = denominator_factor
        else:
            diff_x = k3_perp[0] - qx
            diff_y = k3_perp[1] - qy
            denominator_factor = 1.0 / (
                diff_x * diff_x
                + diff_y * diff_y
                + denominator_regulator * denominator_regulator
            )
            matrix_denominator_factor = denominator_factor

        q_sq = qx * qx + qy * qy
        K_minus_q_sq = K_minus_qx * K_minus_qx + K_minus_qy * K_minus_qy
        phase = b_perp[0] * K_minus_qx + b_perp[1] * K_minus_qy
        transverse_exp = np.exp(
            -0.5 * q_sq / (s1p * s1p)
            -0.5 * K_minus_q_sq / (s2p * s2p)
            + 1j * phase
        )
        z1 = qx + 1j * qy
        z2 = K_minus_qx + 1j * K_minus_qy
        vortex_factor = (
            _vortex_monomial_z(z1, packet1.ell)
            * _vortex_monomial_z(z2, packet2.ell)
        )
        return matrix_denominator_factor * transverse_exp * vortex_factor

    integral = complex_zero()
    sigma_eff = 1.0 / np.sqrt(1.0 / (s1p * s1p) + 1.0 / (s2p * s2p))
    gaussian_center = (s1p * s1p / (s1p * s1p + s2p * s2p)) * K_perp
    center_distance = np.linalg.norm(gaussian_center - q0)
    disk_curvature = 2.0 * A_z * eta_perp / Delta0

    if quadrature.kappa_n_sigma is None:
        radial_lower = 0.0
        radial_upper = R
    else:
        support = quadrature.kappa_n_sigma * sigma_eff
        radial_lower = max(0.0, center_distance - support)
        radial_upper = min(R, center_distance + support)

    if radial_upper > radial_lower:
        (kappa_nodes, kappa_weights), _ = exact_time_nodes(
            quadrature,
            radial_interval=(radial_lower, radial_upper),
        )

        discriminant_factor = 1.0 - disk_curvature * kappa_nodes * kappa_nodes
        valid = discriminant_factor > 0.0

        if np.any(valid):
            kappa_nodes = kappa_nodes[valid]
            kappa_weights = kappa_weights[valid]
            sqrt_factor = np.sqrt(discriminant_factor[valid])

            xi_plus = (-C_z + sqrt_Delta0 * sqrt_factor) / (2.0 * A_z)
            xi_minus = (-C_z - sqrt_Delta0 * sqrt_factor) / (2.0 * A_z)
            root_weight_plus = np.exp(
                longitudinal_exp_arg - B_z * xi_plus * xi_plus + D_z * xi_plus
            )
            root_weight_minus = np.exp(
                longitudinal_exp_arg - B_z * xi_minus * xi_minus + D_z * xi_minus
            )
            root_weight = root_weight_plus + root_weight_minus

            if analytic_theta:
                angular_integral = _exact_time_angular_integral_expanded(
                    kappa_nodes,
                    Kx=K_perp[0],
                    Ky=K_perp[1],
                    q0x=q0[0],
                    q0y=q0[1],
                    k3x=k3_perp[0],
                    k3y=k3_perp[1],
                    k3_perp_sq=k3_perp_sq,
                    b_perp=b_perp,
                    s1p=s1p,
                    s2p=s2p,
                    ell1=packet1.ell,
                    ell2=packet2.ell,
                )
                weights = kappa_weights * kappa_nodes / sqrt_factor * root_weight
                integral = np.sum(weights * angular_integral)
            else:
                _, theta_weights, cos_theta, sin_theta = _cached_exact_time_theta(
                    quadrature.n_theta,
                    quadrature.theta_method,
                )
                qx = q0[0] + kappa_nodes[:, None] * cos_theta[None, :]
                qy = q0[1] + kappa_nodes[:, None] * sin_theta[None, :]
                weights = (
                    kappa_weights[:, None]
                    * theta_weights[None, :]
                    * kappa_nodes[:, None]
                    / sqrt_factor[:, None]
                )
                if matrix_element == "paraxial_massive":
                    A_perp_plus = A_perp_grid(qx, qy, xi=xi_plus[:, None])
                    A_perp_minus = A_perp_grid(qx, qy, xi=xi_minus[:, None])
                    root_integrand = (
                        root_weight_plus[:, None] * A_perp_plus
                        + root_weight_minus[:, None] * A_perp_minus
                    )
                else:
                    root_integrand = root_weight[:, None] * A_perp_grid(qx, qy)
                integral = np.sum(weights * root_integrand)

    disk_factor = 1.0 / sqrt_Delta0

    S = prefactor * disk_factor * integral

    if return_details:
        details = dict(
            N1=N1,
            N2=N2,
            impact_b=b_perp,
            matrix_element=matrix_element,
            matrix_element_scale=matrix_scale,
            K_vec=K_vec,
            K_perp=K_perp,
            Kz=Kz,
            E3=E3,
            E4=E4,
            E_K=E_K,
            eps1=eps1,
            eps2=eps2,
            gamma1=gamma1,
            gamma2=gamma2,
            v1=v1,
            v2=v2,
            DeltaKz=DeltaKz,
            A_z=A_z,
            B_z=B_z,
            C_z=C_z,
            D_z=D_z,
            Omega_z=Omega_z,
            eta_perp=eta_perp,
            q0=q0,
            Delta0=Delta0,
            R=R,
            prefactor=prefactor,
            longitudinal_exp_arg=longitudinal_exp_arg,
            q_disk_radius=R,
            radial_lower=radial_lower,
            radial_upper=radial_upper,
            radial_n=quadrature.n_kappa,
            radial_method=quadrature.kappa_method,
            theta_method=quadrature.theta_method,
            theta_n=quadrature.n_theta,
            disk_curvature=disk_curvature,
            disk_factor=disk_factor,
            integral=integral,
            denominator_mode=denominator_mode,
            denominator_regulator=denominator_regulator,
            quadrature=quadrature,
            S_exact_time=S,
        )
        return S, details

    return S
