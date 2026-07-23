"""First-order impulse correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ELECTRON_CHARGE, ELECTRON_MASS
from .kinematics import vec2, vec3
from .packets import LGPacket
from .smatrix_common import (
    _impulse_prefactor,
    _impulse_prefactor_grid,
)
from .transverse import transverse_integral, transverse_integral_grid


@dataclass(frozen=True)
class _LongitudinalBlock:
    """Longitudinal parameters in the first-order time block."""

    gamma1: float
    gamma2: float
    delta_kz: float
    Omega: float
    a: float
    b: float
    c: float
    d: float


def _longitudinal_block(
    packet1: LGPacket,
    packet2: LGPacket,
    details: dict,
    m: float,
) -> _LongitudinalBlock:
    """Return the longitudinal scalars used by ``S_first_order``.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    details:
        Dictionary produced by ``_impulse_prefactor``.
    m:
        Particle mass.

    Returns
    -------
    _LongitudinalBlock
        Grouped longitudinal parameters for the first-order time block.
    """
    eps1 = details["eps1"]
    eps2 = details["eps2"]
    gamma1 = eps1 / m
    gamma2 = eps2 / m
    delta_kz = details["DeltaKz"]
    inv_s1z2 = 1.0 / packet1.sigma_par ** 2
    inv_s2z2 = 1.0 / packet2.sigma_par ** 2

    Omega = (
        details["Omega_long"]
        + delta_kz ** 2 / (2.0 * eps2 * gamma2 ** 2)
    )

    a_long = 0.5 * (
        1.0 / (eps1 * gamma1 ** 2)
        + 1.0 / (eps2 * gamma2 ** 2)
    )

    b_long = (
        0.5 * inv_s1z2 / gamma1 ** 2
        + 0.5 * inv_s2z2 / gamma2 ** 2
    )

    c_long = (
        details["v1"]
        - details["v2"]
        - delta_kz / (eps2 * gamma2 ** 2)
    )

    d_long = (
        details["A_long"]
        + delta_kz * inv_s2z2 / gamma2 ** 2
    )

    return _LongitudinalBlock(
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
    ) / (12.0 * h ** 2)

    return I0, I1, I2


def _first_order_time_block(
    longitudinal: _LongitudinalBlock,
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
    c2 = c_long ** 2

    longitudinal_exponent = (
        -b_long * Omega ** 2 / c2
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
            - 0.5 * q_time ** 2
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


def S_first_order(
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

    k3 = vec3(k3)
    b_perp = vec2(impact_b)

    k3_perp = k3[:2]
    K_perp = details["K_perp"]

    longitudinal = _longitudinal_block(packet1, packet2, details, m)

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

        return transverse_integral(
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


def S_first_order_grid(
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
):
    """Vectorized first-order impulse S matrix for broadcastable momenta."""
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

    longitudinal = _longitudinal_block(packet1, packet2, details, m)

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
        transverse_integral_grid(
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
