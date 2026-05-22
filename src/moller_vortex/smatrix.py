"""Impulse S-matrix assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .accuracy import NumericalAccuracy, resolve_accuracy
from .constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    FLOAT_DTYPE,
    PI,
    complex_zero,
    real_array,
)
from .kinematics import energy, helicity, kron_delta, vec2, vec3
from .packets import LGPacket, central_energy, resolve_normalizations
from .transverse import transverse_integral_explicit, transverse_integral_numeric_quad
from .quadrature import ExactTimeQuadrature, exact_time_nodes


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
    return (
        packet1.sigma_perp ** L1
        * packet2.sigma_perp ** L2
        * np.sqrt(math.factorial(L1) * math.factorial(L2))
    )


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
    gamma = 1.0 / ( s2z * s2z)

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
    accuracy: NumericalAccuracy | None = None,
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
    accuracy:
        Numerical accuracy configuration for missing normalization constants.

    Returns
    -------
    tuple[complex, dict]
        Common complex factor and diagnostic details.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()

    if not _helicity_conserving(lam1, lam2, lam3, lam4):
        return complex_zero(), {"reason": "helicity delta is zero"}

    b = vec2(impact_b)
    accuracy = resolve_accuracy(accuracy)
    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
        accuracy=accuracy,
    )

    pars = impulse_parameters(packet1, packet2, k3, k4, b, m)

    prefactor = (
        -2j
        * e_charge ** 2
        / (2.0 * PI) ** 4
        * np.sqrt(pars["E3"] * pars["E4"] / (pars["eps1"] * pars["eps2"]))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
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
    accuracy: NumericalAccuracy | None = None,
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
    accuracy:
        Numerical accuracy configuration.
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
        accuracy=accuracy,
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
    mode: str,
    longitudinal: FirstOrderLongitudinal,
    I0: complex,
    I1: complex,
    I2: complex,
) -> tuple[complex, dict]:
    """Assemble the expanded or resummed first-order time block.

    Parameters
    ----------
    mode:
        ``"expanded"`` or ``"resummed"``.
    longitudinal:
        Longitudinal first-order parameters.
    I0, I1, I2:
        Transverse integral and its first two time derivatives at zero.

    Returns
    -------
    tuple[complex, dict]
        Time block and intermediate diagnostic values.
    """
    mode = _mode(mode, ("resummed", "expanded"), "time_mode")

    Omega = longitudinal.Omega
    a_long = longitudinal.a
    b_long = longitudinal.b
    c_long = longitudinal.c
    d_long = longitudinal.d
    c2 = c_long * c_long

    if mode == "expanded":
        time_block = (
            2.0
            * np.sqrt(np.pi)
            / c_long
            * (
                I0
                + (
                    2.0 * Omega * a_long / c2
                    - Omega * Omega * b_long / c2
                    - Omega * d_long / c_long
                )
                * I0
                + 1j
                * (
                    2.0 * Omega * b_long / c2
                    - 2.0 * a_long / c2
                    + d_long / c_long
                )
                * I1
                + b_long * I2 / c2
            )
        )

        return time_block, dict(
            longitudinal_exponent=None,
            longitudinal_factor=None,
            q_time=None,
            transverse_time_bracket=None,
        )

    longitudinal_exponent = (
        -b_long * Omega * Omega / c2
        -d_long * Omega / c_long
    )

    longitudinal_factor = (
        2.0
        * np.sqrt(np.pi)
        / c_long
        * np.exp(longitudinal_exponent)
    )

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
    time_mode: str = "resummed",
    time_step: float | None = None,
    time_step_scale: float = 1.0e-4,
    accuracy: NumericalAccuracy | None = None,
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
    time_mode:
        "resummed" keeps the longitudinal exponential factor

            exp(-b Omega^2 / c^2 - d Omega / c).

        "expanded" uses the strictly expanded first-order expression

            I0
            + (2 Omega a / c^2 - Omega^2 b / c^2 - Omega d / c) I0
            + i(2 Omega b / c^2 - 2a / c^2 + d / c) I1
            + b / c^2 I2.

    accuracy:
        Numerical accuracy configuration for missing normalization constants.
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
        accuracy=accuracy,
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
    time_block, time_details = _first_order_time_block(
        time_mode,
        longitudinal,
        I0,
        I1,
        I2,
    )

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

    base_factor = (
        details["prefactor"]
        / (2.0 * np.sqrt(np.pi))
        * np.exp(details["Xi0"] + xi0_localization)
    )

    S = base_factor * time_block

    if return_details:
        details.update(
            dict(
                time_mode=time_mode,
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
                base_factor=base_factor,
                S_first_order=S,
                **time_details,
            )
        )
        return S, details

    return S


def _vortex_monomial(vec: np.ndarray, ell: int):
    """Return a Cartesian vortex monomial.

    Parameters
    ----------
    vec:
        Transverse two-vector.
    ell:
        Orbital angular momentum integer.

    Returns
    -------
    complex
        ``z**ell`` for non-negative ``ell`` and ``conj(z)**abs(ell)``
        otherwise, where ``z = x + i y``.
    """
    z = vec[0] + 1j * vec[1]
    if ell >= 0:
        return z ** ell
    return np.conjugate(z) ** (-ell)


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
    accuracy: NumericalAccuracy | None = None,
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
        ``"expanded"`` for the first-order Moller denominator expansion or
        ``"exact"`` for the regulated exact denominator.
    denominator_regulator:
        Regulator added in exact-denominator mode.
    accuracy:
        Numerical accuracy configuration for missing normalization constants.
    return_details:
        If True, return ``(S, details)``.

    Returns
    -------
    complex or tuple[complex, dict]
        Exact-time S-matrix value, optionally with diagnostic details.

    This implements the exact-time delta representation with

        q = q0 + R sin(chi) (cos theta, sin theta).

    The exponent is evaluated through the algebraically combined
    ``Xi0 + Xi_perp`` form to avoid artificial cancellations.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    accuracy = resolve_accuracy(accuracy)
    quadrature = ExactTimeQuadrature() if quadrature is None else quadrature
    denominator_mode = _mode(denominator_mode, ("expanded", "exact"), "denominator_mode")

    if not _helicity_conserving(lam1, lam2, lam3, lam4):
        S_zero = complex_zero()
        details = {"reason": "helicity delta is zero"}
        return (S_zero, details) if return_details else S_zero

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
        accuracy=accuracy,
    )

    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(E3 * E4 / (eps1 * eps2))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
    )

    (chi_nodes, chi_weights), (theta_nodes, theta_weights) = exact_time_nodes(quadrature)

    integral = complex_zero()

    for chi, w_chi in zip(chi_nodes, chi_weights):
        sin_chi = np.sin(chi)
        cos_chi = np.cos(chi)
        rho = R * sin_chi

        xi_plus = (-C_z + sqrt_Delta0 * cos_chi) / (2.0 * A_z)
        xi_minus = (-C_z - sqrt_Delta0 * cos_chi) / (2.0 * A_z)

        root_weight = (
            np.exp(-B_z * xi_plus * xi_plus + D_z * xi_plus)
            + np.exp(-B_z * xi_minus * xi_minus + D_z * xi_minus)
        )

        radial_weight = w_chi * sin_chi

        for theta, w_theta in zip(theta_nodes, theta_weights):
            n = real_array([np.cos(theta), np.sin(theta)], shape=(2,))
            q = q0 + rho * n
            K_minus_q = K_perp - q

            if denominator_mode == "expanded":
                if k3_perp_sq == 0.0:
                    raise ValueError("Expanded denominator requires nonzero |k3_perp|.")
                denominator_factor = (
                    1.0 / k3_perp_sq
                    * (1.0 + 2.0 * np.dot(q, k3_perp) / k3_perp_sq)
                )
            else:
                diff = k3_perp - q
                denominator_factor = 1.0 / (
                    np.dot(diff, diff) + denominator_regulator * denominator_regulator
                )

            transverse_exp = np.exp(
                -0.5 * np.dot(q, q) / (s1p * s1p)
                -0.5 * np.dot(K_minus_q, K_minus_q) / (s2p * s2p)
                + 1j * np.dot(b_perp, K_minus_q)
            )

            vortex_factor = (
                _vortex_monomial(q, packet1.ell)
                * _vortex_monomial(K_minus_q, packet2.ell)
            )

            A_perp = denominator_factor * transverse_exp * vortex_factor
            integral += w_theta * radial_weight * A_perp * root_weight

    disk_factor = sqrt_Delta0 / (2.0 * A_z * eta_perp)
    S = prefactor * np.exp(longitudinal_exp_arg) * disk_factor * integral

    if return_details:
        details = dict(
            N1=N1,
            N2=N2,
            impact_b=b_perp,
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
            disk_factor=disk_factor,
            integral=integral,
            denominator_mode=denominator_mode,
            denominator_regulator=denominator_regulator,
            quadrature=quadrature,
            S_exact_time=S,
        )
        return S, details

    return S


def S_impulse_numeric_transverse_quad(
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
    n_phi: int = 64,
    accuracy: NumericalAccuracy | None = None,
    return_details: bool = False,
) -> complex | tuple[complex, dict]:
    """Compute the impulse S matrix with numerical transverse integration.

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
    n_phi:
        Number of azimuthal nodes in the diagnostic transverse integration.
    accuracy:
        Numerical accuracy configuration for adaptive radial integrals.
    return_details:
        If True, return ``(S, details)``.

    Returns
    -------
    complex or tuple[complex, dict]
        S-matrix value computed with numerical transverse integration.
    """
    accuracy = resolve_accuracy(accuracy)

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
        accuracy=accuracy,
    )

    if "reason" in details:
        return (common_factor, details) if return_details else common_factor

    k3_perp = vec3(k3)[:2]
    b = vec2(impact_b)

    Iperp_numeric = transverse_integral_numeric_quad(
        packet1.ell,
        packet2.ell,
        k3_perp,
        details["K_perp"],
        b,
        details["alpha"],
        details["beta"],
        details["gamma"],
        n_phi=n_phi,
        accuracy=accuracy,
    )

    S = common_factor * Iperp_numeric

    if return_details:
        details.update(dict(Iperp_numeric=Iperp_numeric))
        return S, details

    return S
