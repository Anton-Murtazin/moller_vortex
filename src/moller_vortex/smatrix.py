"""Impulse S-matrix assembly."""

from __future__ import annotations

import math

import numpy as np

from .accuracy import ACCURACY, NumericalAccuracy
from .constants import ELECTRON_CHARGE, ELECTRON_MASS, PI
from .kinematics import energy, helicity, kron_delta, vec2, vec3
from .packets import LGPacket, central_energy, normalization_constant
from .transverse import transverse_integral_explicit, transverse_integral_numeric_quad


def impulse_parameters(
    packet1: LGPacket,
    packet2: LGPacket,
    k3,
    k4,
    impact_b,
    m: float = ELECTRON_MASS,
) -> dict:
    """Return scalar parameters used by the closed impulse S-matrix formula."""
    packet1 = packet1.checked()
    packet2 = packet2.checked()

    k3 = vec3(k3)
    k4 = vec3(k4)
    b = vec2(impact_b)

    K_vec = k3 + k4
    K_perp = K_vec[:2]
    Kz = K_vec[2]

    K_vec_sq = np.dot(K_vec, K_vec)
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

    Xi0 = (
        (m * m + eps1 * eps1 - packet1.kbar_z ** 2) / (2.0 * s1z ** 2)
        + (m * m + eps2 * eps2 + K_vec_sq - (Kz - packet2.kbar_z) ** 2) / (2.0 * s2z ** 2)
        - K_perp_sq / (2.0 * s2p ** 2)
        + 1j * np.dot(b, K_perp)
        - eps1 * eps1 / s1z ** 2
        - eps2 * eps2 / s2z ** 2
        + packet1.kbar_z * (packet1.kbar_z / s1z ** 2 - packet2.kbar_z / s2z ** 2)
        - eps2 * v2 * DeltaKz / s2z ** 2
    )

    A_long = (
        packet1.kbar_z / s1z ** 2
        - packet2.kbar_z / s2z ** 2
        - v1 * eps1 / s1z ** 2
        + v2 * eps2 / s2z ** 2
    )

    Omega_long = eps1 + eps2 - E_K + v2 * DeltaKz

    alpha = -(
        1.0 / s1z ** 2
        + 1.0 / s2z ** 2
        - 1.0 / s1p ** 2
        - 1.0 / s2p ** 2
        - eps1 / (eps1 * s1z ** 2)
    )

    beta = 1.0 / s2p ** 2 - 1.0 / s2z ** 2
    gamma = eps2 / (eps2 * s2z ** 2)

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
    """Common factor multiplying the transverse integral in S_impulse_closed_form."""
    packet1 = packet1.checked()
    packet2 = packet2.checked()

    helicity_conserving = kron_delta(helicity(lam3), helicity(lam1)) and kron_delta(
        helicity(lam4), helicity(lam2)
    )
    if not helicity_conserving:
        return 0.0 + 0.0j, {"reason": "helicity delta is zero"}

    b = vec2(impact_b)
    accuracy = ACCURACY if accuracy is None else accuracy

    N1 = normalization_constant(packet1, m, accuracy=accuracy) if N1 is None else N1
    N2 = normalization_constant(packet2, m, accuracy=accuracy) if N2 is None else N2

    pars = impulse_parameters(packet1, packet2, k3, k4, b, m)

    L1 = abs(packet1.ell)
    L2 = abs(packet2.ell)

    packet_denominator = (
        packet1.sigma_perp ** L1
        * packet2.sigma_perp ** L2
        * np.sqrt(math.factorial(L1) * math.factorial(L2))
    )

    prefactor = (
        -2j
        * e_charge ** 2
        / (2.0 * PI) ** 4
        * np.sqrt(pars["E3"] * pars["E4"] / (pars["eps1"] * pars["eps2"]))
        * N1
        * N2
        / packet_denominator
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
    """Closed impulse-approximation S-matrix using the analytic transverse integral."""
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
    accuracy: NumericalAccuracy | None = None,
    return_details: bool = False,
) -> complex | tuple[complex, dict]:
    """S-matrix beyond strict impulse approximation.

    The function replaces the strict impulse longitudinal-time factor by the
    beyond-impulse time block while keeping all global constants consistent
    with S_impulse_closed_form.

    The formula is organized as

        S = prefactor * exp(Xi0_beyond) * L_beyond[I_perp(t)].

    The transverse integral is evaluated with

        alpha(t) = alpha0 - i t / eps1,
        beta(t)  = beta0,
        gamma(t) = gamma0 - i t / eps2.

    The longitudinal sign convention is fixed by the requirement that the
    old impulse result is recovered when

        b_long -> 0,
        gamma_i^{-2} -> 0,
        I_perp(t) -> I_perp(0).

    In that limit this function gives

        exp(Omega_IA * A_long / (v1 - v2)) / |v1 - v2|,

    exactly as S_impulse_closed_form does.
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
    k4 = vec3(k4)
    b_perp = vec2(impact_b)

    k3_perp = k3[:2]
    K_perp = details["K_perp"]

    eps1 = details["eps1"]
    eps2 = details["eps2"]

    gamma1 = eps1 / m
    gamma2 = eps2 / m

    DeltaKz = details["DeltaKz"]

    s2z = packet2.sigma_par

    # Beyond-impulse constant exponent.
    # This term must not be inserted into strict impulse approximation.
    Xi0_beyond = (
        details["Xi0"]
        - DeltaKz * DeltaKz / (2.0 * s2z * s2z * gamma2 * gamma2)
    )

    # Longitudinal parameters beyond strict impulse approximation.
    Omega = (
        details["Omega_long"]
        + DeltaKz * DeltaKz / (2.0 * eps2 * gamma2 * gamma2)
    )

    a_long = 0.5 * (
        1.0 / (eps1 * gamma1 * gamma1)
        + 1.0 / (eps2 * gamma2 * gamma2)
    )

    b_long = (
        eps1
        / (2.0 * packet1.sigma_par * packet1.sigma_par)
        * 1.0
        / (eps1 * gamma1 * gamma1)
        +
        eps2
        / (2.0 * packet2.sigma_par * packet2.sigma_par)
        * 1.0
        / (eps2 * gamma2 * gamma2)
    )

    c_long = (
        details["v1"]
        - details["v2"]
        - DeltaKz / (eps2 * gamma2 * gamma2)
    )

    # This is the linear coefficient appearing in the longitudinal Gaussian.
    # The time-kernel sign is chosen below so that the old impulse limit is
    # recovered with exp(+Omega_IA * A_long / c0).
    d_linear = (
        details["A_long"]
        + DeltaKz / (packet2.sigma_par * packet2.sigma_par * gamma2 * gamma2)
    )

    # Effective d entering the Gaussian time kernel.
    d_time = -d_linear

    alpha0 = details["alpha"]
    beta0 = details["beta"]
    gamma0 = details["gamma"]

    def I_perp_at_time(t):
        alpha_t = alpha0 - 1j * t / eps1
        gamma_t = gamma0 - 1j * t / eps2

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
        A0 = alpha0 + gamma0

        A_dot_abs = abs(1.0 / eps1 + 1.0 / eps2)
        gamma_dot_abs = abs(1.0 / eps2)

        time_scale_A = abs(A0) / A_dot_abs
        time_scale_gamma = abs(gamma0) / gamma_dot_abs

        time_step = time_step_scale * min(time_scale_A, time_scale_gamma)

    I_m2 = I_perp_at_time(-2.0 * time_step)
    I_m1 = I_perp_at_time(-1.0 * time_step)
    I0 = I_perp_at_time(0.0)
    I_p1 = I_perp_at_time(+1.0 * time_step)
    I_p2 = I_perp_at_time(+2.0 * time_step)

    I1 = (
        -I_p2
        + 8.0 * I_p1
        - 8.0 * I_m1
        + I_m2
    ) / (12.0 * time_step)

    I2 = (
        -I_p2
        + 16.0 * I_p1
        - 30.0 * I0
        + 16.0 * I_m1
        - I_m2
    ) / (12.0 * time_step * time_step)

    c2 = c_long * c_long

    # Longitudinal Gaussian factor.
    # With d_time = -d_linear this reproduces the old impulse factor:
    # exp(+Omega_IA * A_long / c0) / |c0|.
    longitudinal_exponent = (
        -b_long * Omega * Omega / c2
        -d_time * Omega / c_long
    )

    longitudinal_factor = np.exp(longitudinal_exponent) / abs(c_long)

    # Moments of the Gaussian time kernel for the transverse Taylor expansion.
    q_time = (
        2.0 * b_long * Omega / c2
        + d_time / c_long
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

    S = (
        details["prefactor"]
        * np.exp(Xi0_beyond)
        * longitudinal_factor
        * transverse_time_bracket
    )

    if return_details:
        details.update(
            dict(
                Xi0_impulse=details["Xi0"],
                Xi0_beyond=Xi0_beyond,
                Xi0_beyond_extra=Xi0_beyond - details["Xi0"],
                Omega_first=Omega,
                a_long_first=a_long,
                b_long_first=b_long,
                c_long_first=c_long,
                d_linear_first=d_linear,
                d_time_first=d_time,
                alpha0=alpha0,
                beta0=beta0,
                gamma0=gamma0,
                time_step=time_step,
                I0=I0,
                I1=I1,
                I2=I2,
                q_time=q_time,
                longitudinal_exponent=longitudinal_exponent,
                longitudinal_factor=longitudinal_factor,
                transverse_time_bracket=transverse_time_bracket,
                S_first_order=S,
            )
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
    """Same S-matrix factorization but with direct numerical transverse integration."""
    accuracy = ACCURACY if accuracy is None else accuracy

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
