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
