"""Shared formulas for packet S-matrix assembly."""

from __future__ import annotations

import math

import numpy as np

from .amplitudes import massive_spinor_scale
from .constants import (
    COMPLEX_DTYPE,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    FLOAT_DTYPE,
    PI,
    complex_zero,
)
from .kinematics import helicity, kron_delta, vec2, vec3
from .packets import LGPacket, central_energy, resolve_normalizations


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
    """Validate a string selector.

    Parameters
    ----------
    value:
        Selector value.
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


def _spinor_scale(
    model: str,
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
    """Return the t-channel numerator scale for the selected spinor model."""
    model = _mode(
        model,
        ("ultrarelativistic", "massive"),
        "model",
    )
    if model == "ultrarelativistic":
        return 1.0 if _helicity_conserving(lam1, lam2, lam3, lam4) else 0.0

    return massive_spinor_scale(
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


# ---------------------------------------------------------------------------
# Analytic ultrarelativistic S-impulse.


def _impulse_parameters(
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
    """Return impulse kinematic scalars for scalar or broadcastable momenta."""
    b = vec2(impact_b)
    Kx = k3x + k4x
    Ky = k3y + k4y
    Kz = k3z + k4z
    K_perp_sq = Kx ** 2 + Ky ** 2

    E3 = np.sqrt(m ** 2 + k3x ** 2 + k3y ** 2 + k3z ** 2)
    E4 = np.sqrt(m ** 2 + k4x ** 2 + k4y ** 2 + k4z ** 2)
    E_K = E3 + E4

    eps1 = central_energy(packet1, m)
    eps2 = central_energy(packet2, m)
    v1 = packet1.kbar_z / eps1
    v2 = packet2.kbar_z / eps2
    if v1 == v2:
        raise ValueError("The longitudinal impulse integral requires v1 != v2.")

    DeltaKz = Kz - packet1.kbar_z - packet2.kbar_z

    s1p = packet1.sigma_perp
    s2p = packet2.sigma_perp
    s2z = packet2.sigma_par
    inv_s1p2 = 1.0 / s1p ** 2
    inv_s2p2 = 1.0 / s2p ** 2
    inv_s2z2 = 1.0 / s2z ** 2

    Xi0 = 0.5 * K_perp_sq * (inv_s2z2 - inv_s2p2) + 1j * (b[0] * Kx + b[1] * Ky)
    A_long = FLOAT_DTYPE(0.0)
    Omega_long = eps1 + eps2 - E_K + v2 * DeltaKz

    alpha = inv_s1p2 + inv_s2p2 - inv_s2z2
    beta = inv_s2p2 - inv_s2z2
    gamma = inv_s2z2

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


def _apply_impulse_prefactor(
    pars: dict,
    packet1: LGPacket,
    packet2: LGPacket,
    *,
    impact_b,
    N1: float,
    N2: float,
    e_charge: float,
) -> tuple[complex, dict]:
    """Attach the common S-matrix prefactor to impulse parameters."""
    prefactor = (
        -1j
        * e_charge ** 2
        / (PI * (2.0 * PI) ** 4)
        * np.sqrt(pars["E3"] * pars["E4"] / (pars["eps1"] * pars["eps2"]))
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
    )
    longitudinal_factor = 1.0 / abs(pars["v1"] - pars["v2"])
    common_factor = prefactor * np.exp(pars["Xi0"]) * longitudinal_factor

    details = dict(pars)
    details.update(
        dict(
            N1=N1,
            N2=N2,
            impact_b=impact_b,
            prefactor=prefactor,
            longitudinal_factor=longitudinal_factor,
            common_factor=common_factor,
        )
    )
    return common_factor, details


def _impulse_prefactor(
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
) -> tuple[complex, dict]:
    """Return the scalar impulse prefactor outside the transverse integral."""
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    if not _helicity_conserving(lam1, lam2, lam3, lam4):
        return complex_zero(), {"reason": "helicity delta is zero"}

    b = vec2(impact_b)
    N1, N2 = resolve_normalizations(packet1, packet2, N1=N1, N2=N2, m=m)
    k3 = vec3(k3)
    k4 = vec3(k4)
    pars = _impulse_parameters(
        k3[0],
        k3[1],
        k3[2],
        k4[0],
        k4[1],
        k4[2],
        packet1,
        packet2,
        impact_b=b,
        m=m,
    )
    K_vec = k3 + k4
    pars["K_vec"] = K_vec
    pars["K_perp"] = K_vec[:2]
    return _apply_impulse_prefactor(
        pars,
        packet1,
        packet2,
        impact_b=b,
        N1=N1,
        N2=N2,
        e_charge=e_charge,
    )


def _impulse_prefactor_grid(
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
) -> tuple[np.ndarray, dict]:
    """Return the vectorized impulse prefactor outside transverse integrals."""
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    if not _helicity_conserving(lam1, lam2, lam3, lam4):
        shape = np.broadcast(k3x, k3y, k3z, k4x, k4y, k4z).shape
        return np.zeros(shape, dtype=COMPLEX_DTYPE), {
            "reason": "helicity delta is zero"
        }
    b = vec2(impact_b)
    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )

    pars = _impulse_parameters(
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
    return _apply_impulse_prefactor(
        pars,
        packet1,
        packet2,
        impact_b=b,
        N1=N1,
        N2=N2,
        e_charge=e_charge,
    )
