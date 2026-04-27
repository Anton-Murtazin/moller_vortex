"""On-axis Laguerre-Gaussian packets and normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from .accuracy import ACCURACY, NumericalAccuracy
from .constants import ELECTRON_MASS, PI
from .kinematics import energy, vec2, vec3


@dataclass(frozen=True)
class LGPacket:
    """Physical parameters of one on-axis momentum-space LG packet.

    The packet is restricted to kbar_perp = 0. The normalization constant is
    not stored in the object; compute it explicitly with normalization_constant.
    """

    ell: int
    sigma_perp: float
    sigma_par: float
    kbar_z: float

    def checked(self) -> "LGPacket":
        if not isinstance(self.ell, int):
            raise TypeError("ell must be an integer.")
        if self.sigma_perp <= 0.0 or self.sigma_par <= 0.0:
            raise ValueError("Packet widths must be positive.")
        if self.sigma_perp > self.sigma_par:
            raise ValueError("The on-axis normalization integral requires sigma_perp <= sigma_par.")
        return self


def central_energy(packet: LGPacket, m: float = ELECTRON_MASS) -> float:
    """Central energy eps = sqrt(m^2 + kbar_z^2) for kbar_perp = 0."""
    packet = packet.checked()
    return np.sqrt(m * m + packet.kbar_z ** 2)


def normalization_constant(
    packet: LGPacket,
    m: float = ELECTRON_MASS,
    accuracy: NumericalAccuracy | None = None,
) -> float:
    """Compute the on-axis relativistic normalization constant N_ell.

    The implementation uses a stable form of the product

        exp(2*m**2/sigma_par**2) * K_0(2*m*sqrt(m**2+k_perp**2)/sigma_par**2).

    This avoids overflow for narrow packets.
    """
    from scipy.integrate import quad
    from scipy.special import kve

    packet = packet.checked()
    accuracy = ACCURACY if accuracy is None else accuracy

    ell_abs = abs(packet.ell)
    sigma_perp = packet.sigma_perp
    sigma_par = packet.sigma_par

    radial_alpha = 1.0 / sigma_perp ** 2 - 1.0 / sigma_par ** 2

    def integrand(k_perp: float) -> float:
        eps_perp = np.sqrt(m * m + k_perp * k_perp)
        bessel_arg = 2.0 * m * eps_perp / sigma_par ** 2

        stable_exponent = (
            -radial_alpha * k_perp * k_perp
            -2.0 * m * (eps_perp - m) / sigma_par ** 2
        )

        return (
            k_perp ** (2 * ell_abs + 1)
            * np.exp(stable_exponent)
            * kve(0, bessel_arg)
        )

    integral = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=accuracy.quad_epsabs,
        epsrel=accuracy.quad_epsrel,
        limit=accuracy.quad_limit,
    )[0]

    coefficient = 1.0 / (
        4.0
        * PI ** 2
        * sigma_perp ** (2 * ell_abs)
        * math.factorial(ell_abs)
    )

    return 1.0 / np.sqrt(coefficient * integral)


def spherical_normalization_constant(packet: LGPacket, m: float = ELECTRON_MASS) -> float:
    """Closed N_ell for sigma_perp = sigma_par.

    Stable version using kve instead of kv.
    """
    from scipy.special import kve

    packet = packet.checked()

    if packet.sigma_perp != packet.sigma_par:
        raise ValueError("The spherical normalization formula requires sigma_perp = sigma_par.")

    sigma = packet.sigma_perp
    ell_abs = abs(packet.ell)
    argument = 2.0 * m * m / sigma ** 2

    return (
        2.0 ** 1.5
        * PI
        / (sigma * np.sqrt(kve(ell_abs + 1, argument)))
    )


def lg_packet_phi(
    k,
    packet: LGPacket,
    N: float,
    m: float = ELECTRON_MASS,
    impact_b=(0.0, 0.0),
) -> complex:
    """Value of the on-axis LG packet in momentum space.

    impact_b is included only when this function is deliberately used for the
    displaced second incoming packet.
    """
    packet = packet.checked()
    k = vec3(k)
    b = vec2(impact_b)

    k_perp = k[:2]
    k_perp_abs = np.linalg.norm(k_perp)
    phi = np.arctan2(k_perp[1], k_perp[0]) if k_perp_abs != 0.0 else 0.0

    ell_abs = abs(packet.ell)
    E = energy(k, m)
    Ebar = central_energy(packet, m)

    exponent = (
        (E - Ebar) ** 2 / (2.0 * packet.sigma_par ** 2)
        - k_perp_abs ** 2 / (2.0 * packet.sigma_perp ** 2)
        - (k[2] - packet.kbar_z) ** 2 / (2.0 * packet.sigma_par ** 2)
        + 1j * packet.ell * phi
        + 1j * np.dot(b, k_perp)
    )

    prefactor = N * k_perp_abs ** ell_abs
    prefactor /= packet.sigma_perp ** ell_abs * np.sqrt(math.factorial(ell_abs))

    return prefactor * np.exp(exponent)
