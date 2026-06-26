"""On-axis Laguerre-Gaussian packets and normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from scipy.integrate import quad
from scipy.special import kve

from .constants import ELECTRON_MASS, PI


NORMALIZATION_QUAD_EPSABS = 1.0e-16
NORMALIZATION_QUAD_EPSREL = 1.0e-10
NORMALIZATION_QUAD_LIMIT = 150


@dataclass(frozen=True)
class LGPacket:
    """Physical parameters of one on-axis momentum-space LG packet.

    Parameters
    ----------
    ell:
        Integer orbital angular momentum.
    sigma_perp:
        Transverse momentum width in MeV.
    sigma_par:
        Longitudinal momentum width in MeV.
    kbar_z:
        Central longitudinal momentum in MeV.

    Returns
    -------
    LGPacket
        Immutable packet parameter container.  The packet is restricted to
        kbar_perp = 0. The normalization constant is not stored in the object;
        compute it explicitly with ``normalization_constant``.
    """

    ell: int
    sigma_perp: float
    sigma_par: float
    kbar_z: float

    def checked(self) -> "LGPacket":
        """Validate packet parameters and return the packet itself.

        Parameters
        ----------
        None

        Returns
        -------
        LGPacket
            The validated packet.
        """
        if not isinstance(self.ell, int):
            raise TypeError("ell must be an integer.")
        if self.sigma_perp <= 0.0 or self.sigma_par <= 0.0:
            raise ValueError("Packet widths must be positive.")
        if self.sigma_perp > self.sigma_par:
            raise ValueError("The normalization integral requires sigma_perp <= sigma_par.")
        return self


def central_energy(packet: LGPacket, m: float = ELECTRON_MASS) -> float:
    """Return the central on-shell energy of an on-axis packet.

    Parameters
    ----------
    packet:
        Incoming LG packet.
    m:
        Particle mass.

    Returns
    -------
    float
        ``sqrt(m**2 + packet.kbar_z**2)``.
    """
    packet = packet.checked()
    return np.sqrt(m ** 2 + packet.kbar_z ** 2)


def normalization_constant(
    packet: LGPacket,
    m: float = ELECTRON_MASS,
    *,
    quad_epsabs: float = NORMALIZATION_QUAD_EPSABS,
    quad_epsrel: float = NORMALIZATION_QUAD_EPSREL,
    quad_limit: int = NORMALIZATION_QUAD_LIMIT,
) -> float:
    """Compute the on-axis relativistic normalization constant N_ell.

    Parameters
    ----------
    packet:
        Packet whose normalization constant is computed.
    m:
        Particle mass.
    quad_epsabs, quad_epsrel, quad_limit:
        Parameters passed directly to ``scipy.integrate.quad`` for the
        adaptive one-dimensional normalization integral.

    Returns
    -------
    float
        Relativistic normalization constant.

    Stable form for narrow packets.

    The integration variable is

        y = k_perp / sigma_perp.

    This avoids large powers of k_perp and keeps the radial integral in a
    dimensionless variable.
    """
    packet = packet.checked()

    ell_abs = abs(packet.ell)
    sigma_perp = packet.sigma_perp
    sigma_par = packet.sigma_par

    radial_coeff = 1.0 - sigma_perp ** 2 / sigma_par ** 2

    def integrand(y: float) -> float:
        """Return the dimensionless radial normalization integrand.

        Parameters
        ----------
        y:
            Dimensionless radial variable ``k_perp / sigma_perp``.

        Returns
        -------
        float
            Integrand value for the one-dimensional normalization integral.
        """
        k_perp = sigma_perp * y
        eps_perp = np.sqrt(m ** 2 + k_perp ** 2)

        eps_minus_m = k_perp ** 2 / (eps_perp + m)
        bessel_arg = 2.0 * m * eps_perp / sigma_par ** 2

        exponent = (
            -radial_coeff * y ** 2
            -2.0 * m * eps_minus_m / sigma_par ** 2
        )

        return (
            y ** (2 * ell_abs + 1)
            * np.exp(exponent)
            * kve(0, bessel_arg)
        )

    integral = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=quad_epsabs,
        epsrel=quad_epsrel,
        limit=quad_limit,
    )[0]

    norm_without_N = (
        sigma_perp ** 2
        * integral
        / (
            4.0
            * PI ** 2
            * math.factorial(ell_abs)
        )
    )

    return 1.0 / np.sqrt(norm_without_N)


def spherical_normalization_constant(
    packet: LGPacket,
    m: float = ELECTRON_MASS,
) -> float:
    """Return the analytic normalization constant in the spherical limit.

    Parameters
    ----------
    packet:
        Packet with ``sigma_perp == sigma_par``.
    m:
        Particle mass.

    Returns
    -------
    float
        Closed Bessel-K normalization constant.
    """
    packet = packet.checked()

    if packet.sigma_perp != packet.sigma_par:
        raise ValueError("The spherical normalization formula requires sigma_perp = sigma_par.")

    ell_abs = abs(packet.ell)
    sigma = packet.sigma_perp
    argument = 2.0 * m ** 2 / sigma ** 2

    return (
        2.0 ** 1.5
        * PI
        / (sigma * np.sqrt(kve(ell_abs + 1, argument)))
    )


def resolve_normalizations(
    packet1: LGPacket,
    packet2: LGPacket,
    *,
    N1: float | None = None,
    N2: float | None = None,
    m: float = ELECTRON_MASS,
) -> tuple[float, float]:
    """Return explicit or newly computed normalization constants.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    N1, N2:
        Optional precomputed normalization constants.
    m:
        Particle mass.

    Returns
    -------
    tuple[float, float]
        Normalization constants ``(N1, N2)``.
    """
    if N1 is None:
        N1 = normalization_constant(packet1, m=m)
    if N2 is None:
        N2 = normalization_constant(packet2, m=m)
    return N1, N2
