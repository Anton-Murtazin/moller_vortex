"""On-axis Laguerre-Gaussian packets and normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.integrate import quad
from scipy.special import gammaln, k0e, k1e

from .constants import ELECTRON_MASS, PI


NORMALIZATION_QUAD_EPSABS = 1.0e-16
NORMALIZATION_QUAD_EPSREL = 1.0e-10
NORMALIZATION_QUAD_LIMIT = 150


def _scaled_bessel_k_integer(order: int, argument: float) -> float:
    """Return exp(argument) K_order(argument) for a non-negative integer order."""
    if order < 0:
        raise ValueError("Bessel order must be non-negative.")
    if argument <= 0.0:
        return np.inf
    if order == 0:
        return k0e(argument)
    if order == 1:
        return k1e(argument)

    previous = k0e(argument)
    current = k1e(argument)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(1, order):
            previous, current = current, previous + 2.0 * n * current / argument
            if not np.isfinite(current):
                return current
    return current


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

    The on-axis packet used here is normalizable for
    ``sigma_perp <= sigma_par``. In coordinate coherence lengths this is
    ``L_perp >= L_par`` because ``sigma = hbar c / L``.
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
        if self.sigma_perp > self.sigma_par * (1.0 + 1.0e-14):
            raise ValueError(
                "This on-axis packet is normalizable only for momentum widths "
                "sigma_perp <= sigma_par. Equivalently, coordinate coherence "
                "lengths must satisfy L_perp >= L_par."
            )
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

    Dimensionless radial form.

    The integration variable is

        t = (k_perp / sigma_perp)^2.

    The measure factor is

        y^(2|ell|+1) dy / |ell|! = 0.5 t^|ell| dt / Gamma(|ell|+1),

    so the large factorial is absorbed directly into the exponent of the
    dimensionless integrand.
    """
    packet = packet.checked()

    ell_abs = abs(packet.ell)
    sigma_perp = packet.sigma_perp
    sigma_par = packet.sigma_par

    radial_coeff = 1.0 - sigma_perp ** 2 / sigma_par ** 2
    if radial_coeff < 0.0 and radial_coeff > -1.0e-14:
        radial_coeff = 0.0
    log_factorial = gammaln(ell_abs + 1)
    if radial_coeff > 1.0e-3:
        y_tail = max(12.0, 8.0 * np.sqrt((ell_abs + 1.0) / radial_coeff))
    else:
        y_tail = max(
            12.0,
            8.0 * np.sqrt(ell_abs + 1.0),
            (2.0 * ell_abs + 80.0) * sigma_par ** 2 / (m * sigma_perp),
        )
    t_upper = y_tail ** 2

    def log_integrand(t: float) -> float:
        if t == 0.0:
            log_power = 0.0 if ell_abs == 0 else -np.inf
        else:
            log_power = ell_abs * np.log(t) - log_factorial
        if not np.isfinite(log_power):
            return -np.inf

        k_perp = sigma_perp * np.sqrt(t)
        eps_perp = np.sqrt(m ** 2 + k_perp ** 2)
        eps_minus_m = k_perp ** 2 / (eps_perp + m)
        bessel_arg = 2.0 * m * eps_perp / sigma_par ** 2
        bessel_scaled = k0e(bessel_arg)

        if not np.isfinite(bessel_scaled) or bessel_scaled <= 0.0:
            raise FloatingPointError(
                "scipy.special.k0e is not finite in the normalization integral."
            )

        exponent = (
            -radial_coeff * t
            - 2.0 * m * eps_minus_m / sigma_par ** 2
        )
        return log_power + exponent + np.log(bessel_scaled)

    scale_nodes = np.linspace(0.0, y_tail, 257) ** 2
    scale_values = np.array([log_integrand(t) for t in scale_nodes])
    finite_scale_values = scale_values[np.isfinite(scale_values)]
    if finite_scale_values.size == 0:
        raise FloatingPointError("The normalization integrand is not finite.")
    log_scale = finite_scale_values.max()

    def integrand(t: float) -> float:
        log_value = log_integrand(t) - log_scale
        if not np.isfinite(log_value) or log_value < -745.0:
            return 0.0
        return np.exp(log_value)

    scaled_integral = quad(
        integrand,
        0.0,
        t_upper,
        epsabs=quad_epsabs,
        epsrel=quad_epsrel,
        limit=quad_limit,
    )[0]

    if not np.isfinite(scaled_integral) or scaled_integral <= 0.0:
        raise FloatingPointError("The normalization integral did not converge.")

    log_norm_without_N = (
        np.log(sigma_perp ** 2 / (8.0 * PI ** 2))
        + log_scale
        + np.log(scaled_integral)
    )
    return np.exp(-0.5 * log_norm_without_N)


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
    bessel_scaled = _scaled_bessel_k_integer(ell_abs + 1, argument)

    if not np.isfinite(bessel_scaled) or bessel_scaled <= 0.0:
        return normalization_constant(packet, m=m)

    return (
        2.0 ** 1.5
        * PI
        / (sigma * np.sqrt(bessel_scaled))
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
