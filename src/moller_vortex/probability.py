"""Differential probability for fixed transverse total momentum.

This module evaluates the phase-space integral over final momenta at fixed
K_perp = k3_perp + k4_perp in the impulse approximation.

The numerical scheme is deterministic:
    - Gauss-Legendre quadrature for k3_perp, k3z, k4z on finite intervals;
    - periodic uniform quadrature for the azimuthal angle of k3_perp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .accuracy import ACCURACY, NumericalAccuracy
from .constants import ELECTRON_CHARGE, ELECTRON_MASS, PI
from .packets import LGPacket, normalization_constant
from .smatrix import S_impulse_closed_form


@dataclass(frozen=True)
class ProbabilityQuadrature:
    """Quadrature parameters for the differential probability integral.

    k3_perp_range:
        Integration interval for rho = |k3_perp|.

    k3z_range, k4z_range:
        Integration intervals for the final longitudinal momenta.

    K_perp_range:
        Integration interval for K = |K_perp| in the final outer transverse
        integration.

    n_k3_perp, n_k3z, n_k4z:
        Numbers of Gauss-Legendre nodes for the corresponding finite intervals.

    n_phi:
        Number of uniformly spaced azimuthal nodes for the angle of k3_perp.

    n_K_perp:
        Number of Gauss-Legendre nodes for K = |K_perp|.

    n_K_phi:
        Number of uniformly spaced azimuthal nodes for the angle of K_perp.
    """

    k3_perp_range: tuple[float, float]
    k3z_range: tuple[float, float]
    k4z_range: tuple[float, float]
    K_perp_range: tuple[float, float]
    n_k3_perp: int
    n_phi: int
    n_k3z: int
    n_k4z: int
    n_K_perp: int
    n_K_phi: int

def legendre_nodes_and_weights(interval: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes and weights on a finite interval."""
    a, b = interval
    x, w = np.polynomial.legendre.leggauss(n)

    nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w

    return nodes, weights


def spin_averaged_s_abs2_impulse(
    k3: np.ndarray,
    k4: np.ndarray,
    packet1: LGPacket,
    packet2: LGPacket,
    *,
    impact_b: np.ndarray,
    N1: float,
    N2: float,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    accuracy: NumericalAccuracy | None = None,
    helicities: Iterable[float] = (-0.5, 0.5),
    explicit_spin_sum: bool = False,
) -> float:
    """Return the unpolarized spin average of |S_fi|^2 in impulse approximation.

    In the current impulse approximation,

        S_fi = S_0 delta_{lambda3,lambda1} delta_{lambda4,lambda2},

    and S_0 has no additional helicity dependence. Therefore

        (1/4) sum_{lambda1,lambda2} sum_{lambda3,lambda4} |S_fi|^2 = |S_0|^2.

    The fast branch evaluates one helicity-conserving amplitude. The explicit
    branch performs the literal 16-term sum and is useful only as a diagnostic.
    """
    accuracy = ACCURACY if accuracy is None else accuracy
    helicities = tuple(helicities)

    if not explicit_spin_sum:
        lam1 = helicities[0]
        lam2 = helicities[1]
        lam3 = lam1
        lam4 = lam2

        S = S_impulse_closed_form(
            k3,
            k4,
            packet1,
            packet2,
            lam1=lam1,
            lam2=lam2,
            lam3=lam3,
            lam4=lam4,
            impact_b=impact_b,
            N1=N1,
            N2=N2,
            m=m,
            e_charge=e_charge,
            accuracy=accuracy,
        )

        return float(abs(S) ** 2)

    total = 0.0

    for lam1 in helicities:
        for lam2 in helicities:
            for lam3 in helicities:
                for lam4 in helicities:
                    S = S_impulse_closed_form(
                        k3,
                        k4,
                        packet1,
                        packet2,
                        lam1=lam1,
                        lam2=lam2,
                        lam3=lam3,
                        lam4=lam4,
                        impact_b=impact_b,
                        N1=N1,
                        N2=N2,
                        m=m,
                        e_charge=e_charge,
                        accuracy=accuracy,
                    )
                    total += abs(S) ** 2

    return float(0.25 * total)


def diff_probability(
    K_perp,
    packet1: LGPacket,
    packet2: LGPacket,
    quadrature: ProbabilityQuadrature,
    *,
    impact_b,
    N1: float | None = None,
    N2: float | None = None,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    accuracy: NumericalAccuracy | None = None,
    helicities: Iterable[float] = (-0.5, 0.5),
    explicit_spin_sum: bool = False,
) -> float:
    """Compute the differential probability density at fixed K_perp.

    The implemented integral is

        w(K_perp) =
        int d^2 k3_perp dk3z dk4z
            [(1/4) sum_spins |S_fi|^2]
            / [(2*pi)^6 4 E3 E4],

    with

        k4_perp = K_perp - k3_perp.

    The transverse integration over k3_perp is done in polar coordinates,

        d^2 k3_perp = rho d rho d phi.

    The integration limits are supplied through ProbabilityQuadrature.
    """
    accuracy = ACCURACY if accuracy is None else accuracy

    K = np.asarray(K_perp, dtype=float)
    b = np.asarray(impact_b, dtype=float)

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    rho_nodes, rho_weights = legendre_nodes_and_weights(
        quadrature.k3_perp_range,
        quadrature.n_k3_perp,
    )
    z3_nodes, z3_weights = legendre_nodes_and_weights(
        quadrature.k3z_range,
        quadrature.n_k3z,
    )
    z4_nodes, z4_weights = legendre_nodes_and_weights(
        quadrature.k4z_range,
        quadrature.n_k4z,
    )

    phi_nodes = np.linspace(0.0, 2.0 * PI, quadrature.n_phi, endpoint=False)
    phi_weight = 2.0 * PI / quadrature.n_phi

    total = 0.0
    phase_space_const = 1.0 / (2.0 * PI) ** 6

    for rho, w_rho in zip(rho_nodes, rho_weights):
        for phi in phi_nodes:
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            k3x = rho * cos_phi
            k3y = rho * sin_phi

            k4x = K[0] - k3x
            k4y = K[1] - k3y
            k4_perp_sq = k4x * k4x + k4y * k4y

            for k3z, w_z3 in zip(z3_nodes, z3_weights):
                E3 = np.sqrt(m * m + rho * rho + k3z * k3z)

                for k4z, w_z4 in zip(z4_nodes, z4_weights):
                    E4 = np.sqrt(m * m + k4_perp_sq + k4z * k4z)

                    k3 = np.array([k3x, k3y, k3z], dtype=float)
                    k4 = np.array([k4x, k4y, k4z], dtype=float)

                    s_abs2 = spin_averaged_s_abs2_impulse(
                        k3,
                        k4,
                        packet1,
                        packet2,
                        impact_b=b,
                        N1=N1,
                        N2=N2,
                        m=m,
                        e_charge=e_charge,
                        accuracy=accuracy,
                        helicities=helicities,
                        explicit_spin_sum=explicit_spin_sum,
                    )

                    weight = w_rho * phi_weight * w_z3 * w_z4
                    measure = rho
                    phase_space = phase_space_const / (4.0 * E3 * E4)

                    total += weight * measure * phase_space * s_abs2

    return float(total)


def diff_probability_grid(
    Kx_values: np.ndarray,
    Ky_values: np.ndarray,
    packet1: LGPacket,
    packet2: LGPacket,
    quadrature: ProbabilityQuadrature,
    *,
    impact_b,
    N1: float | None = None,
    N2: float | None = None,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    accuracy: NumericalAccuracy | None = None,
    helicities: Iterable[float] = (-0.5, 0.5),
    explicit_spin_sum: bool = False,
) -> np.ndarray:
    """Compute diff_probability on a rectangular grid of Kx and Ky values."""
    accuracy = ACCURACY if accuracy is None else accuracy

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    values = np.empty((len(Ky_values), len(Kx_values)), dtype=float)

    for iy, Ky in enumerate(Ky_values):
        for ix, Kx in enumerate(Kx_values):
            K_perp = np.array([Kx, Ky], dtype=float)
            values[iy, ix] = diff_probability(
                K_perp,
                packet1,
                packet2,
                quadrature,
                impact_b=impact_b,
                N1=N1,
                N2=N2,
                m=m,
                e_charge=e_charge,
                accuracy=accuracy,
                helicities=helicities,
                explicit_spin_sum=explicit_spin_sum,
            )

    return values

def total_probability(
    packet1: LGPacket,
    packet2: LGPacket,
    quadrature: ProbabilityQuadrature,
    *,
    impact_b,
    N1: float | None = None,
    N2: float | None = None,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    accuracy: NumericalAccuracy | None = None,
    helicities: Iterable[float] = (-0.5, 0.5),
    explicit_spin_sum: bool = False,
) -> float:
    """Compute the total probability in the selected transverse K_perp domain.

    The remaining transverse integration is performed in polar coordinates,

        K_perp = K (cos phi_K, sin phi_K),
        d^2 K_perp = K dK dphi_K.

    The radial K integral is Gauss-Legendre. The angular integral is the
    periodic trapezoidal rule.
    """
    accuracy = ACCURACY if accuracy is None else accuracy

    b = np.asarray(impact_b, dtype=float)

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    K_nodes, K_weights = legendre_nodes_and_weights(
        quadrature.K_perp_range,
        quadrature.n_K_perp,
    )

    phi_nodes = np.linspace(0.0, 2.0 * PI, quadrature.n_K_phi, endpoint=False)
    phi_weight = 2.0 * PI / quadrature.n_K_phi

    total = 0.0

    for K, w_K in zip(K_nodes, K_weights):
        for phi_K in phi_nodes:
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)

            K_perp = np.array([Kx, Ky], dtype=float)

            w_value = diff_probability(
                K_perp,
                packet1,
                packet2,
                quadrature,
                impact_b=b,
                N1=N1,
                N2=N2,
                m=m,
                e_charge=e_charge,
                accuracy=accuracy,
                helicities=helicities,
                explicit_spin_sum=explicit_spin_sum,
            )

            total += w_K * phi_weight * K * w_value

    return float(total)


def Ky_average(
    packet1: LGPacket,
    packet2: LGPacket,
    quadrature: ProbabilityQuadrature,
    *,
    impact_b,
    N1: float | None = None,
    N2: float | None = None,
    m: float = ELECTRON_MASS,
    e_charge: float = ELECTRON_CHARGE,
    accuracy: NumericalAccuracy | None = None,
    helicities: Iterable[float] = (-0.5, 0.5),
    explicit_spin_sum: bool = False,
) -> float:
    """Compute <K_y> in the selected transverse K_perp domain.

    The implemented expression is

        <K_y> =
            int K_y w(K_perp) d^2 K_perp
            /
            int w(K_perp) d^2 K_perp.

    The integration is performed in polar coordinates using the K_perp
    quadrature parameters stored in ProbabilityQuadrature.
    """
    accuracy = ACCURACY if accuracy is None else accuracy

    b = np.asarray(impact_b, dtype=float)

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    K_nodes, K_weights = legendre_nodes_and_weights(
        quadrature.K_perp_range,
        quadrature.n_K_perp,
    )

    phi_nodes = np.linspace(0.0, 2.0 * PI, quadrature.n_K_phi, endpoint=False)
    phi_weight = 2.0 * PI / quadrature.n_K_phi

    probability = 0.0
    numerator = 0.0

    for K, w_K in zip(K_nodes, K_weights):
        for phi_K in phi_nodes:
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)

            K_perp = np.array([Kx, Ky], dtype=float)

            w_value = diff_probability(
                K_perp,
                packet1,
                packet2,
                quadrature,
                impact_b=b,
                N1=N1,
                N2=N2,
                m=m,
                e_charge=e_charge,
                accuracy=accuracy,
                helicities=helicities,
                explicit_spin_sum=explicit_spin_sum,
            )

            weight = w_K * phi_weight * K

            probability += weight * w_value
            numerator += weight * Ky * w_value

    return float(numerator / probability)