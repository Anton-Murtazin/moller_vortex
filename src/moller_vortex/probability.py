"""Differential probability for fixed transverse total momentum.

This module evaluates the phase-space integral over final momenta at fixed
K_perp = k3_perp + k4_perp.

The numerical scheme is deterministic.  The finite non-periodic integrals use
Boole quadrature by default, while azimuthal angles use the endpoint-free
trapezoidal rule by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .accuracy import ACCURACY, NumericalAccuracy
from .constants import ELECTRON_CHARGE, ELECTRON_MASS, PI
from .packets import LGPacket, normalization_constant
from .quadrature import nodes_and_weights
from .smatrix import S_exact_time, S_impulse_closed_form, S_impulse_first_order


@dataclass(frozen=True)
class ProbabilityQuadrature:
    """Quadrature parameters for the differential probability integral.

    The ``*_method`` fields are dispatched by ``quadrature.nodes_and_weights``.
    Recommended defaults are Boole for finite non-periodic variables and
    endpoint-free trapezoid for azimuthal angles.
    """

    k3_perp_range: tuple[float, float] | None = None
    k3z_range: tuple[float, float] | None = None
    k4z_range: tuple[float, float] | None = None
    K_perp_range: tuple[float, float] | None = None
    n_k3_perp: int | None = None
    n_phi: int | None = None
    n_k3z: int | None = None
    n_k4z: int | None = None
    n_K_perp: int | None = None
    n_K_phi: int | None = None
    k3_perp_method: str = "boole"
    k3z_method: str = "boole"
    k4z_method: str = "boole"
    K_perp_method: str = "boole"
    phi_method: str = "trapezoid"
    K_phi_method: str = "trapezoid"


def _resolve_s_matrix(s_matrix: str | Callable):
    """Resolve a public S-matrix selector into a callable."""
    if callable(s_matrix):
        return s_matrix
    if s_matrix == "closed":
        return S_impulse_closed_form
    if s_matrix == "first_order":
        return S_impulse_first_order
    if s_matrix == "exact_time":
        return S_exact_time
    raise ValueError(
        "s_matrix must be 'closed', 'first_order', 'exact_time', or a callable."
    )


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
    s_matrix: str | Callable = "closed",
    s_matrix_kwargs: dict | None = None,
) -> np.float64:
    """Return the unpolarized spin average of |S_fi|^2.

    The fast branch evaluates one helicity-conserving amplitude.  The explicit
    branch performs the literal 16-term sum and is useful as a diagnostic.
    """
    accuracy = ACCURACY if accuracy is None else accuracy
    helicities = tuple(helicities)
    S_function = _resolve_s_matrix(s_matrix)
    kwargs = {} if s_matrix_kwargs is None else dict(s_matrix_kwargs)
    kwargs.pop("return_details", None)

    if not explicit_spin_sum:
        lam1 = helicities[0]
        lam2 = helicities[1]
        lam3 = lam1
        lam4 = lam2

        S = S_function(
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
            **kwargs,
        )

        return np.real(np.abs(S) ** 2)

    total = np.float64(0.0)

    for lam1 in helicities:
        for lam2 in helicities:
            for lam3 in helicities:
                for lam4 in helicities:
                    S = S_function(
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
                        **kwargs,
                    )
                    total = total + np.real(np.abs(S) ** 2)

    return np.float64(0.25) * total


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
    s_matrix: str | Callable = "closed",
    s_matrix_kwargs: dict | None = None,
) -> np.float64:
    """Compute the differential probability density at fixed K_perp.

    The implemented integral is

        w(K_perp) = int d^2 k3_perp dk3z dk4z
            [(1/4) sum_spins |S_fi|^2] / [(2*pi)^6 4 E3 E4],

    with k4_perp = K_perp - k3_perp.
    """
    accuracy = ACCURACY if accuracy is None else accuracy

    K = np.asarray(K_perp, dtype=np.float64)
    b = np.asarray(impact_b, dtype=np.float64)

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    rho_nodes, rho_weights = nodes_and_weights(
        quadrature.k3_perp_range,
        quadrature.n_k3_perp,
        method=quadrature.k3_perp_method,
    )
    z3_nodes, z3_weights = nodes_and_weights(
        quadrature.k3z_range,
        quadrature.n_k3z,
        method=quadrature.k3z_method,
    )
    z4_nodes, z4_weights = nodes_and_weights(
        quadrature.k4z_range,
        quadrature.n_k4z,
        method=quadrature.k4z_method,
    )
    phi_nodes, phi_weights = nodes_and_weights(
        (0.0, 2.0 * PI),
        quadrature.n_phi,
        method=quadrature.phi_method,
        endpoint=False,
    )

    total = np.float64(0.0)
    phase_space_const = 1.0 / (2.0 * PI) ** 6

    for rho, w_rho in zip(rho_nodes, rho_weights):
        for phi, w_phi in zip(phi_nodes, phi_weights):
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

                    k3 = np.array([k3x, k3y, k3z], dtype=np.float64)
                    k4 = np.array([k4x, k4y, k4z], dtype=np.float64)

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
                        s_matrix=s_matrix,
                        s_matrix_kwargs=s_matrix_kwargs,
                    )

                    weight = w_rho * w_phi * w_z3 * w_z4
                    measure = rho
                    phase_space = phase_space_const / (4.0 * E3 * E4)

                    total = total + weight * measure * phase_space * s_abs2

    return total


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
    s_matrix: str | Callable = "closed",
    s_matrix_kwargs: dict | None = None,
) -> np.ndarray:
    """Compute diff_probability on a rectangular grid of Kx and Ky values."""
    accuracy = ACCURACY if accuracy is None else accuracy

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    values = np.empty((len(Ky_values), len(Kx_values)), dtype=np.float64)

    for iy, Ky in enumerate(Ky_values):
        for ix, Kx in enumerate(Kx_values):
            K_perp = np.array([Kx, Ky], dtype=np.float64)
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
                s_matrix=s_matrix,
                s_matrix_kwargs=s_matrix_kwargs,
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
    s_matrix: str | Callable = "closed",
    s_matrix_kwargs: dict | None = None,
) -> np.float64:
    """Compute the total probability in the selected transverse K_perp domain."""
    accuracy = ACCURACY if accuracy is None else accuracy

    b = np.asarray(impact_b, dtype=np.float64)

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    K_nodes, K_weights = nodes_and_weights(
        quadrature.K_perp_range,
        quadrature.n_K_perp,
        method=quadrature.K_perp_method,
    )
    phi_nodes, phi_weights = nodes_and_weights(
        (0.0, 2.0 * PI),
        quadrature.n_K_phi,
        method=quadrature.K_phi_method,
        endpoint=False,
    )

    total = np.float64(0.0)

    for K, w_K in zip(K_nodes, K_weights):
        for phi_K, w_phi_K in zip(phi_nodes, phi_weights):
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)

            K_perp = np.array([Kx, Ky], dtype=np.float64)

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
                s_matrix=s_matrix,
                s_matrix_kwargs=s_matrix_kwargs,
            )

            total = total + w_K * w_phi_K * K * w_value

    return total


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
    s_matrix: str | Callable = "closed",
    s_matrix_kwargs: dict | None = None,
) -> np.float64:
    """Compute <K_y> in the selected transverse K_perp domain."""
    accuracy = ACCURACY if accuracy is None else accuracy

    b = np.asarray(impact_b, dtype=np.float64)

    if N1 is None:
        N1 = normalization_constant(packet1, m=m, accuracy=accuracy)

    if N2 is None:
        N2 = normalization_constant(packet2, m=m, accuracy=accuracy)

    K_nodes, K_weights = nodes_and_weights(
        quadrature.K_perp_range,
        quadrature.n_K_perp,
        method=quadrature.K_perp_method,
    )
    phi_nodes, phi_weights = nodes_and_weights(
        (0.0, 2.0 * PI),
        quadrature.n_K_phi,
        method=quadrature.K_phi_method,
        endpoint=False,
    )

    probability = np.float64(0.0)
    numerator = np.float64(0.0)

    for K, w_K in zip(K_nodes, K_weights):
        for phi_K, w_phi_K in zip(phi_nodes, phi_weights):
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)

            K_perp = np.array([Kx, Ky], dtype=np.float64)

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
                s_matrix=s_matrix,
                s_matrix_kwargs=s_matrix_kwargs,
            )

            weight = w_K * w_phi_K * K

            probability = probability + weight * w_value
            numerator = numerator + weight * Ky * w_value

    return numerator / probability
