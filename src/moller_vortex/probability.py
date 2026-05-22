"""Differential probability for fixed transverse total momentum.

This module evaluates the phase-space integral over final momenta at fixed
K_perp = k3_perp + k4_perp.

The numerical scheme is deterministic. Finite non-periodic probability axes use
composite Boole quadrature by default, while azimuthal angles use the
endpoint-free trapezoidal rule by default.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from .accuracy import NumericalAccuracy, resolve_accuracy
from .constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    FLOAT_DTYPE,
    PI,
    real_array,
    real_empty,
)
from .kinematics import vec2
from .packets import LGPacket, resolve_normalizations
from .quadrature import (
    ProbabilityQuadrature,
    probability_inner_nodes,
    probability_outer_nodes,
    probability_transverse_nodes,
)
from .smatrix import S_exact_time, S_impulse_closed_form, S_impulse_first_order


def _resolve_s_matrix(s_matrix: str | Callable):
    """Resolve a public S-matrix selector into a callable.

    Parameters
    ----------
    s_matrix:
        ``"closed"``, ``"first_order"``, ``"exact_time"``, or a callable
        with the same signature as the public S-matrix functions.

    Returns
    -------
    Callable
        The S-matrix function to evaluate.
    """
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


def _prepare_probability_inputs(
    packet1: LGPacket,
    packet2: LGPacket,
    *,
    impact_b,
    N1: float | None,
    N2: float | None,
    m: float,
    e_charge: float,
    accuracy: NumericalAccuracy | None,
    helicities: Iterable[float],
    explicit_spin_sum: bool,
    s_matrix: str | Callable,
    s_matrix_kwargs: dict | None,
):
    """Validate probability inputs and compute missing normalizations.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    impact_b:
        Transverse impact parameter of the second packet.
    N1, N2:
        Optional precomputed normalization constants.
    m, e_charge:
        Particle mass and electric charge in project units.
    accuracy:
        Optional numerical accuracy configuration.
    helicities:
        Helicity labels to include in spin averaging.
    explicit_spin_sum:
        Whether to evaluate all helicity channels explicitly.
    s_matrix, s_matrix_kwargs:
        S-matrix selector and optional extra keyword arguments.

    Returns
    -------
    dict
        Validated values passed to lower-level probability routines.
    """
    accuracy = resolve_accuracy(accuracy)
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
        accuracy=accuracy,
    )
    return dict(
        packet1=packet1,
        packet2=packet2,
        impact_b=vec2(impact_b),
        N1=N1,
        N2=N2,
        m=m,
        e_charge=e_charge,
        accuracy=accuracy,
        helicities=tuple(helicities),
        explicit_spin_sum=explicit_spin_sum,
        s_matrix=s_matrix,
        s_matrix_kwargs=s_matrix_kwargs,
    )


def spin_averaged_s_abs2(
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
    """Return the unpolarized spin average of ``|S_fi|^2``.

    Parameters
    ----------
    k3, k4:
        Final three-momenta.
    packet1, packet2:
        Incoming wave packets.
    impact_b:
        Transverse impact parameter of the second packet.
    N1, N2:
        Precomputed packet normalization constants.
    m, e_charge:
        Particle mass and electric charge in project units.
    accuracy:
        Numerical accuracy configuration passed to S-matrix routines.
    helicities:
        Iterable of helicity labels, normally ``(-0.5, 0.5)``.
    explicit_spin_sum:
        If True, compute the literal 16-term helicity sum.
    s_matrix:
        S-matrix selector: ``"closed"``, ``"first_order"``,
        ``"exact_time"``, or a compatible callable.
    s_matrix_kwargs:
        Extra keyword arguments for the selected S-matrix function.

    Returns
    -------
    np.float64
        Spin-averaged squared modulus of the selected S matrix.

    The fast branch evaluates one helicity-conserving amplitude.  The explicit
    branch performs the literal 16-term sum and is useful as a diagnostic.
    """
    accuracy = resolve_accuracy(accuracy)
    helicities = tuple(helicities)
    if len(helicities) != 2:
        raise ValueError("spin averaging expects exactly two helicity labels.")
    k3 = real_array(k3, shape=(3,))
    k4 = real_array(k4, shape=(3,))
    impact_b = real_array(impact_b, shape=(2,))
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

        return np.abs(S) ** 2

    total = FLOAT_DTYPE(0.0)

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
                    total = total + np.abs(S) ** 2

    return FLOAT_DTYPE(0.25) * total


def _diff_probability_resolved(
    K_perp,
    quadrature: ProbabilityQuadrature,
    *,
    packet1: LGPacket,
    packet2: LGPacket,
    impact_b,
    N1: float,
    N2: float,
    m: float,
    e_charge: float,
    accuracy: NumericalAccuracy,
    helicities: tuple[float, ...],
    explicit_spin_sum: bool,
    s_matrix: str | Callable,
    s_matrix_kwargs: dict | None,
) -> np.float64:
    """Compute ``w(K_perp)`` with already prepared inputs.

    Parameters
    ----------
    K_perp:
        Fixed total final transverse momentum.
    quadrature:
        Inner probability quadrature settings.
    packet1, packet2, impact_b, N1, N2, m, e_charge, accuracy, helicities,
    explicit_spin_sum, s_matrix, s_matrix_kwargs:
        Validated values from ``_prepare_probability_inputs``.

    Returns
    -------
    np.float64
        Differential probability density at the selected ``K_perp``.
    """
    K = vec2(K_perp)

    (
        (rho_nodes, rho_weights),
        (phi_nodes, phi_weights),
        (z3_nodes, z3_weights),
        (z4_nodes, z4_weights),
    ) = probability_inner_nodes(quadrature)

    total = FLOAT_DTYPE(0.0)
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

                    k3 = real_array([k3x, k3y, k3z], shape=(3,))
                    k4 = real_array([k4x, k4y, k4z], shape=(3,))
                    s_abs2 = spin_averaged_s_abs2(
                        k3,
                        k4,
                        packet1,
                        packet2,
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

                    weight = w_rho * w_phi * w_z3 * w_z4
                    measure = rho
                    phase_space = phase_space_const / (4.0 * E3 * E4)

                    total = total + weight * measure * phase_space * s_abs2

    return total


def _longitudinal_density_resolved(
    k3z: float,
    k4z: float,
    K_perp,
    quadrature: ProbabilityQuadrature,
    *,
    packet1: LGPacket,
    packet2: LGPacket,
    impact_b,
    N1: float,
    N2: float,
    m: float,
    e_charge: float,
    accuracy: NumericalAccuracy,
    helicities: tuple[float, ...],
    explicit_spin_sum: bool,
    s_matrix: str | Callable,
    s_matrix_kwargs: dict | None,
) -> np.float64:
    """Compute the longitudinal density at fixed ``k3z`` and ``k4z``.

    Parameters
    ----------
    k3z, k4z:
        Fixed final longitudinal momenta.
    K_perp:
        Fixed total final transverse momentum.
    quadrature:
        Probability quadrature settings for ``k3_perp`` and ``phi``.
    packet1, packet2, impact_b, N1, N2, m, e_charge, accuracy, helicities,
    explicit_spin_sum, s_matrix, s_matrix_kwargs:
        Validated values from ``_prepare_probability_inputs``.

    Returns
    -------
    np.float64
        Density ``d^2 w / (dk3z dk4z)`` at the selected point.
    """
    K = vec2(K_perp)
    (rho_nodes, rho_weights), (phi_nodes, phi_weights) = probability_transverse_nodes(
        quadrature
    )

    total = FLOAT_DTYPE(0.0)
    phase_space_const = 1.0 / (2.0 * PI) ** 6

    for rho, w_rho in zip(rho_nodes, rho_weights):
        for phi, w_phi in zip(phi_nodes, phi_weights):
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            k3x = rho * cos_phi
            k3y = rho * sin_phi
            k4x = K[0] - k3x
            k4y = K[1] - k3y

            E3 = np.sqrt(m * m + rho * rho + k3z * k3z)
            E4 = np.sqrt(m * m + k4x * k4x + k4y * k4y + k4z * k4z)

            k3 = real_array([k3x, k3y, k3z], shape=(3,))
            k4 = real_array([k4x, k4y, k4z], shape=(3,))
            s_abs2 = spin_averaged_s_abs2(
                k3,
                k4,
                packet1,
                packet2,
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

            weight = w_rho * w_phi
            measure = rho
            phase_space = phase_space_const / (4.0 * E3 * E4)
            total = total + weight * measure * phase_space * s_abs2

    return total


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

    Parameters
    ----------
    K_perp:
        Fixed total final transverse momentum as a two-vector.
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Inner integration ranges, node counts and methods.
    impact_b:
        Transverse impact parameter of the second packet.
    N1, N2:
        Optional precomputed normalization constants.
    m, e_charge:
        Particle mass and electric charge in project units.
    accuracy:
        Optional numerical accuracy configuration.
    helicities:
        Helicity labels used for spin averaging.
    explicit_spin_sum:
        Whether to compute the literal helicity sum.
    s_matrix, s_matrix_kwargs:
        S-matrix selector and optional extra keyword arguments.

    Returns
    -------
    np.float64
        Differential probability density.

    The implemented integral is

        w(K_perp) = int d^2 k3_perp dk3z dk4z
            [(1/4) sum_spins |S_fi|^2] / [(2*pi)^6 4 E3 E4],

    with k4_perp = K_perp - k3_perp.
    """
    inputs = _prepare_probability_inputs(
        packet1,
        packet2,
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
    return _diff_probability_resolved(K_perp, quadrature, **inputs)


def longitudinal_density(
    k3z: float,
    k4z: float,
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
    """Compute ``d^2 w / (dk3z dk4z)`` at one longitudinal point.

    Parameters
    ----------
    k3z, k4z:
        Fixed final longitudinal momenta in MeV.
    K_perp:
        Fixed total final transverse momentum as a two-vector.
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings for the remaining transverse integral.
    impact_b:
        Transverse impact parameter of the second packet.
    N1, N2:
        Optional precomputed normalization constants.
    m, e_charge:
        Particle mass and electric charge in project units.
    accuracy:
        Optional numerical accuracy configuration.
    helicities:
        Helicity labels used for spin averaging.
    explicit_spin_sum:
        Whether to compute the literal helicity sum.
    s_matrix, s_matrix_kwargs:
        S-matrix selector and optional extra keyword arguments.

    Returns
    -------
    np.float64
        Longitudinal density whose integral over ``k3z`` and ``k4z`` gives
        ``w(K_perp)``.
    """
    inputs = _prepare_probability_inputs(
        packet1,
        packet2,
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
    return _longitudinal_density_resolved(k3z, k4z, K_perp, quadrature, **inputs)


def longitudinal_density_grid(
    k3z_values: np.ndarray,
    k4z_values: np.ndarray,
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
    progress: bool = False,
) -> np.ndarray:
    """Compute ``longitudinal_density`` on a ``k3z``/``k4z`` grid.

    Parameters
    ----------
    k3z_values, k4z_values:
        Grid coordinates for the final longitudinal momenta.
    K_perp:
        Fixed total final transverse momentum as a two-vector.
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings for the remaining transverse integral.
    impact_b, N1, N2, m, e_charge, accuracy, helicities,
    explicit_spin_sum, s_matrix, s_matrix_kwargs:
        Same meaning as in ``longitudinal_density``.
    progress:
        If True, print one progress line after each completed ``k4z`` row.

    Returns
    -------
    np.ndarray
        Array with shape ``(len(k4z_values), len(k3z_values))``.
    """
    k3z_values = real_array(k3z_values)
    k4z_values = real_array(k4z_values)
    K_perp = vec2(K_perp)
    inputs = _prepare_probability_inputs(
        packet1,
        packet2,
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

    values = real_empty((len(k4z_values), len(k3z_values)))
    start_total = time.perf_counter()

    for i4, k4z in enumerate(k4z_values):
        start_row = time.perf_counter()

        for i3, k3z in enumerate(k3z_values):
            values[i4, i3] = _longitudinal_density_resolved(
                k3z,
                k4z,
                K_perp,
                quadrature,
                **inputs,
            )

        if progress:
            elapsed_row = time.perf_counter() - start_row
            elapsed_total = time.perf_counter() - start_total
            print(
                f"row {i4 + 1:3d} / {len(k4z_values):3d} completed | "
                f"k4z = {k4z:.12e} MeV | "
                f"row time = {elapsed_row:.2f} s | "
                f"total time = {elapsed_total:.2f} s",
                flush=True,
            )

    return values


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
    """Compute ``diff_probability`` on a rectangular ``Kx``/``Ky`` grid.

    Parameters
    ----------
    Kx_values, Ky_values:
        Grid coordinates for total transverse momentum.
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings.
    impact_b, N1, N2, m, e_charge, accuracy, helicities,
    explicit_spin_sum, s_matrix, s_matrix_kwargs:
        Same meaning as in ``diff_probability``.

    Returns
    -------
    np.ndarray
        Array with shape ``(len(Ky_values), len(Kx_values))``.
    """
    Kx_values = real_array(Kx_values)
    Ky_values = real_array(Ky_values)
    inputs = _prepare_probability_inputs(
        packet1,
        packet2,
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

    values = real_empty((len(Ky_values), len(Kx_values)))

    for iy, Ky in enumerate(Ky_values):
        for ix, Kx in enumerate(Kx_values):
            K_perp = real_array([Kx, Ky], shape=(2,))
            values[iy, ix] = _diff_probability_resolved(K_perp, quadrature, **inputs)

    return values


def _outer_K_moments(
    quadrature: ProbabilityQuadrature,
    inputs: dict,
) -> tuple[np.float64, np.float64]:
    """Return total probability and the ``K_y`` numerator.

    Parameters
    ----------
    quadrature:
        Probability quadrature settings including the outer ``K_perp`` axes.
    inputs:
        Validated values from ``_prepare_probability_inputs``.

    Returns
    -------
    tuple[np.float64, np.float64]
        Total probability and numerator for ``Ky_average``.
    """
    (K_nodes, K_weights), (phi_nodes, phi_weights) = probability_outer_nodes(quadrature)

    probability = FLOAT_DTYPE(0.0)
    ky_numerator = FLOAT_DTYPE(0.0)

    for K, w_K in zip(K_nodes, K_weights):
        for phi_K, w_phi_K in zip(phi_nodes, phi_weights):
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)
            K_perp = real_array([Kx, Ky], shape=(2,))

            w_value = _diff_probability_resolved(K_perp, quadrature, **inputs)
            weight = w_K * w_phi_K * K

            probability = probability + weight * w_value
            ky_numerator = ky_numerator + weight * Ky * w_value

    return probability, ky_numerator


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
    """Compute the total probability in the selected ``K_perp`` domain.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings including ``K_perp_range``.
    impact_b, N1, N2, m, e_charge, accuracy, helicities,
    explicit_spin_sum, s_matrix, s_matrix_kwargs:
        Same meaning as in ``diff_probability``.

    Returns
    -------
    np.float64
        Total probability over the configured outer domain.
    """
    inputs = _prepare_probability_inputs(
        packet1,
        packet2,
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
    probability, _ = _outer_K_moments(quadrature, inputs)
    return probability


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
    """Compute ``<K_y>`` in the selected transverse ``K_perp`` domain.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings including ``K_perp_range``.
    impact_b, N1, N2, m, e_charge, accuracy, helicities,
    explicit_spin_sum, s_matrix, s_matrix_kwargs:
        Same meaning as in ``diff_probability``.

    Returns
    -------
    np.float64
        Probability-weighted average final total transverse momentum ``K_y``.
    """
    inputs = _prepare_probability_inputs(
        packet1,
        packet2,
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
    probability, numerator = _outer_K_moments(quadrature, inputs)
    return numerator / probability
