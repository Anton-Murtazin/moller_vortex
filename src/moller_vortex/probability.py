"""Differential probability for fixed transverse total momentum.

This module evaluates the phase-space integral over final momenta at fixed
K_perp = k3_perp + k4_perp.

The numerical scheme is deterministic. Finite non-periodic probability axes use
composite Boole quadrature by default, while azimuthal angles use the
endpoint-free trapezoidal rule by default.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Iterable

import numpy as np

from .constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    FLOAT_DTYPE,
    PI,
    real_array,
    real_empty,
)
from .kinematics import helicity, vec2
from .packets import LGPacket, resolve_normalizations
from .quadrature import (
    ProbabilityQuadrature,
    probability_inner_nodes,
    probability_outer_nodes,
    probability_transverse_nodes,
)
from .smatrix import (
    S_exact_time_grid,
    S_impulse_closed_grid,
    S_impulse_first_order_grid,
)


def _print_progress(
    *,
    label: str,
    completed: int,
    total: int,
    elapsed: float,
    point_text: str | None = None,
) -> None:
    """Update one compact in-place progress line for deterministic scans."""
    percent = 100.0 * completed / total if total else 100.0
    bar_width = 24
    filled = int(bar_width * completed / total) if total else bar_width
    filled = max(0, min(bar_width, filled))
    bar = "#" * filled + "-" * (bar_width - filled)
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    eta = (total - completed) / rate if rate > 0.0 else None
    parts = [
        f"{label}: [{bar}] {completed:6d} / {total:6d}",
        f"{percent:6.2f}%",
        f"elapsed = {elapsed:.2f} s",
    ]
    if eta is not None:
        parts.append(f"eta = {eta:.2f} s")
    if point_text is not None:
        parts.append(point_text)

    end = "\n" if completed >= total else ""
    print("\r" + " | ".join(parts) + " " * 20, end=end, flush=True)


def _resolve_workers(workers: int | None) -> int:
    """Return a validated worker count for outer grid scans."""
    if workers is None:
        return 1
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer or None.")
    return workers


def _vectorized_s_abs2_grid(
    k3x,
    k3y,
    k3z,
    k4x,
    k4y,
    k4z,
    packet1: LGPacket,
    packet2: LGPacket,
    impact_b,
    N1: float,
    N2: float,
    m: float,
    e_charge: float,
    helicities: tuple[float, ...],
    s_matrix: str,
    s_matrix_kwargs: dict | None,
):
    """Return vectorized |S|^2 for supported S-matrix selectors."""
    if len(helicities) != 2:
        raise ValueError("spin averaging expects exactly two helicity labels.")

    kwargs = {} if s_matrix_kwargs is None else dict(s_matrix_kwargs)
    kwargs.pop("return_details", None)
    matrix_element = kwargs.get("matrix_element", "ultrarelativistic")
    if matrix_element not in ("ultrarelativistic", "paraxial_massive"):
        raise ValueError(
            "matrix_element must be 'ultrarelativistic' or 'paraxial_massive'."
        )
    if matrix_element == "paraxial_massive" and s_matrix != "exact_time":
        raise ValueError(
            "matrix_element='paraxial_massive' is implemented only for "
            "s_matrix='exact_time' with denominator_mode='minkowski'. The "
            "closed and first_order probability kernels use the old "
            "ultrarelativistic transverse denominator."
        )
    lam1 = helicities[0]
    lam2 = helicities[1]

    def evaluate_channel(channel_lam1, channel_lam2, channel_lam3, channel_lam4):
        channel_kwargs = dict(kwargs)
        if s_matrix == "closed":
            allowed = {"matrix_element"}
            if any(key not in allowed for key in channel_kwargs):
                raise ValueError(
                    "s_matrix_kwargs for 'closed' may contain only "
                    "'matrix_element'."
                )
            return S_impulse_closed_grid(
                k3x,
                k3y,
                k3z,
                k4x,
                k4y,
                k4z,
                packet1,
                packet2,
                lam1=channel_lam1,
                lam2=channel_lam2,
                lam3=channel_lam3,
                lam4=channel_lam4,
                impact_b=impact_b,
                N1=N1,
                N2=N2,
                m=m,
                e_charge=e_charge,
                **channel_kwargs,
            )

        if s_matrix == "first_order":
            allowed = {"time_step", "time_step_scale", "matrix_element"}
            if any(key not in allowed for key in channel_kwargs):
                raise ValueError(
                    "s_matrix_kwargs for 'first_order' may contain only "
                    "'time_step', 'time_step_scale', and 'matrix_element'."
                )
            return S_impulse_first_order_grid(
                k3x,
                k3y,
                k3z,
                k4x,
                k4y,
                k4z,
                packet1,
                packet2,
                lam1=channel_lam1,
                lam2=channel_lam2,
                lam3=channel_lam3,
                lam4=channel_lam4,
                impact_b=impact_b,
                N1=N1,
                N2=N2,
                m=m,
                e_charge=e_charge,
                **channel_kwargs,
            )

        if s_matrix == "exact_time":
            allowed = {
                "quadrature",
                "denominator_mode",
                "denominator_regulator",
                "batch_size",
                "matrix_element",
            }
            if any(key not in allowed for key in channel_kwargs):
                raise ValueError(
                    "s_matrix_kwargs for 'exact_time' may contain only "
                    "'quadrature', 'denominator_mode', 'denominator_regulator', "
                    "'batch_size', and 'matrix_element'."
                )
            if channel_kwargs.get("batch_size") is None:
                channel_kwargs.pop("batch_size", None)
            return S_exact_time_grid(
                k3x,
                k3y,
                k3z,
                k4x,
                k4y,
                k4z,
                packet1,
                packet2,
                lam1=channel_lam1,
                lam2=channel_lam2,
                lam3=channel_lam3,
                lam4=channel_lam4,
                impact_b=impact_b,
                N1=N1,
                N2=N2,
                m=m,
                e_charge=e_charge,
                **channel_kwargs,
            )

        raise ValueError("s_matrix must be 'closed', 'first_order', or 'exact_time'.")

    if matrix_element == "paraxial_massive":
        h0 = helicity(helicities[0])
        h1 = helicity(helicities[1])
        if h0 != -h1:
            raise ValueError("spin averaging expects the two helicities +0.5 and -0.5.")

        same_h0_direct = evaluate_channel(h0, h0, h0, h0)
        same_h0_exchange = evaluate_channel(h0, h0, h1, h1)
        same_h1_direct = evaluate_channel(h1, h1, h1, h1)
        same_h1_exchange = evaluate_channel(h1, h1, h0, h0)

        opposite_h0_h1 = evaluate_channel(h0, h1, h0, h1)
        opposite_h1_h0 = evaluate_channel(h1, h0, h1, h0)

        return 0.25 * (
            np.abs(same_h0_direct) ** 2
            + np.abs(same_h0_exchange) ** 2
            + np.abs(same_h1_direct) ** 2
            + np.abs(same_h1_exchange) ** 2
            + np.abs(opposite_h0_h1) ** 2
            + np.abs(opposite_h1_h0) ** 2
        )

    if s_matrix == "closed":
        S = evaluate_channel(lam1, lam2, lam1, lam2)
        return np.abs(S) ** 2

    if s_matrix == "first_order":
        S = evaluate_channel(lam1, lam2, lam1, lam2)
        return np.abs(S) ** 2

    if s_matrix == "exact_time":
        S = evaluate_channel(lam1, lam2, lam1, lam2)
        return np.abs(S) ** 2

    raise ValueError("s_matrix must be 'closed', 'first_order', or 'exact_time'.")


def _prepare_probability_inputs(
    packet1: LGPacket,
    packet2: LGPacket,
    *,
    impact_b,
    N1: float | None,
    N2: float | None,
    m: float,
    e_charge: float,
    helicities: Iterable[float],
    s_matrix: str,
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
    helicities:
        Helicity labels to include in spin averaging.
    s_matrix, s_matrix_kwargs:
        S-matrix selector and optional extra keyword arguments.

    Returns
    -------
    dict
        Validated values passed to lower-level probability routines.
    """
    packet1 = packet1.checked()
    packet2 = packet2.checked()
    N1, N2 = resolve_normalizations(
        packet1,
        packet2,
        N1=N1,
        N2=N2,
        m=m,
    )
    return dict(
        packet1=packet1,
        packet2=packet2,
        impact_b=vec2(impact_b),
        N1=N1,
        N2=N2,
        m=m,
        e_charge=e_charge,
        helicities=tuple(helicities),
        s_matrix=s_matrix,
        s_matrix_kwargs=s_matrix_kwargs,
    )


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
    helicities: tuple[float, ...],
    s_matrix: str,
    s_matrix_kwargs: dict | None,
) -> np.float64:
    """Compute ``w(K_perp)`` with already prepared inputs.

    Parameters
    ----------
    K_perp:
        Fixed total final transverse momentum.
    quadrature:
        Inner probability quadrature settings.
    packet1, packet2, impact_b, N1, N2, m, e_charge, helicities,
    s_matrix, s_matrix_kwargs:
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

    phase_space_const = 1.0 / (2.0 * PI) ** 6

    rho_grid = rho_nodes[:, None, None, None]
    phi_cos = np.cos(phi_nodes)[None, :, None, None]
    phi_sin = np.sin(phi_nodes)[None, :, None, None]
    z3_grid = z3_nodes[None, None, :, None]
    z4_grid = z4_nodes[None, None, None, :]

    k3x_grid = rho_grid * phi_cos
    k3y_grid = rho_grid * phi_sin
    k4x_grid = K[0] - k3x_grid
    k4y_grid = K[1] - k3y_grid

    s_abs2_grid = _vectorized_s_abs2_grid(
        k3x_grid,
        k3y_grid,
        z3_grid,
        k4x_grid,
        k4y_grid,
        z4_grid,
        packet1,
        packet2,
        impact_b,
        N1,
        N2,
        m,
        e_charge,
        helicities,
        s_matrix,
        s_matrix_kwargs,
    )

    E3 = np.sqrt(
        m * m + k3x_grid * k3x_grid + k3y_grid * k3y_grid + z3_grid * z3_grid
    )
    E4 = np.sqrt(
        m * m + k4x_grid * k4x_grid + k4y_grid * k4y_grid + z4_grid * z4_grid
    )
    weights = (
        rho_weights[:, None, None, None]
        * phi_weights[None, :, None, None]
        * z3_weights[None, None, :, None]
        * z4_weights[None, None, None, :]
    )
    phase_space = phase_space_const / (4.0 * E3 * E4)
    return FLOAT_DTYPE(np.sum(weights * rho_grid * phase_space * s_abs2_grid))


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
    helicities: tuple[float, ...],
    s_matrix: str,
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
    packet1, packet2, impact_b, N1, N2, m, e_charge, helicities,
    s_matrix, s_matrix_kwargs:
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

    phase_space_const = 1.0 / (2.0 * PI) ** 6

    rho_grid = rho_nodes[:, None, None, None]
    phi_cos = np.cos(phi_nodes)[None, :, None, None]
    phi_sin = np.sin(phi_nodes)[None, :, None, None]
    z3_grid = np.asarray(k3z, dtype=FLOAT_DTYPE)[None, None, None, None]
    z4_grid = np.asarray(k4z, dtype=FLOAT_DTYPE)[None, None, None, None]

    k3x_grid = rho_grid * phi_cos
    k3y_grid = rho_grid * phi_sin
    k4x_grid = K[0] - k3x_grid
    k4y_grid = K[1] - k3y_grid

    s_abs2_grid = _vectorized_s_abs2_grid(
        k3x_grid,
        k3y_grid,
        z3_grid,
        k4x_grid,
        k4y_grid,
        z4_grid,
        packet1,
        packet2,
        impact_b,
        N1,
        N2,
        m,
        e_charge,
        helicities,
        s_matrix,
        s_matrix_kwargs,
    )

    E3 = np.sqrt(
        m * m + k3x_grid * k3x_grid + k3y_grid * k3y_grid + z3_grid * z3_grid
    )
    E4 = np.sqrt(
        m * m + k4x_grid * k4x_grid + k4y_grid * k4y_grid + z4_grid * z4_grid
    )
    weights = rho_weights[:, None, None, None] * phi_weights[None, :, None, None]
    phase_space = phase_space_const / (4.0 * E3 * E4)
    return FLOAT_DTYPE(np.sum(weights * rho_grid * phase_space * s_abs2_grid))


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
    helicities: Iterable[float] = (-0.5, 0.5),
    s_matrix: str = "closed",
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
    helicities:
        Helicity labels used for spin averaging.
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
        helicities=helicities,
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
    helicities: Iterable[float] = (-0.5, 0.5),
    s_matrix: str = "closed",
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
    helicities:
        Helicity labels used for spin averaging.
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
        helicities=helicities,
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
    helicities: Iterable[float] = (-0.5, 0.5),
    s_matrix: str = "closed",
    s_matrix_kwargs: dict | None = None,
    progress: bool = False,
    workers: int | None = None,
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
    impact_b, N1, N2, m, e_charge, helicities, s_matrix, s_matrix_kwargs:
        Same meaning as in ``longitudinal_density``.
    progress:
        If True, update completed point count, total point count, percent,
        elapsed time and ETA after each completed grid point.
    workers:
        Number of CPU worker threads for independent grid points.  The default
        ``None`` is equivalent to ``1`` and preserves deterministic sequential
        evaluation.

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
        helicities=helicities,
        s_matrix=s_matrix,
        s_matrix_kwargs=s_matrix_kwargs,
    )

    values = real_empty((len(k4z_values), len(k3z_values)))
    start_total = time.perf_counter()
    total_points = len(k4z_values) * len(k3z_values)
    workers = _resolve_workers(workers)

    if workers > 1:
        def evaluate_point(i4, i3, k4z, k3z):
            value = _longitudinal_density_resolved(
                k3z,
                k4z,
                K_perp,
                quadrature,
                **inputs,
            )
            return i4, i3, value

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(evaluate_point, i4, i3, k4z, k3z)
                for i4, k4z in enumerate(k4z_values)
                for i3, k3z in enumerate(k3z_values)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                i4, i3, value = future.result()
                values[i4, i3] = value

                if progress:
                    elapsed_total = time.perf_counter() - start_total
                    _print_progress(
                        label="longitudinal_density_grid",
                        completed=completed,
                        total=total_points,
                        elapsed=elapsed_total,
                        point_text=f"workers = {workers}",
                    )

        return values

    for i4, k4z in enumerate(k4z_values):
        for i3, k3z in enumerate(k3z_values):
            values[i4, i3] = _longitudinal_density_resolved(
                k3z,
                k4z,
                K_perp,
                quadrature,
                **inputs,
            )

            if progress:
                completed = i4 * len(k3z_values) + i3 + 1
                elapsed_total = time.perf_counter() - start_total
                _print_progress(
                    label="longitudinal_density_grid",
                    completed=completed,
                    total=total_points,
                    elapsed=elapsed_total,
                    point_text=(
                        f"row {i4 + 1} / {len(k4z_values)}"
                        f" | col {i3 + 1} / {len(k3z_values)}"
                        f" | k4z = {k4z:.12e} MeV"
                    ),
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
    helicities: Iterable[float] = (-0.5, 0.5),
    s_matrix: str = "closed",
    s_matrix_kwargs: dict | None = None,
    progress: bool = False,
    workers: int | None = None,
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
    impact_b, N1, N2, m, e_charge, helicities, s_matrix, s_matrix_kwargs:
        Same meaning as in ``diff_probability``.
    progress:
        If True, update completed point count, total point count, percent,
        elapsed time and ETA after each completed grid point.
    workers:
        Number of CPU worker threads for independent grid points.  The default
        ``None`` is equivalent to ``1`` and preserves deterministic sequential
        evaluation.

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
        helicities=helicities,
        s_matrix=s_matrix,
        s_matrix_kwargs=s_matrix_kwargs,
    )

    values = real_empty((len(Ky_values), len(Kx_values)))
    start_total = time.perf_counter()
    total_points = len(Ky_values) * len(Kx_values)
    workers = _resolve_workers(workers)

    if workers > 1:
        def evaluate_point(iy, ix, Ky, Kx):
            K_perp = real_array([Kx, Ky], shape=(2,))
            value = _diff_probability_resolved(K_perp, quadrature, **inputs)
            return iy, ix, value

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(evaluate_point, iy, ix, Ky, Kx)
                for iy, Ky in enumerate(Ky_values)
                for ix, Kx in enumerate(Kx_values)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                iy, ix, value = future.result()
                values[iy, ix] = value

                if progress:
                    elapsed_total = time.perf_counter() - start_total
                    _print_progress(
                        label="diff_probability_grid",
                        completed=completed,
                        total=total_points,
                        elapsed=elapsed_total,
                        point_text=f"workers = {workers}",
                    )

        return values

    for iy, Ky in enumerate(Ky_values):
        for ix, Kx in enumerate(Kx_values):
            K_perp = real_array([Kx, Ky], shape=(2,))
            values[iy, ix] = _diff_probability_resolved(K_perp, quadrature, **inputs)

            if progress:
                completed = iy * len(Kx_values) + ix + 1
                elapsed_total = time.perf_counter() - start_total
                _print_progress(
                    label="diff_probability_grid",
                    completed=completed,
                    total=total_points,
                    elapsed=elapsed_total,
                    point_text=(
                        f"row {iy + 1} / {len(Ky_values)}"
                        f" | col {ix + 1} / {len(Kx_values)}"
                        f" | Ky = {Ky:.12e} MeV"
                    ),
                )

    return values


def _outer_K_moments(
    quadrature: ProbabilityQuadrature,
    inputs: dict,
    *,
    progress: bool = False,
    workers: int | None = None,
) -> tuple[np.float64, np.float64]:
    """Return total probability and the ``K_y`` numerator.

    Parameters
    ----------
    quadrature:
        Probability quadrature settings including the outer ``K_perp`` axes.
    inputs:
        Validated values from ``_prepare_probability_inputs``.
    progress:
        If True, update completed outer ``K_perp`` point count, total point
        count, percent, elapsed time and ETA after each completed point.
    workers:
        Number of CPU worker threads for independent outer ``K_perp`` points.

    Returns
    -------
    tuple[np.float64, np.float64]
        Total probability and numerator for ``Ky_average``.
    """
    (K_nodes, K_weights), (phi_nodes, phi_weights) = probability_outer_nodes(quadrature)

    probability = FLOAT_DTYPE(0.0)
    ky_numerator = FLOAT_DTYPE(0.0)
    start_total = time.perf_counter()
    total_points = len(K_nodes) * len(phi_nodes)
    workers = _resolve_workers(workers)

    if workers > 1:
        probability_parts = real_empty(total_points)
        ky_parts = real_empty(total_points)

        def evaluate_point(index, K, w_K, phi_K, w_phi_K):
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)
            K_perp = real_array([Kx, Ky], shape=(2,))
            w_value = _diff_probability_resolved(K_perp, quadrature, **inputs)
            weight = w_K * w_phi_K * K
            return index, weight * w_value, weight * Ky * w_value

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            index = 0
            for K, w_K in zip(K_nodes, K_weights):
                for phi_K, w_phi_K in zip(phi_nodes, phi_weights):
                    futures.append(
                        executor.submit(
                            evaluate_point,
                            index,
                            K,
                            w_K,
                            phi_K,
                            w_phi_K,
                        )
                    )
                    index += 1

            for completed, future in enumerate(as_completed(futures), start=1):
                index, probability_part, ky_part = future.result()
                probability_parts[index] = probability_part
                ky_parts[index] = ky_part

                if progress:
                    elapsed_total = time.perf_counter() - start_total
                    _print_progress(
                        label="outer_K_moments",
                        completed=completed,
                        total=total_points,
                        elapsed=elapsed_total,
                        point_text=f"workers = {workers}",
                    )

        probability = np.sum(probability_parts, dtype=FLOAT_DTYPE)
        ky_numerator = np.sum(ky_parts, dtype=FLOAT_DTYPE)
        return probability, ky_numerator

    for iK, (K, w_K) in enumerate(zip(K_nodes, K_weights)):
        for i_phi, (phi_K, w_phi_K) in enumerate(zip(phi_nodes, phi_weights)):
            Kx = K * np.cos(phi_K)
            Ky = K * np.sin(phi_K)
            K_perp = real_array([Kx, Ky], shape=(2,))

            w_value = _diff_probability_resolved(K_perp, quadrature, **inputs)
            weight = w_K * w_phi_K * K

            probability = probability + weight * w_value
            ky_numerator = ky_numerator + weight * Ky * w_value

            if progress:
                completed = iK * len(phi_nodes) + i_phi + 1
                elapsed_total = time.perf_counter() - start_total
                _print_progress(
                    label="outer_K_moments",
                    completed=completed,
                    total=total_points,
                    elapsed=elapsed_total,
                    point_text=(
                        f"K row {iK + 1} / {len(K_nodes)}"
                        f" | phi {i_phi + 1} / {len(phi_nodes)}"
                        f" | K = {K:.12e} MeV"
                    ),
                )

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
    helicities: Iterable[float] = (-0.5, 0.5),
    s_matrix: str = "closed",
    s_matrix_kwargs: dict | None = None,
    progress: bool = False,
    workers: int | None = None,
) -> np.float64:
    """Compute the total probability in the selected ``K_perp`` domain.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings including ``K_perp_range``.
    impact_b, N1, N2, m, e_charge, helicities, s_matrix, s_matrix_kwargs:
        Same meaning as in ``diff_probability``.
    progress:
        If True, update completed outer ``K_perp`` point count, total point
        count, percent, elapsed time and ETA after each completed point.
    workers:
        Number of CPU worker threads for independent outer ``K_perp`` points.

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
        helicities=helicities,
        s_matrix=s_matrix,
        s_matrix_kwargs=s_matrix_kwargs,
    )
    probability, _ = _outer_K_moments(
        quadrature,
        inputs,
        progress=progress,
        workers=workers,
    )
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
    helicities: Iterable[float] = (-0.5, 0.5),
    s_matrix: str = "closed",
    s_matrix_kwargs: dict | None = None,
    progress: bool = False,
    workers: int | None = None,
) -> np.float64:
    """Compute ``<K_y>`` in the selected transverse ``K_perp`` domain.

    Parameters
    ----------
    packet1, packet2:
        Incoming wave packets.
    quadrature:
        Probability quadrature settings including ``K_perp_range``.
    impact_b, N1, N2, m, e_charge, helicities, s_matrix, s_matrix_kwargs:
        Same meaning as in ``diff_probability``.
    progress:
        If True, update completed outer ``K_perp`` point count, total point
        count, percent, elapsed time and ETA after each completed point.
    workers:
        Number of CPU worker threads for independent outer ``K_perp`` points.

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
        helicities=helicities,
        s_matrix=s_matrix,
        s_matrix_kwargs=s_matrix_kwargs,
    )
    probability, numerator = _outer_K_moments(
        quadrature,
        inputs,
        progress=progress,
        workers=workers,
    )
    return numerator / probability
