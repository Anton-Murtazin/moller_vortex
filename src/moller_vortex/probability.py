"""Differential and integrated probabilities for the new vortex states."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PI,
    ProbabilityGrid,
    REAL_DTYPE,
    ScatteringGrid,
    VortexPacket,
)
from .scattering import (
    _base_prefactor,
    _longitudinal_from_energies,
    transverse_integral,
)
from .states import central_energy


def _vector2(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=REAL_DTYPE)
    if value.shape != (2,):
        raise ValueError(f"{name} must have shape (2,).")
    return value


def _workers(value: int | None) -> int:
    if value is None:
        return 1
    value = int(value)
    if value < 1:
        raise ValueError("workers must be a positive integer or None.")
    return value


def _transverse_final_grid(K, grid: ProbabilityGrid):
    rho, rho_weight = grid.k3_perp.nodes_weights()
    phi, phi_weight = grid.k3_phi.nodes_weights()
    rho_mesh = rho[:, None]
    phi_mesh = phi[None, :]
    k3x = rho_mesh * np.cos(phi_mesh)
    k3y = rho_mesh * np.sin(phi_mesh)
    k3_perp = np.stack((k3x, k3y), axis=-1).reshape(-1, 2)
    k4_perp = K[None, :] - k3_perp
    weights = (rho_weight[:, None] * phi_weight[None, :] * rho_mesh).reshape(-1)
    return k3_perp, k4_perp, weights


def _probability_constant(
    packet1: VortexPacket,
    packet2: VortexPacket,
    norms,
    mass: float,
    charge: float,
) -> float:
    """Constant after the E3 E4 cancellation in the final phase space."""
    base = _base_prefactor(packet1, packet2, norms, charge)
    phase_space = 1.0 / (4.0 * (2.0 * PI) ** 6)
    return float(
        phase_space
        * abs(base) ** 2
        / (central_energy(packet1, mass) * central_energy(packet2, mass))
    )


def _probability_at_K(
    K,
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact,
    mass: float,
    charge: float,
) -> tuple[float, float]:
    """Return ``w(K_perp)`` and its Kz-weighted inner moment."""
    K = _vector2(K, "K")
    k3_perp, k4_perp, transverse_weight = _transverse_final_grid(
        K, probability_grid
    )
    K_batch = np.broadcast_to(K, k3_perp.shape)
    T = transverse_integral(
        k3_perp,
        K_batch,
        packet1,
        packet2,
        scattering_grid,
        impact=impact,
    )

    z3, z3_weight = probability_grid.k3_z.nodes_weights()
    z4, z4_weight = probability_grid.k4_z.nodes_weights()
    z_weight = z3_weight[None, :, None] * z4_weight[None, None, :]
    K_z = z3[None, :, None] + z4[None, None, :]

    probability_sum = REAL_DTYPE(0.0)
    kz_sum = REAL_DTYPE(0.0)
    batch_size = scattering_grid.batch_size
    k3_perp_squared = np.sum(k3_perp * k3_perp, axis=1)
    k4_perp_squared = np.sum(k4_perp * k4_perp, axis=1)

    for start in range(0, k3_perp.shape[0], batch_size):
        stop = min(start + batch_size, k3_perp.shape[0])
        E3 = np.sqrt(
            mass**2
            + k3_perp_squared[start:stop, None, None]
            + z3[None, :, None] ** 2
        )
        E4 = np.sqrt(
            mass**2
            + k4_perp_squared[start:stop, None, None]
            + z4[None, None, :] ** 2
        )
        L = _longitudinal_from_energies(
            E3, E4, K_z, packet1, packet2, mass
        )
        longitudinal_integral = np.sum(z_weight * L**2, axis=(1, 2))
        kz_integral = np.sum(z_weight * K_z * L**2, axis=(1, 2))
        common = (
            transverse_weight[start:stop]
            * np.abs(T[start:stop]) ** 2
        )
        probability_sum += np.sum(common * longitudinal_integral)
        kz_sum += np.sum(common * kz_integral)

    constant = _probability_constant(packet1, packet2, norms, mass, charge)
    return float(constant * probability_sum), float(constant * kz_sum)


def longitudinal_density_grid(
    k3_z_values,
    k4_z_values,
    K_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
) -> np.ndarray:
    """Return ``d^2 w/(dk3_z dk4_z)`` on a longitudinal grid.

    The transverse integral is evaluated once and reused for all longitudinal
    points. The returned shape is ``(len(k4_z_values), len(k3_z_values))``.
    """
    k3_z = np.asarray(k3_z_values, dtype=REAL_DTYPE)
    k4_z = np.asarray(k4_z_values, dtype=REAL_DTYPE)
    if k3_z.ndim != 1 or k4_z.ndim != 1:
        raise ValueError("k3_z_values and k4_z_values must be one-dimensional.")
    K = _vector2(K_perp, "K_perp")
    k3_perp, k4_perp, transverse_weight = _transverse_final_grid(
        K, probability_grid
    )
    T = transverse_integral(
        k3_perp,
        np.broadcast_to(K, k3_perp.shape),
        packet1,
        packet2,
        scattering_grid,
        impact=impact,
    )

    density = np.zeros((k3_z.size, k4_z.size), dtype=REAL_DTYPE)
    k3_perp_squared = np.sum(k3_perp**2, axis=1)
    k4_perp_squared = np.sum(k4_perp**2, axis=1)
    K_z = k3_z[None, :, None] + k4_z[None, None, :]

    for start in range(0, k3_perp.shape[0], scattering_grid.batch_size):
        stop = min(start + scattering_grid.batch_size, k3_perp.shape[0])
        E3 = np.sqrt(
            mass**2
            + k3_perp_squared[start:stop, None, None]
            + k3_z[None, :, None] ** 2
        )
        E4 = np.sqrt(
            mass**2
            + k4_perp_squared[start:stop, None, None]
            + k4_z[None, None, :] ** 2
        )
        L = _longitudinal_from_energies(
            E3, E4, K_z, packet1, packet2, mass
        )
        common = (
            transverse_weight[start:stop]
            * np.abs(T[start:stop]) ** 2
        )
        density += np.sum(common[:, None, None] * L**2, axis=0)

    constant = _probability_constant(packet1, packet2, norms, mass, charge)
    return np.asarray(constant * density.T, dtype=REAL_DTYPE)


def longitudinal_density(
    k3_z: float,
    k4_z: float,
    K_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
) -> float:
    """Return ``d^2 w/(dk3_z dk4_z)`` at one longitudinal point."""
    values = longitudinal_density_grid(
        [k3_z],
        [k4_z],
        K_perp,
        packet1,
        packet2,
        scattering_grid,
        probability_grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
    )
    return float(values[0, 0])


def differential_probability(
    K_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
) -> float:
    """Compute ``w(K_perp)`` with the inner four-dimensional quadrature."""
    value, _ = _probability_at_K(
        K_perp,
        packet1,
        packet2,
        scattering_grid,
        probability_grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
    )
    return value


def differential_probability_grid(
    Kx_values,
    Ky_values,
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    workers: int | None = 1,
) -> np.ndarray:
    """Evaluate ``w(Kx, Ky)``; independent K points may use worker threads."""
    Kx = np.asarray(Kx_values, dtype=REAL_DTYPE)
    Ky = np.asarray(Ky_values, dtype=REAL_DTYPE)
    if Kx.ndim != 1 or Ky.ndim != 1:
        raise ValueError("Kx_values and Ky_values must be one-dimensional.")
    points = [np.array((x, y), dtype=REAL_DTYPE) for y in Ky for x in Kx]

    def evaluate(K):
        return differential_probability(
            K,
            packet1,
            packet2,
            scattering_grid,
            probability_grid,
            norms=norms,
            impact=impact,
            mass=mass,
            charge=charge,
        )

    worker_count = _workers(workers)
    if worker_count == 1:
        values = [evaluate(point) for point in points]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            values = list(executor.map(evaluate, points))
    return np.asarray(values, dtype=REAL_DTYPE).reshape(Ky.size, Kx.size)


def _outer_integral(
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact,
    mass: float,
    charge: float,
    workers: int | None,
) -> tuple[float, np.ndarray]:
    if probability_grid.K_perp is None or probability_grid.K_phi is None:
        raise ValueError(
            "ProbabilityGrid.K_perp and K_phi are required for outer integration."
        )

    K_radius, K_radius_weight = probability_grid.K_perp.nodes_weights()
    K_phi, K_phi_weight = probability_grid.K_phi.nodes_weights()
    points: list[np.ndarray] = []
    weights: list[float] = []
    for radius, w_radius in zip(K_radius, K_radius_weight):
        for phi, w_phi in zip(K_phi, K_phi_weight):
            points.append(
                np.array((radius * np.cos(phi), radius * np.sin(phi)), dtype=REAL_DTYPE)
            )
            weights.append(float(w_radius * w_phi * radius))

    def evaluate(K):
        return _probability_at_K(
            K,
            packet1,
            packet2,
            scattering_grid,
            probability_grid,
            norms=norms,
            impact=impact,
            mass=mass,
            charge=charge,
        )

    worker_count = _workers(workers)
    if worker_count == 1:
        values = [evaluate(point) for point in points]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            values = list(executor.map(evaluate, points))

    weights_array = np.asarray(weights, dtype=REAL_DTYPE)
    probability_values = np.asarray([value[0] for value in values], dtype=REAL_DTYPE)
    kz_values = np.asarray([value[1] for value in values], dtype=REAL_DTYPE)
    points_array = np.asarray(points, dtype=REAL_DTYPE)
    total = float(np.sum(weights_array * probability_values))
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("Total probability is not positive and finite.")

    momentum_numerator = np.array(
        [
            np.sum(weights_array * points_array[:, 0] * probability_values),
            np.sum(weights_array * points_array[:, 1] * probability_values),
            np.sum(weights_array * kz_values),
        ],
        dtype=REAL_DTYPE,
    )
    return total, momentum_numerator / total


def total_probability(
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    workers: int | None = 1,
    return_mean: bool = False,
):
    """Integrate over K_perp; optionally return ``(P, <K>)`` in one pass."""
    probability, mean = _outer_integral(
        packet1,
        packet2,
        scattering_grid,
        probability_grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
        workers=workers,
    )
    return (probability, mean) if return_mean else probability


def mean_total_momentum(
    packet1: VortexPacket,
    packet2: VortexPacket,
    scattering_grid: ScatteringGrid,
    probability_grid: ProbabilityGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    workers: int | None = 1,
) -> np.ndarray:
    """Return the probability-weighted ``(<Kx>, <Ky>, <Kz>)``."""
    _, mean = _outer_integral(
        packet1,
        packet2,
        scattering_grid,
        probability_grid,
        norms=norms,
        impact=impact,
        mass=mass,
        charge=charge,
        workers=workers,
    )
    return mean
