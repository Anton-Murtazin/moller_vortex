"""Paraxial ultrarelativistic S-matrix from Eqs. (21)-(31)."""

from __future__ import annotations

import math

import numpy as np

from .config import (
    COMPLEX_DTYPE,
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PI,
    REAL_DTYPE,
    ScatteringGrid,
    VortexPacket,
)
from .states import _vortex_power, central_energy, effective_sigma


def _vectors(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.shape == () or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}).")
    return array


def _broadcast_vectors(first, second, size: int, names: tuple[str, str]):
    first = _vectors(first, size, names[0])
    second = _vectors(second, size, names[1])
    try:
        return np.broadcast_arrays(first, second)
    except ValueError as error:
        raise ValueError(f"{names[0]} and {names[1]} are not broadcastable.") from error


def _validate_norms(norms) -> tuple[float, float]:
    if len(norms) != 2:
        raise ValueError("norms must contain (N1, N2).")
    N1, N2 = (float(norms[0]), float(norms[1]))
    if not all(np.isfinite(value) and value > 0.0 for value in (N1, N2)):
        raise ValueError("Both normalization constants must be positive and finite.")
    return N1, N2


def _packet_denominator(packet1: VortexPacket, packet2: VortexPacket) -> float:
    l1 = abs(packet1.ell)
    l2 = abs(packet2.ell)
    log_value = (
        l1 * math.log(packet1.sigma_perp)
        + l2 * math.log(packet2.sigma_perp)
        + 0.5 * (math.lgamma(l1 + 1) + math.lgamma(l2 + 1))
    )
    return math.exp(log_value)


def _base_prefactor(
    packet1: VortexPacket,
    packet2: VortexPacket,
    norms,
    charge: float,
) -> complex:
    N1, N2 = _validate_norms(norms)
    return COMPLEX_DTYPE(
        -1j
        * charge**2
        / (4.0 * PI**3)
        * N1
        * N2
        / _packet_denominator(packet1, packet2)
    )


def _longitudinal_from_energies(
    E3,
    E4,
    K_z,
    packet1: VortexPacket,
    packet2: VortexPacket,
    mass: float,
):
    """Vectorized Eq. (28) for already computed final energies."""
    E01 = central_energy(packet1, mass)
    E02 = central_energy(packet2, mass)
    v1 = packet1.k0_z / E01
    v2 = packet2.k0_z / E02
    delta_v = v1 - v2
    if abs(delta_v) <= np.finfo(REAL_DTYPE).eps:
        raise ValueError("The impulse approximation requires v1 != v2.")

    delta_Kz = K_z - packet1.k0_z - packet2.k0_z
    delta_E = E3 + E4 - E01 - E02
    sigma1 = effective_sigma(packet1, mass)
    sigma2 = effective_sigma(packet2, mass)
    exponent = (
        -(delta_E - v2 * delta_Kz) ** 2
        / (2.0 * sigma1**2 * delta_v**2)
        -(delta_E - v1 * delta_Kz) ** 2
        / (2.0 * sigma2**2 * delta_v**2)
    )
    return 2.0 * PI / abs(delta_v) * np.exp(exponent)


def longitudinal_factor(
    k3,
    k4,
    packet1: VortexPacket,
    packet2: VortexPacket,
    *,
    mass: float = ELECTRON_MASS,
):
    """Evaluate the separated longitudinal factor L in Eq. (28)."""
    k3, k4 = _broadcast_vectors(k3, k4, 3, ("k3", "k4"))
    E3 = np.sqrt(mass**2 + np.sum(k3 * k3, axis=-1))
    E4 = np.sqrt(mass**2 + np.sum(k4 * k4, axis=-1))
    return _longitudinal_from_energies(
        E3, E4, k3[..., 2] + k4[..., 2], packet1, packet2, mass
    )


def transverse_integral(
    k3_perp,
    K_perp,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ScatteringGrid,
    *,
    impact=(0.0, 0.0),
):
    """Evaluate the two-dimensional transverse integral T in Eq. (29).

    ``k3_perp`` and ``K_perp`` may be vectors or broadcastable arrays of
    vectors.  The final-point dimension is batched; each batch evaluates all
    transverse quadrature nodes with NumPy broadcasting.
    """
    k3, K = _broadcast_vectors(k3_perp, K_perp, 2, ("k3_perp", "K_perp"))
    b = np.asarray(impact, dtype=REAL_DTYPE)
    if b.shape != (2,):
        raise ValueError("impact must have shape (2,).")

    output_shape = k3.shape[:-1]
    k3_flat = k3.reshape(-1, 2)
    K_flat = K.reshape(-1, 2)
    output = np.empty(k3_flat.shape[0], dtype=COMPLEX_DTYPE)

    transfer, transfer_weight = grid.q_perp.nodes_weights()
    phi, phi_weight = grid.q_phi.nodes_weights()
    q = transfer[None, :, None]
    qx = q * np.cos(phi)[None, None, :]
    qy = q * np.sin(phi)[None, None, :]
    # d^2q / q^2 = dq dphi / q. The positive lower boundary is q_min.
    weight = transfer_weight[None, :, None] * phi_weight[None, None, :] / q

    for start in range(0, k3_flat.shape[0], grid.batch_size):
        stop = min(start + grid.batch_size, k3_flat.shape[0])
        k3_batch = k3_flat[start:stop]
        K_batch = K_flat[start:stop]

        x1 = k3_batch[:, 0, None, None] - qx
        y1 = k3_batch[:, 1, None, None] - qy
        x2 = K_batch[:, 0, None, None] - x1
        y2 = K_batch[:, 1, None, None] - y1

        packet1_power = _vortex_power(x1, y1, packet1.ell)
        packet1_gaussian = np.exp(
            -(x1**2 + y1**2) / (2.0 * packet1.sigma_perp**2)
        )
        packet2_power = _vortex_power(x2, y2, packet2.ell)
        packet2_gaussian = np.exp(
            -(x2**2 + y2**2) / (2.0 * packet2.sigma_perp**2)
        )
        impact_phase = np.exp(1j * (b[0] * x2 + b[1] * y2))
        integrand = (
            packet1_power
            * packet2_power
            * packet1_gaussian
            * packet2_gaussian
            * impact_phase
        )
        output[start:stop] = np.sum(weight * integrand, axis=(1, 2))

    output = output.reshape(output_shape)
    return output.item() if output.ndim == 0 else output


def s_matrix(
    k3,
    k4,
    packet1: VortexPacket,
    packet2: VortexPacket,
    grid: ScatteringGrid,
    *,
    norms,
    impact=(0.0, 0.0),
    helicities=(0.5, -0.5, 0.5, -0.5),
    mass: float = ELECTRON_MASS,
    charge: float = ELECTRON_CHARGE,
    return_info: bool = False,
):
    """Compute the S-matrix in Eq. (31) for scalar or broadcast momenta."""
    k3, k4 = _broadcast_vectors(k3, k4, 3, ("k3", "k4"))
    if len(helicities) != 4 or any(value not in (-0.5, 0.5) for value in helicities):
        raise ValueError("helicities must contain four values equal to +/-0.5.")

    output_shape = k3.shape[:-1]
    helicity_conserving = (
        helicities[2] == helicities[0] and helicities[3] == helicities[1]
    )
    if not helicity_conserving:
        zero = np.zeros(output_shape, dtype=COMPLEX_DTYPE)
        zero = zero.item() if zero.ndim == 0 else zero
        return (zero, {"reason": "helicity delta is zero"}) if return_info else zero

    E3 = np.sqrt(mass**2 + np.sum(k3 * k3, axis=-1))
    E4 = np.sqrt(mass**2 + np.sum(k4 * k4, axis=-1))
    E01 = central_energy(packet1, mass)
    E02 = central_energy(packet2, mass)
    K = k3 + k4
    L = _longitudinal_from_energies(
        E3, E4, K[..., 2], packet1, packet2, mass
    )
    T = transverse_integral(
        k3[..., :2], K[..., :2], packet1, packet2, grid, impact=impact
    )
    prefactor = _base_prefactor(packet1, packet2, norms, charge)
    result = prefactor * np.sqrt(E3 * E4 / (E01 * E02)) * L * T
    result = np.asarray(result, dtype=COMPLEX_DTYPE)
    result_out = result.item() if result.ndim == 0 else result

    if not return_info:
        return result_out
    return result_out, {
        "longitudinal_factor": L,
        "transverse_integral": T,
        "base_prefactor": prefactor,
        "E3": E3,
        "E4": E4,
        "E01": E01,
        "E02": E02,
        "K": K,
        "q_min": grid.q_min,
    }
