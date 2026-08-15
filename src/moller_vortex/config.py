"""Global numerical type, constants, and explicit quadrature axes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Change this one line to switch the real dtype throughout the package.
REAL_DTYPE = np.float64
COMPLEX_DTYPE = np.result_type(REAL_DTYPE, np.complex64).type

PI = REAL_DTYPE(np.pi)
ALPHA_EM = REAL_DTYPE(1.0 / 137.035999084)
ELECTRON_CHARGE = REAL_DTYPE(np.sqrt(4.0 * PI * ALPHA_EM))
ELECTRON_MASS = REAL_DTYPE(0.51099895000)  # MeV
HBARC_MEV_NM = REAL_DTYPE(1.973269804e-4)  # MeV nm


@dataclass(frozen=True)
class Axis:
    """A finite one-dimensional quadrature axis.

    ``rule="gauss"`` uses Gauss-Legendre quadrature. ``rule="trapezoid"``
    includes both endpoints. ``rule="periodic"`` is the endpoint-free
    trapezoidal rule intended for azimuthal angles.
    """

    start: float
    stop: float
    points: int
    rule: str = "gauss"

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.stop):
            raise ValueError("Axis bounds must be finite.")
        if self.stop <= self.start:
            raise ValueError("Axis.stop must be greater than Axis.start.")
        if not isinstance(self.points, (int, np.integer)) or self.points < 1:
            raise ValueError("Axis.points must be a positive integer.")
        if self.rule not in {"gauss", "trapezoid", "periodic"}:
            raise ValueError("Axis.rule must be 'gauss', 'trapezoid', or 'periodic'.")
        if self.rule == "trapezoid" and self.points < 2:
            raise ValueError("Trapezoid quadrature needs at least two points.")

    def nodes_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return quadrature nodes and weights in the global real dtype."""
        a = REAL_DTYPE(self.start)
        b = REAL_DTYPE(self.stop)
        n = int(self.points)

        if self.rule == "gauss":
            nodes, weights = np.polynomial.legendre.leggauss(n)
            nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
            weights = 0.5 * (b - a) * weights
        elif self.rule == "periodic":
            nodes = np.linspace(a, b, n, endpoint=False, dtype=REAL_DTYPE)
            weights = np.full(n, (b - a) / n, dtype=REAL_DTYPE)
        else:
            nodes = np.linspace(a, b, n, endpoint=True, dtype=REAL_DTYPE)
            weights = np.full(n, (b - a) / (n - 1), dtype=REAL_DTYPE)
            weights[[0, -1]] *= REAL_DTYPE(0.5)

        return (
            np.asarray(nodes, dtype=REAL_DTYPE),
            np.asarray(weights, dtype=REAL_DTYPE),
        )


@dataclass(frozen=True)
class VortexPacket:
    """Parameters of the reference-frame vortex state in PDF Eq. (4)."""

    ell: int
    sigma_perp: float
    sigma_parallel: float
    sigma_energy: float
    k0_z: float

    def __post_init__(self) -> None:
        if not isinstance(self.ell, (int, np.integer)):
            raise TypeError("ell must be an integer.")
        widths = (self.sigma_perp, self.sigma_parallel, self.sigma_energy)
        if not all(np.isfinite(value) and value > 0.0 for value in widths):
            raise ValueError("All packet widths must be positive and finite.")
        if not np.isfinite(self.k0_z):
            raise ValueError("k0_z must be finite.")


@dataclass(frozen=True)
class NormalizationGrid:
    """Two-dimensional finite domain for the normalization integral."""

    k_perp: Axis
    k_z: Axis
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.k_perp.start < 0.0:
            raise ValueError("Normalization k_perp must start at zero or above.")
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


@dataclass(frozen=True)
class ProbabilityGrid:
    """Quadrature axes for differential and integrated probabilities.

    The four inner axes define ``dP/d^2K_perp``. The optional outer axes are
    required only by ``total_probability`` and ``mean_total_momentum``.
    """

    k3_perp: Axis
    k3_phi: Axis
    k3_z: Axis
    k4_z: Axis
    K_perp: Axis | None = None
    K_phi: Axis | None = None
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.k3_perp.start <= 0.0:
            raise ValueError(
                "k3_perp.start must be positive because PDF Eq. (32) contains 1/k3_perp^2."
            )
        if self.k3_phi.rule != "periodic":
            raise ValueError("k3_phi must use rule='periodic'.")
        if (self.K_perp is None) != (self.K_phi is None):
            raise ValueError("Set both outer axes K_perp and K_phi, or neither.")
        if self.K_perp is not None and self.K_perp.start < 0.0:
            raise ValueError("Outer K_perp must start at zero or above.")
        if self.K_phi is not None and self.K_phi.rule != "periodic":
            raise ValueError("K_phi must use rule='periodic'.")
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")


def spatial_width_nm_to_momentum_mev(width_nm: float) -> float:
    """Convert the Gaussian coordinate width in nm to momentum width in MeV."""
    if not np.isfinite(width_nm) or width_nm <= 0.0:
        raise ValueError("width_nm must be positive and finite.")
    return float(HBARC_MEV_NM / width_nm)
