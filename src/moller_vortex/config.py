"""Global numerical type, physical constants, and integration grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Change this one line to switch the real dtype for the whole package.
REAL_DTYPE = np.float64
COMPLEX_DTYPE = np.result_type(REAL_DTYPE, np.complex64).type

PI = REAL_DTYPE(np.pi)
ALPHA_EM = REAL_DTYPE(1.0 / 137.035999084)
ELECTRON_CHARGE = REAL_DTYPE(np.sqrt(4.0 * PI * ALPHA_EM))
ELECTRON_MASS = REAL_DTYPE(0.51099895000)  # MeV
HBARC_MEV_NM = REAL_DTYPE(1.973269804e-4)


@dataclass(frozen=True)
class Axis:
    """One finite quadrature axis.

    ``rule`` is ``"boole"`` (composite Boole), ``"gauss"``
    (Gauss-Legendre), ``"trapezoid"`` (closed), or ``"periodic"``
    (endpoint-free trapezoid, intended for angles). Composite Boole is the
    default and requires ``points = 4*m + 1``.
    """

    start: float
    stop: float
    points: int
    rule: str = "boole"

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.stop):
            raise ValueError("Axis bounds must be finite.")
        if self.stop <= self.start:
            raise ValueError("Axis.stop must be greater than Axis.start.")
        if int(self.points) != self.points or self.points < 1:
            raise ValueError("Axis.points must be a positive integer.")
        if self.rule not in {"boole", "gauss", "trapezoid", "periodic"}:
            raise ValueError(
                "Axis.rule must be 'boole', 'gauss', 'trapezoid', or 'periodic'."
            )
        if self.rule == "boole" and (self.points < 5 or (self.points - 1) % 4):
            raise ValueError("Composite Boole requires points = 4*m + 1 >= 5.")
        if self.rule == "trapezoid" and self.points < 2:
            raise ValueError("Closed trapezoid quadrature needs at least 2 points.")

    def nodes_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return nodes and weights in the global real dtype."""
        a = REAL_DTYPE(self.start)
        b = REAL_DTYPE(self.stop)
        n = int(self.points)

        if self.rule == "boole":
            nodes = np.linspace(a, b, n, endpoint=True)
            weights = np.zeros(n, dtype=REAL_DTYPE)
            for index in range(0, n - 1, 4):
                weights[index : index + 5] += (7.0, 32.0, 12.0, 32.0, 7.0)
            weights *= 2.0 * (b - a) / ((n - 1) * 45.0)
        elif self.rule == "gauss":
            x, w = np.polynomial.legendre.leggauss(n)
            nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
            weights = 0.5 * (b - a) * w
        elif self.rule == "periodic":
            nodes = np.linspace(a, b, n, endpoint=False)
            weights = np.full(n, (b - a) / n)
        else:
            nodes = np.linspace(a, b, n, endpoint=True)
            weights = np.full(n, (b - a) / (n - 1))
            weights[[0, -1]] *= 0.5

        return (
            np.asarray(nodes, dtype=REAL_DTYPE),
            np.asarray(weights, dtype=REAL_DTYPE),
        )


@dataclass(frozen=True)
class VortexPacket:
    """Parameters of the Lorentz-covariant vortex state from Eq. (4)."""

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
    """Two-dimensional integration domain for packet normalization."""

    k_perp: Axis
    k_z: Axis

    def __post_init__(self) -> None:
        if self.k_perp.start < 0.0:
            raise ValueError("Normalization k_perp must start at zero or above.")


@dataclass(frozen=True)
class ScatteringGrid:
    """Quadrature settings for the transverse integral T in Eq. (29).

    The radial variable is the momentum transfer
    ``q_perp = |k3_perp-k1_perp|``. ``q_perp.start`` is therefore the physical
    forward cutoff and must be positive. This change of variables puts the
    pole at an integration boundary instead of masking nodes inside a disk.
    """

    q_perp: Axis
    q_phi: Axis = Axis(0.0, 2.0 * np.pi, 64, "periodic")
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.q_perp.start <= 0.0:
            raise ValueError("Scattering q_perp.start must be a positive cutoff.")
        if int(self.batch_size) != self.batch_size or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

    @property
    def q_min(self) -> float:
        return self.q_perp.start


@dataclass(frozen=True)
class ProbabilityGrid:
    """Inner and optional outer phase-space quadratures.

    The four inner axes define ``w(K_perp)``.  ``K_perp`` and ``K_phi`` are
    additionally required for the total probability and mean momentum.
    """

    k3_perp: Axis
    k3_z: Axis
    k4_z: Axis
    k3_phi: Axis = Axis(0.0, 2.0 * np.pi, 32, "periodic")
    K_perp: Axis | None = None
    K_phi: Axis | None = None

    def __post_init__(self) -> None:
        if self.k3_perp.start < 0.0:
            raise ValueError("Probability k3_perp must start at zero or above.")
        if (self.K_perp is None) != (self.K_phi is None):
            raise ValueError("Set both K_perp and K_phi, or neither of them.")
        if self.K_perp is not None and self.K_perp.start < 0.0:
            raise ValueError("Outer K_perp must start at zero or above.")


def spatial_width_nm_to_momentum_mev(width_nm: float) -> float:
    """Convert a coordinate width in nm to a momentum width in MeV."""
    if width_nm <= 0.0:
        raise ValueError("width_nm must be positive.")
    return float(HBARC_MEV_NM / width_nm)
