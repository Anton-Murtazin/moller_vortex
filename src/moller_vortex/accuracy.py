"""Global numerical accuracy configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalAccuracy:
    """Numerical accuracy parameters used across the project.

    quad_epsabs, quad_epsrel and quad_limit are passed to scipy.integrate.quad.
    root_residual_atol is reserved for delta-reduced S-matrix routines where
    physical roots are validated by an energy residual.
    """

    quad_epsabs: float = 1.0e-10
    quad_epsrel: float = 1.0e-10
    quad_limit: int = 300
    root_residual_atol: float = 1.0e-10


ACCURACY = NumericalAccuracy()
