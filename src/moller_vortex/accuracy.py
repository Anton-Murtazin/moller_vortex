"""Global numerical accuracy configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NumericalAccuracy:
    """Numerical accuracy parameters used across the project.

    Parameters
    ----------
    quad_epsabs:
        Absolute tolerance passed to ``scipy.integrate.quad``.
    quad_epsrel:
        Relative tolerance passed to ``scipy.integrate.quad``.
    quad_limit:
        Maximum number of subintervals passed to ``scipy.integrate.quad``.
    root_residual_atol:
        Absolute tolerance reserved for validating energy-residual roots in
        delta-reduced expressions.

    Returns
    -------
    NumericalAccuracy
        Immutable container for shared numerical tolerances.
    """

    quad_epsabs: float = 1.0e-10
    quad_epsrel: float = 1.0e-10
    quad_limit: int = 30
    root_residual_atol: float = 1.0e-10

    def quad_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``scipy.integrate.quad``.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, Any]
            Dictionary with ``epsabs``, ``epsrel`` and ``limit``.
        """
        return dict(
            epsabs=self.quad_epsabs,
            epsrel=self.quad_epsrel,
            limit=self.quad_limit,
        )


ACCURACY = NumericalAccuracy()


def resolve_accuracy(accuracy: NumericalAccuracy | None = None) -> NumericalAccuracy:
    """Return the explicit accuracy object or the package default.

    Parameters
    ----------
    accuracy:
        Optional numerical accuracy object.

    Returns
    -------
    NumericalAccuracy
        ``accuracy`` if provided, otherwise ``ACCURACY``.
    """
    return ACCURACY if accuracy is None else accuracy
