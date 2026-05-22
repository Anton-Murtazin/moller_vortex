"""Shared deterministic quadrature rules."""

from __future__ import annotations

import numpy as np


def legendre_nodes_and_weights(interval: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on a finite interval."""
    a, b = interval
    n = int(n)
    if n <= 0:
        raise ValueError("Gauss-Legendre quadrature requires n > 0.")

    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w
    return nodes, weights


def boole_nodes_and_weights(interval: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Boole rule on a finite interval.

    The number of nodes must satisfy n = 4*m + 1.
    """
    a, b = interval
    n = int(n)

    if n < 5:
        raise ValueError("Composite Boole quadrature requires at least 5 nodes.")

    n_intervals = n - 1
    if n_intervals % 4 != 0:
        raise ValueError("Composite Boole quadrature requires n - 1 divisible by 4.")

    nodes = np.linspace(a, b, n)
    h = (b - a) / n_intervals

    weights = np.zeros(n, dtype=np.float64)
    for j in range(0, n_intervals, 4):
        weights[j] += 7.0
        weights[j + 1] += 32.0
        weights[j + 2] += 12.0
        weights[j + 3] += 32.0
        weights[j + 4] += 7.0

    weights *= 2.0 * h / 45.0
    return nodes, weights


def trapezoid_nodes_and_weights(
    interval: tuple[float, float],
    n: int,
    *,
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoidal-rule nodes and weights on a finite interval.

    With endpoint=False the rule is the endpoint-free periodic trapezoidal rule.
    This is the preferred choice for azimuthal angles.
    """
    a, b = interval
    n = int(n)

    if n <= 0:
        raise ValueError("Trapezoidal quadrature requires n > 0.")

    if endpoint:
        if n < 2:
            raise ValueError("Closed trapezoidal quadrature requires n >= 2.")
        nodes = np.linspace(a, b, n, endpoint=True)
        h = (b - a) / (n - 1)
        weights = np.full(n, h, dtype=np.float64)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return nodes, weights

    nodes = np.linspace(a, b, n, endpoint=False)
    weights = np.full(n, (b - a) / n, dtype=np.float64)
    return nodes, weights


def nodes_and_weights(
    interval: tuple[float, float],
    n: int,
    *,
    method: str = "boole",
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch finite-interval quadrature rules by name.

    Supported methods are ``"boole"``, ``"legendre"`` and ``"trapezoid"``.
    The ``endpoint`` flag is used only by ``"trapezoid"``.
    """
    if method == "boole":
        return boole_nodes_and_weights(interval, n)
    if method == "legendre":
        return legendre_nodes_and_weights(interval, n)
    if method == "trapezoid":
        return trapezoid_nodes_and_weights(interval, n, endpoint=endpoint)
    raise ValueError("Unknown quadrature method: {!r}".format(method))
