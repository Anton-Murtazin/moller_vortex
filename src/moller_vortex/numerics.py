"""Reusable quadrature, vector, and sequential progress helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np

from .config import REAL_DTYPE

Input = TypeVar("Input")
Result = TypeVar("Result")


@dataclass(frozen=True)
class Axis:
    """A finite one-dimensional integration axis.

    Available rules are composite five-point Boole, Gauss-Legendre, the
    ordinary trapezoidal rule, and the endpoint-free periodic trapezoidal
    rule. Boole is the default and requires ``points = 4 * n + 1``.
    """

    start: float
    stop: float
    points: int
    rule: Literal["boole", "gauss", "trapezoid", "periodic"] = "boole"

    def nodes_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return integration nodes and weights in the global real dtype."""
        start = REAL_DTYPE(self.start)
        stop = REAL_DTYPE(self.stop)
        points = self.points

        if self.rule == "gauss":
            nodes, weights = np.polynomial.legendre.leggauss(points)
            nodes = 0.5 * (stop - start) * nodes + 0.5 * (start + stop)
            weights = 0.5 * (stop - start) * weights
            return (
                np.asarray(nodes, dtype=REAL_DTYPE),
                np.asarray(weights, dtype=REAL_DTYPE),
            )

        if self.rule == "periodic":
            nodes = np.linspace(
                start, stop, points, endpoint=False, dtype=REAL_DTYPE
            )
            weights = np.full(points, (stop - start) / points, dtype=REAL_DTYPE)
            return nodes, weights

        if self.rule == "trapezoid":
            nodes = np.linspace(start, stop, points, dtype=REAL_DTYPE)
            weights = np.full(
                points, (stop - start) / (points - 1), dtype=REAL_DTYPE
            )
            weights[[0, -1]] *= 0.5
            return nodes, weights

        if self.rule != "boole":
            raise ValueError(
                "Axis.rule must be 'boole', 'gauss', 'trapezoid', or 'periodic'."
            )
        if (points - 1) % 4:
            raise ValueError("Boole quadrature requires points = 4 * n + 1.")

        nodes = np.linspace(start, stop, points, dtype=REAL_DTYPE)
        step = (stop - start) / (points - 1)
        weights = np.zeros(points, dtype=REAL_DTYPE)
        panel_weights = (
            2.0
            * step
            / 45.0
            * np.asarray((7.0, 32.0, 12.0, 32.0, 7.0), dtype=REAL_DTYPE)
        )
        for panel_start in range(0, points - 1, 4):
            weights[panel_start : panel_start + 5] += panel_weights
        return nodes, weights


def vector(values, size: int, name: str = "vector") -> np.ndarray:
    """Convert values to vectors and verify only their trailing dimension."""
    array = np.asarray(values, dtype=REAL_DTYPE)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}).")
    return array


def scalar_product(first, second):
    """Return the Euclidean scalar product along the final array axis."""
    return np.sum(first * second, axis=-1)


def transverse_projection(values, sign: int):
    """Return the scalar product with the transverse basis vector e_+ or e_-."""
    return values[..., 0] + 1j * sign * values[..., 1]


def sequential_map(
    function: Callable[[Input], Result],
    values: Iterable[Input],
    *,
    progress: bool = False,
    description: str | None = None,
) -> list[Result]:
    """Evaluate values sequentially, optionally displaying progress."""
    total = len(values) if isinstance(values, Sized) else None
    progress_bar = None
    if progress:
        try:
            from tqdm.auto import tqdm as progress_bar
        except ImportError as error:
            raise ImportError(
                "Progress display requires the 'notebook' optional dependencies. "
                "Install them with: pip install -e '.[notebook]'"
            ) from error

    def with_progress(iterable):
        if not progress:
            return iterable
        return progress_bar(
            iterable,
            total=total,
            desc=description,
            unit="point",
            dynamic_ncols=True,
        )

    return [function(value) for value in with_progress(values)]
