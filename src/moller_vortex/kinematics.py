"""Basic vector checks and on-shell kinematics."""

from __future__ import annotations

import numpy as np

from .constants import ELECTRON_MASS, FLOAT_DTYPE, RealArray


def vec2(v) -> RealArray:
    """Return a transverse two-vector as a float64 NumPy array."""
    arr = np.asarray(v, dtype=FLOAT_DTYPE)
    if arr.shape != (2,):
        raise ValueError(f"Expected a transverse vector with shape (2,), got {arr.shape}.")
    return arr


def vec3(v) -> RealArray:
    """Return a three-vector as a float64 NumPy array."""
    arr = np.asarray(v, dtype=FLOAT_DTYPE)
    if arr.shape != (3,):
        raise ValueError(f"Expected a three-vector with shape (3,), got {arr.shape}.")
    return arr


def energy(k, m: float = ELECTRON_MASS) -> float:
    """On-shell energy E = sqrt(m^2 + k^2)."""
    k = vec3(k)
    if m < 0.0:
        raise ValueError("Mass must be non-negative.")
    return np.sqrt(m * m + np.dot(k, k))


def helicity(lam: float) -> float:
    """Validate an electron helicity label."""
    if lam == 0.5 or lam == -0.5:
        return lam
    raise ValueError("Helicity must be exactly +0.5 or -0.5.")


def kron_delta(a, b) -> int:
    """Kronecker delta for discrete labels."""
    return int(a == b)


def absolute_error(a: complex, b: complex) -> float:
    """Absolute difference |a - b|."""
    return abs(a - b)


def relative_error(a: complex, b: complex) -> float:
    """Relative error |a - b| / |b| with b used as the reference value.

    If the reference value is exactly zero, the relative error is undefined.
    The function returns 0 for a = b = 0 and infinity otherwise.
    """
    if b == 0:
        return 0.0 if a == 0 else np.inf
    return abs(a - b) / abs(b)
