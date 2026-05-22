"""Basic vector checks and on-shell kinematics."""

from __future__ import annotations

import numpy as np

from .constants import ELECTRON_MASS, RealArray, real_array


def vec2(v) -> RealArray:
    """Return a transverse vector as a float64 array.

    Parameters
    ----------
    v:
        Array-like object with shape ``(2,)``.

    Returns
    -------
    RealArray
        The input converted to project real dtype.
    """
    return real_array(v, shape=(2,))


def vec3(v) -> RealArray:
    """Return a three-momentum vector as a float64 array.

    Parameters
    ----------
    v:
        Array-like object with shape ``(3,)``.

    Returns
    -------
    RealArray
        The input converted to project real dtype.
    """
    return real_array(v, shape=(3,))


def energy(k, m: float = ELECTRON_MASS) -> float:
    """Return on-shell energy.

    Parameters
    ----------
    k:
        Momentum three-vector.
    m:
        Particle mass.

    Returns
    -------
    float
        ``sqrt(m**2 + dot(k, k))``.
    """
    k = vec3(k)
    if m < 0.0:
        raise ValueError("Mass must be non-negative.")
    return np.sqrt(m * m + np.dot(k, k))


def helicity(lam: float) -> float:
    """Validate an electron helicity label.

    Parameters
    ----------
    lam:
        Candidate helicity label.

    Returns
    -------
    float
        The validated helicity, either ``+0.5`` or ``-0.5``.
    """
    if lam == 0.5 or lam == -0.5:
        return lam
    raise ValueError("Helicity must be exactly +0.5 or -0.5.")


def kron_delta(a, b) -> int:
    """Return Kronecker delta for discrete labels.

    Parameters
    ----------
    a, b:
        Values to compare.

    Returns
    -------
    int
        ``1`` if the values are equal, otherwise ``0``.
    """
    return int(a == b)


def absolute_error(a: complex, b: complex) -> float:
    """Return absolute error ``|a - b|``.

    Parameters
    ----------
    a, b:
        Values to compare.

    Returns
    -------
    float
        Absolute difference.
    """
    return abs(a - b)


def relative_error(a: complex, b: complex) -> float:
    """Return relative error with ``b`` as the reference value.

    Parameters
    ----------
    a:
        Candidate value.
    b:
        Reference value.

    Returns
    -------
    float
        ``|a - b| / |b|``. If ``b == 0``, returns ``0`` when both values are
        zero and ``inf`` otherwise.

    If the reference value is exactly zero, the relative error is undefined.
    The function returns 0 for a = b = 0 and infinity otherwise.
    """
    if b == 0:
        return 0.0 if a == 0 else np.inf
    return abs(a - b) / abs(b)
