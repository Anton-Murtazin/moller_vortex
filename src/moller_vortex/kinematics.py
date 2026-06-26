"""Basic vector validation and on-shell kinematics."""

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
    return np.sqrt(m ** 2 + np.dot(k, k))


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
