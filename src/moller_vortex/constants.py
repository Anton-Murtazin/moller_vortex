"""Numerical constants and unit convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FLOAT_DTYPE = np.float64
COMPLEX_DTYPE = np.complex128

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

PI = np.pi

# Natural units: hbar = c = 1.
# Energies, momenta and masses are measured in MeV.
# Lengths and impact parameters are measured in MeV^{-1}.
ALPHA_EM = 1.0 / 137.035999084
ELECTRON_CHARGE = np.sqrt(4.0 * PI * ALPHA_EM)
ELECTRON_MASS = 0.51099895000

HBARC_MEV_NM = 1.97463e-4  # MeV * nm
NM_TO_MEV_INV = 1.0 / HBARC_MEV_NM


def real_array(values, *, shape: tuple[int, ...] | None = None) -> RealArray:
    """Return values as a float64 NumPy array.

    Parameters
    ----------
    values:
        Any array-like real input.
    shape:
        Optional exact shape validation.

    Returns
    -------
    RealArray
        A NumPy array with dtype ``FLOAT_DTYPE``.
    """
    arr = np.asarray(values, dtype=FLOAT_DTYPE)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}.")
    return arr


def complex_array(values, *, shape: tuple[int, ...] | None = None) -> ComplexArray:
    """Return values as a complex128 NumPy array.

    Parameters
    ----------
    values:
        Any array-like complex input.
    shape:
        Optional exact shape validation.

    Returns
    -------
    ComplexArray
        A NumPy array with dtype ``COMPLEX_DTYPE``.
    """
    arr = np.asarray(values, dtype=COMPLEX_DTYPE)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}.")
    return arr


def real_empty(shape) -> RealArray:
    """Return an uninitialized float64 array with the requested shape.

    Parameters
    ----------
    shape:
        Shape accepted by ``numpy.empty``.

    Returns
    -------
    RealArray
        Uninitialized array with dtype ``FLOAT_DTYPE``.
    """
    return np.empty(shape, dtype=FLOAT_DTYPE)


def real_zeros(shape) -> RealArray:
    """Return a zero-filled float64 array with the requested shape.

    Parameters
    ----------
    shape:
        Shape accepted by ``numpy.zeros``.

    Returns
    -------
    RealArray
        Zero-filled array with dtype ``FLOAT_DTYPE``.
    """
    return np.zeros(shape, dtype=FLOAT_DTYPE)


def real_full(shape, fill_value: float) -> RealArray:
    """Return a float64 array filled with ``fill_value``.

    Parameters
    ----------
    shape:
        Shape accepted by ``numpy.full``.
    fill_value:
        Scalar value used to fill the array.

    Returns
    -------
    RealArray
        Filled array with dtype ``FLOAT_DTYPE``.
    """
    return np.full(shape, fill_value, dtype=FLOAT_DTYPE)


def complex_zero() -> np.complex128:
    """Return scalar complex zero with the project complex dtype.

    Parameters
    ----------
    None

    Returns
    -------
    np.complex128
        Scalar zero with dtype ``COMPLEX_DTYPE``.
    """
    return COMPLEX_DTYPE(0.0)


def spatial_width_nm_to_momentum_mev(width_nm: float) -> float:
    """Convert coordinate width in nm to momentum width in MeV.

    Parameters
    ----------
    width_nm:
        Coordinate-space width in nanometers.

    Returns
    -------
    float
        Momentum-space width in MeV.
    """
    return HBARC_MEV_NM / width_nm
