"""Closed transverse integrals."""

from __future__ import annotations

import math

import numpy as np
from .constants import PI, complex_array
from .kinematics import vec2


def laguerre_derivative(a: int, b: int, c1: complex, c2: complex, c12: complex) -> complex:
    """Return d_t1^a d_t2^b exp(c1*t1 + c2*t2 - c12*t1*t2) at zero.

    Parameters
    ----------
    a, b:
        Non-negative derivative orders.
    c1, c2, c12:
        Coefficients in the generating exponential.

    Returns
    -------
    complex
        The differentiated generating function at ``t1 = t2 = 0``.

    The formula is

        D_ab = sum_r a! b! (-c12)^r c1^(a-r) c2^(b-r)
               / [(a-r)! (b-r)! r!],

    equivalently, for b >= a,

        D_ab = (-c12)^a a! c2^(b-a) L_a^(b-a)(c1*c2/c12),

    and with 1 <-> 2 for a > b.  In particular D_0b = c2^b;
    no extra factor (-c2)^(b-a) is present.

    The associated Laguerre polynomial is evaluated by its finite integer-order
    series, so complex arguments are handled without relying on scipy.special.
    """
    if a < 0 or b < 0:
        raise ValueError("Derivative orders must be non-negative.")
    if c12 == 0.0:
        raise ZeroDivisionError("c12 must be non-zero in the Laguerre representation.")

    def associated_laguerre(n: int, k: int, x: complex) -> complex:
        """Return the associated Laguerre polynomial by finite summation.

        Parameters
        ----------
        n:
            Non-negative polynomial order.
        k:
            Non-negative associated index.
        x:
            Real or complex polynomial argument.

        Returns
        -------
        complex
            Value of ``L_n^k(x)``.
        """
        value = 0.0 + 0.0j
        for s in range(n + 1):
            binom = math.factorial(n + k) / (
                math.factorial(n - s) * math.factorial(k + s)
            )
            value += binom * (-x) ** s / math.factorial(s)
        return value

    x = c1 * c2 / c12

    if b >= a:
        return (
            (-c12) ** a
            * math.factorial(a)
            * c2 ** (b - a)
            * associated_laguerre(a, b - a, x)
        )

    return (
        (-c12) ** b
        * math.factorial(b)
        * c1 ** (a - b)
        * associated_laguerre(b, a - b, x)
    )


def transverse_integral(
    ell1: int,
    ell2: int,
    k3_perp,
    K_perp,
    b_perp,
    alpha: complex,
    beta: complex,
    gamma: complex,
) -> complex:
    """Analytic first-order transverse integral.

    Parameters
    ----------
    ell1, ell2:
        Incoming packet OAM integers.
    k3_perp:
        Transverse momentum of final particle 3.
    K_perp:
        Total final transverse momentum.
    b_perp:
        Transverse impact parameter.
    alpha, beta, gamma:
        Transverse Gaussian coefficients.
    Returns
    -------
    complex
        Analytic transverse integral.

    The integral is evaluated for the first-order expansion

        1/|k3_perp-k_perp|^2 = 1/k3^2 [1 + exp(-i phi3) k_+/k3
                                           + exp(+i phi3) k_-/k3],

    where k_+ = k_x + i k_y and k_- = k_x - i k_y.  The returned value
    includes the factors 1/k3^2, 2*pi/(alpha+gamma), and
    exp[J0^2/(2(alpha+gamma)) - gamma*K_perp^2/2].
    """
    k3p = vec2(k3_perp)
    K_real = vec2(K_perp)
    b_real = vec2(b_perp)

    k3_abs = np.linalg.norm(k3p)
    if k3_abs == 0.0:
        raise ZeroDivisionError("The ultrarelativistic transverse denominator requires k3_perp != 0.")

    K = complex_array([K_real[0], K_real[1]], shape=(2,))
    b = complex_array([b_real[0], b_real[1]], shape=(2,))

    A = alpha + gamma
    if A == 0.0:
        raise ZeroDivisionError("alpha + gamma is exactly zero.")

    phi3 = np.arctan2(k3p[1], k3p[0])
    chi_plus = np.exp(-1j * phi3) / k3_abs
    chi_minus = np.exp(1j * phi3) / k3_abs

    J0 = (beta + gamma) * K - 1j * b
    J0_sq = J0[0] ** 2 + J0[1] ** 2
    K_sq = K[0] ** 2 + K[1] ** 2

    p_plus = (J0[0] + 1j * J0[1]) / A
    p_minus = (J0[0] - 1j * J0[1]) / A
    K_plus = K[0] + 1j * K[1]
    K_minus = K[0] - 1j * K[1]
    q_plus = K_plus - p_plus
    q_minus = K_minus - p_minus

    prefactor = (
        2.0 * PI
        / (k3_abs ** 2 * A)
        * np.exp(J0_sq / (2.0 * A) - gamma * K_sq / 2.0)
    )

    if ell1 == 0 and ell2 == 0:
        bracket = 1.0 + chi_plus * p_plus + chi_minus * p_minus
        return prefactor * bracket

    if ell1 == 0:
        m = abs(ell2)

        if ell2 > 0:
            bracket = q_plus ** m
            bracket += chi_plus * p_plus * q_plus ** m
            bracket += chi_minus * (p_minus * q_plus ** m - 2.0 * m * q_plus ** (m - 1) / A)
        else:
            bracket = q_minus ** m
            bracket += chi_plus * (p_plus * q_minus ** m - 2.0 * m * q_minus ** (m - 1) / A)
            bracket += chi_minus * p_minus * q_minus ** m

        return prefactor * bracket

    if ell2 == 0:
        n = abs(ell1)

        if ell1 > 0:
            bracket = p_plus ** n
            bracket += chi_plus * p_plus ** (n + 1)
            bracket += chi_minus * (p_minus * p_plus ** n + 2.0 * n * p_plus ** (n - 1) / A)
        else:
            bracket = p_minus ** n
            bracket += chi_plus * (p_plus * p_minus ** n + 2.0 * n * p_minus ** (n - 1) / A)
            bracket += chi_minus * p_minus ** (n + 1)

        return prefactor * bracket

    n = abs(ell1)
    m = abs(ell2)

    if ell1 > 0 and ell2 > 0:
        bracket = p_plus ** n * q_plus ** m
        bracket += chi_plus * p_plus ** (n + 1) * q_plus ** m
        bracket += chi_minus * (
            p_minus * p_plus ** n * q_plus ** m
            + 2.0 * n * p_plus ** (n - 1) * q_plus ** m / A
            - 2.0 * m * p_plus ** n * q_plus ** (m - 1) / A
        )
        return prefactor * bracket

    if ell1 < 0 and ell2 < 0:
        bracket = p_minus ** n * q_minus ** m
        bracket += chi_plus * (
            p_plus * p_minus ** n * q_minus ** m
            + 2.0 * n * p_minus ** (n - 1) * q_minus ** m / A
            - 2.0 * m * p_minus ** n * q_minus ** (m - 1) / A
        )
        bracket += chi_minus * p_minus ** (n + 1) * q_minus ** m
        return prefactor * bracket

    mu = 2.0 / A

    if ell1 > 0 and ell2 < 0:
        c1 = p_plus
        c2 = q_minus
        K_opposite = K_minus
        bracket = laguerre_derivative(n, m, c1, c2, mu)
        bracket += chi_plus * laguerre_derivative(n + 1, m, c1, c2, mu)
        bracket += chi_minus * (
            K_opposite * laguerre_derivative(n, m, c1, c2, mu)
            - laguerre_derivative(n, m + 1, c1, c2, mu)
        )
        return prefactor * bracket

    c1 = p_minus
    c2 = q_plus
    K_opposite = K_plus
    bracket = laguerre_derivative(n, m, c1, c2, mu)
    bracket += chi_minus * laguerre_derivative(n + 1, m, c1, c2, mu)
    bracket += chi_plus * (
        K_opposite * laguerre_derivative(n, m, c1, c2, mu)
        - laguerre_derivative(n, m + 1, c1, c2, mu)
    )
    return prefactor * bracket


def transverse_integral_grid(
    ell1: int,
    ell2: int,
    k3x,
    k3y,
    K_perp,
    b_perp,
    alpha: complex,
    beta: complex,
    gamma: complex,
):
    """Vectorized analytic first-order transverse integral.

    This is the array-valued counterpart of ``transverse_integral``.
    It keeps exactly the same branch formulas, but accepts broadcastable
    arrays for the final-particle transverse momentum components.
    """
    k3_abs_sq = k3x ** 2 + k3y ** 2
    if np.any(k3_abs_sq == 0.0):
        raise ZeroDivisionError("The ultrarelativistic transverse denominator requires k3_perp != 0.")

    Kx = K_perp[0] + 0.0j
    Ky = K_perp[1] + 0.0j
    bx = b_perp[0] + 0.0j
    by = b_perp[1] + 0.0j

    A = alpha + gamma
    if A == 0.0:
        raise ZeroDivisionError("alpha + gamma is exactly zero.")

    chi_plus = (k3x - 1j * k3y) / k3_abs_sq
    chi_minus = (k3x + 1j * k3y) / k3_abs_sq

    J0x = (beta + gamma) * Kx - 1j * bx
    J0y = (beta + gamma) * Ky - 1j * by
    J0_sq = J0x ** 2 + J0y ** 2
    K_sq = Kx ** 2 + Ky ** 2

    p_plus = (J0x + 1j * J0y) / A
    p_minus = (J0x - 1j * J0y) / A
    K_plus = Kx + 1j * Ky
    K_minus = Kx - 1j * Ky
    q_plus = K_plus - p_plus
    q_minus = K_minus - p_minus

    prefactor = (
        2.0
        * PI
        / (k3_abs_sq * A)
        * np.exp(J0_sq / (2.0 * A) - gamma * K_sq / 2.0)
    )

    if ell1 == 0 and ell2 == 0:
        bracket = 1.0 + chi_plus * p_plus + chi_minus * p_minus
        return prefactor * bracket

    if ell1 == 0:
        m = abs(ell2)
        if ell2 > 0:
            bracket = q_plus ** m
            bracket += chi_plus * p_plus * q_plus ** m
            bracket += chi_minus * (
                p_minus * q_plus ** m
                - 2.0 * m * q_plus ** (m - 1) / A
            )
        else:
            bracket = q_minus ** m
            bracket += chi_plus * (
                p_plus * q_minus ** m
                - 2.0 * m * q_minus ** (m - 1) / A
            )
            bracket += chi_minus * p_minus * q_minus ** m
        return prefactor * bracket

    if ell2 == 0:
        n = abs(ell1)
        if ell1 > 0:
            bracket = p_plus ** n
            bracket += chi_plus * p_plus ** (n + 1)
            bracket += chi_minus * (
                p_minus * p_plus ** n
                + 2.0 * n * p_plus ** (n - 1) / A
            )
        else:
            bracket = p_minus ** n
            bracket += chi_plus * (
                p_plus * p_minus ** n
                + 2.0 * n * p_minus ** (n - 1) / A
            )
            bracket += chi_minus * p_minus ** (n + 1)
        return prefactor * bracket

    n = abs(ell1)
    m = abs(ell2)

    if ell1 > 0 and ell2 > 0:
        bracket = p_plus ** n * q_plus ** m
        bracket += chi_plus * p_plus ** (n + 1) * q_plus ** m
        bracket += chi_minus * (
            p_minus * p_plus ** n * q_plus ** m
            + 2.0 * n * p_plus ** (n - 1) * q_plus ** m / A
            - 2.0 * m * p_plus ** n * q_plus ** (m - 1) / A
        )
        return prefactor * bracket

    if ell1 < 0 and ell2 < 0:
        bracket = p_minus ** n * q_minus ** m
        bracket += chi_plus * (
            p_plus * p_minus ** n * q_minus ** m
            + 2.0 * n * p_minus ** (n - 1) * q_minus ** m / A
            - 2.0 * m * p_minus ** n * q_minus ** (m - 1) / A
        )
        bracket += chi_minus * p_minus ** (n + 1) * q_minus ** m
        return prefactor * bracket

    mu = 2.0 / A

    if ell1 > 0 and ell2 < 0:
        c1 = p_plus
        c2 = q_minus
        K_opposite = K_minus
        bracket = laguerre_derivative(n, m, c1, c2, mu)
        bracket += chi_plus * laguerre_derivative(n + 1, m, c1, c2, mu)
        bracket += chi_minus * (
            K_opposite * laguerre_derivative(n, m, c1, c2, mu)
            - laguerre_derivative(n, m + 1, c1, c2, mu)
        )
        return prefactor * bracket

    c1 = p_minus
    c2 = q_plus
    K_opposite = K_plus
    bracket = laguerre_derivative(n, m, c1, c2, mu)
    bracket += chi_minus * laguerre_derivative(n + 1, m, c1, c2, mu)
    bracket += chi_plus * (
        K_opposite * laguerre_derivative(n, m, c1, c2, mu)
        - laguerre_derivative(n, m + 1, c1, c2, mu)
    )
    return prefactor * bracket
