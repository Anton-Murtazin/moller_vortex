"""Closed and direct numerical transverse integrals."""

from __future__ import annotations

import math

import numpy as np
from .accuracy import ACCURACY, NumericalAccuracy
from .constants import COMPLEX_DTYPE, FLOAT_DTYPE, PI
from .kinematics import vec2


def laguerre_derivative(a: int, b: int, c1: complex, c2: complex, c12: complex) -> complex:
    """Return d_t1^a d_t2^b exp(c1*t1 + c2*t2 - c12*t1*t2) at zero.

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


def laguerre_derivative_sum(a: int, b: int, c1: complex, c2: complex, c12: complex) -> complex:
    """Direct finite-sum version of ``laguerre_derivative``.

    This is useful for diagnostics because it follows immediately from the
    Taylor expansion of exp(c1*t1 + c2*t2 - c12*t1*t2) and does not use
    Laguerre polynomials.
    """
    if a < 0 or b < 0:
        raise ValueError("Derivative orders must be non-negative.")

    value = 0.0 + 0.0j
    for r in range(min(a, b) + 1):
        coeff = (
            math.factorial(a)
            * math.factorial(b)
            / (math.factorial(a - r) * math.factorial(b - r) * math.factorial(r))
        )
        value += coeff * (-c12) ** r * c1 ** (a - r) * c2 ** (b - r)
    return value


def transverse_integral_explicit(
    ell1: int,
    ell2: int,
    k3_perp,
    K_perp,
    b_perp,
    alpha: complex,
    beta: complex,
    gamma: complex,
    return_case: bool = False,
) -> complex | tuple[complex, str]:
    """Closed first-order transverse integral.

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
        raise ZeroDivisionError("The expanded transverse denominator requires k3_perp != 0.")

    K = np.array([K_real[0], K_real[1]], dtype=COMPLEX_DTYPE)
    b = np.array([b_real[0], b_real[1]], dtype=COMPLEX_DTYPE)

    A = alpha + gamma
    if A == 0.0:
        raise ZeroDivisionError("alpha + gamma is exactly zero.")

    phi3 = np.arctan2(k3p[1], k3p[0])
    chi_plus = np.exp(-1j * phi3) / k3_abs
    chi_minus = np.exp(1j * phi3) / k3_abs

    J0 = (beta + gamma) * K - 1j * b
    J0_sq = J0[0] * J0[0] + J0[1] * J0[1]
    K_sq = K[0] * K[0] + K[1] * K[1]

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
        value = prefactor * bracket
        case = "ell1=0, ell2=0"
        return (value, case) if return_case else value

    if ell1 == 0:
        m = abs(ell2)

        if ell2 > 0:
            bracket = q_plus ** m
            bracket += chi_plus * p_plus * q_plus ** m
            bracket += chi_minus * (p_minus * q_plus ** m - 2.0 * m * q_plus ** (m - 1) / A)
            case = "ell1=0, ell2>0"
        else:
            bracket = q_minus ** m
            bracket += chi_plus * (p_plus * q_minus ** m - 2.0 * m * q_minus ** (m - 1) / A)
            bracket += chi_minus * p_minus * q_minus ** m
            case = "ell1=0, ell2<0"

        value = prefactor * bracket
        return (value, case) if return_case else value

    if ell2 == 0:
        n = abs(ell1)

        if ell1 > 0:
            bracket = p_plus ** n
            bracket += chi_plus * p_plus ** (n + 1)
            bracket += chi_minus * (p_minus * p_plus ** n + 2.0 * n * p_plus ** (n - 1) / A)
            case = "ell1>0, ell2=0"
        else:
            bracket = p_minus ** n
            bracket += chi_plus * (p_plus * p_minus ** n + 2.0 * n * p_minus ** (n - 1) / A)
            bracket += chi_minus * p_minus ** (n + 1)
            case = "ell1<0, ell2=0"

        value = prefactor * bracket
        return (value, case) if return_case else value

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
        value = prefactor * bracket
        case = "ell1>0, ell2>0"
        return (value, case) if return_case else value

    if ell1 < 0 and ell2 < 0:
        bracket = p_minus ** n * q_minus ** m
        bracket += chi_plus * (
            p_plus * p_minus ** n * q_minus ** m
            + 2.0 * n * p_minus ** (n - 1) * q_minus ** m / A
            - 2.0 * m * p_minus ** n * q_minus ** (m - 1) / A
        )
        bracket += chi_minus * p_minus ** (n + 1) * q_minus ** m
        value = prefactor * bracket
        case = "ell1<0, ell2<0"
        return (value, case) if return_case else value

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
        value = prefactor * bracket
        case = "ell1>0, ell2<0"
        return (value, case) if return_case else value

    c1 = p_minus
    c2 = q_plus
    K_opposite = K_plus
    bracket = laguerre_derivative(n, m, c1, c2, mu)
    bracket += chi_minus * laguerre_derivative(n + 1, m, c1, c2, mu)
    bracket += chi_plus * (
        K_opposite * laguerre_derivative(n, m, c1, c2, mu)
        - laguerre_derivative(n, m + 1, c1, c2, mu)
    )
    value = prefactor * bracket
    case = "ell1<0, ell2>0"
    return (value, case) if return_case else value


def vortex_factor(z: complex, ell: int) -> complex:
    """Return z^ell for ell>0, conjugate(z)^|ell| for ell<0, and 1 for ell=0."""
    if ell > 0:
        return z ** ell
    if ell < 0:
        return np.conjugate(z) ** abs(ell)
    return 1.0 + 0.0j


def transverse_integral_numeric_quad(
    ell1: int,
    ell2: int,
    k3_perp,
    K_perp,
    b_perp,
    alpha: complex,
    beta: complex,
    gamma: complex,
    n_phi: int = 64,
    accuracy: NumericalAccuracy | None = None,
) -> complex:
    """Direct polar numerical check of the same first-order transverse integral."""
    from scipy.integrate import quad

    accuracy = ACCURACY if accuracy is None else accuracy

    k3p = vec2(k3_perp)
    Kp = vec2(K_perp)
    bp = vec2(b_perp)

    k3_abs = np.linalg.norm(k3p)
    if k3_abs == 0.0:
        raise ZeroDivisionError("The expanded transverse denominator requires k3_perp != 0.")

    phi3 = np.arctan2(k3p[1], k3p[0])
    exp_minus_i_phi3 = np.exp(-1j * phi3)
    exp_plus_i_phi3 = np.exp(1j * phi3)

    phis = np.linspace(0.0, 2.0 * PI, n_phi, endpoint=False)
    dphi = 2.0 * PI / n_phi

    total = 0.0 + 0.0j

    for phi in phis:
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        def radial_integrand(r: float) -> complex:
            k = np.array([r * cos_phi, r * sin_phi], dtype=FLOAT_DTYPE)
            K_minus_k = Kp - k

            z1 = k[0] + 1j * k[1]
            z2 = K_minus_k[0] + 1j * K_minus_k[1]

            vortex1 = vortex_factor(z1, ell1)
            vortex2 = vortex_factor(z2, ell2)

            exponent = (
                -alpha * np.dot(k, k) / 2.0
                + beta * np.dot(k, Kp)
                - gamma * np.dot(K_minus_k, K_minus_k) / 2.0
                - 1j * np.dot(bp, k)
            )

            denominator_expansion = (
                1.0
                + exp_minus_i_phi3 * z1 / k3_abs
                + exp_plus_i_phi3 * np.conjugate(z1) / k3_abs
            ) / k3_abs ** 2

            return r * vortex1 * vortex2 * np.exp(exponent) * denominator_expansion

        real_part = quad(
            lambda r: np.real(radial_integrand(r)),
            0.0,
            np.inf,
            epsabs=accuracy.quad_epsabs,
            epsrel=accuracy.quad_epsrel,
            limit=accuracy.quad_limit,
        )[0]

        imag_part = quad(
            lambda r: np.imag(radial_integrand(r)),
            0.0,
            np.inf,
            epsabs=accuracy.quad_epsabs,
            epsrel=accuracy.quad_epsrel,
            limit=accuracy.quad_limit,
        )[0]

        total += dphi * (real_part + 1j * imag_part)

    return total
