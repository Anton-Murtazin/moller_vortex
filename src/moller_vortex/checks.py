"""Ordinary numerical check functions for the research workflow.

The functions in this module do not implement pass/fail tests. They compute
comparison errors and return them as plain dictionaries. This is intentional:
in exploratory work it is often more useful to see the actual numerical errors
than to hide them behind a testing framework or an acceptance threshold.
"""

from __future__ import annotations

import numpy as np

from .accuracy import ACCURACY, NumericalAccuracy
from .kinematics import relative_error
from .packets import LGPacket, normalization_constant, spherical_normalization_constant
from .smatrix import S_impulse_closed_form, S_impulse_numeric_transverse_quad
from .transverse import (
    laguerre_derivative,
    laguerre_derivative_sum,
    transverse_integral_explicit,
    transverse_integral_numeric_quad,
)


def _print_errors(title: str, errors: dict[str, float]) -> None:
    """Print a compact table of numerical errors."""
    print(title)
    for name, error in errors.items():
        print(f"  {name:<45} {error:.6e}")


def check_normalization(
    accuracy: NumericalAccuracy | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Return errors for normalization in the spherical analytic limit.

    The numerical on-axis normalization formula is compared with the closed
    expression valid at sigma_perp = sigma_par. No acceptance threshold is
    applied; the function only returns the numerical relative errors.
    """
    accuracy = ACCURACY if accuracy is None else accuracy

    packets = [
        LGPacket(ell=0, sigma_perp=0.70, sigma_par=0.70, kbar_z=4.0),
        LGPacket(ell=2, sigma_perp=0.70, sigma_par=0.70, kbar_z=4.0),
        LGPacket(ell=-3, sigma_perp=0.85, sigma_par=0.85, kbar_z=6.0),
    ]

    errors = {}
    for packet in packets:
        N_numeric = normalization_constant(packet, accuracy=accuracy)
        N_closed = spherical_normalization_constant(packet)
        key = f"spherical normalization, ell={packet.ell}"
        errors[key] = relative_error(N_numeric, N_closed)

    if verbose:
        _print_errors("Normalization errors", errors)

    return errors


def check_laguerre_derivative(verbose: bool = True) -> dict[str, float]:
    """Return errors for the Laguerre derivative formula against the direct sum.

    This check monitors the sign convention in
    d_t1^a d_t2^b exp(c1*t1 + c2*t2 - c12*t1*t2).
    """
    c1 = 0.31 - 0.17j
    c2 = -0.42 + 0.23j
    c12 = 0.58 + 0.11j

    cases = [(0, 0), (0, 3), (3, 0), (1, 4), (4, 1), (3, 3)]
    errors = {}
    for a, b in cases:
        closed = laguerre_derivative(a, b, c1, c2, c12)
        direct = laguerre_derivative_sum(a, b, c1, c2, c12)
        errors[f"Laguerre derivative, a={a}, b={b}"] = relative_error(closed, direct)

    if verbose:
        _print_errors("Laguerre-derivative errors", errors)

    return errors


def check_transverse_integral(
    n_phi: int = 16,
    accuracy: NumericalAccuracy | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Return errors for closed transverse expressions.

    Each closed expression is compared with direct polar quadrature. No
    acceptance threshold is applied; the function only returns numerical
    relative errors.
    """
    accuracy = ACCURACY if accuracy is None else accuracy

    k3_perp = np.array([1.10, 0.45])
    K_perp = np.array([0.25, -0.18])
    b_perp = np.array([0.08, -0.04])

    alpha = 2.2 + 0.0j
    beta = 0.3 + 0.0j
    gamma = 1.6 + 0.0j

    cases = [
        (1, 2),
        (-1, -2),
        (1, -2),
        (-1, 2),
        (0, 2),
        (0, -2),
        (1, 0),
        (-1, 0),
        (0, 0),
        (2, -1),
    ]

    errors = {}
    for ell1, ell2 in cases:
        analytic = transverse_integral_explicit(
            ell1,
            ell2,
            k3_perp,
            K_perp,
            b_perp,
            alpha,
            beta,
            gamma,
        )
        numeric = transverse_integral_numeric_quad(
            ell1,
            ell2,
            k3_perp,
            K_perp,
            b_perp,
            alpha,
            beta,
            gamma,
            n_phi=n_phi,
            accuracy=accuracy,
        )
        key = f"transverse integral, ell1={ell1}, ell2={ell2}"
        errors[key] = relative_error(analytic, numeric)

    if verbose:
        _print_errors("Transverse-integral errors", errors)

    return errors


def check_smatrix(
    n_phi: int = 16,
    accuracy: NumericalAccuracy | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Return error for closed impulse S matrix vs numerical transverse integration."""
    accuracy = ACCURACY if accuracy is None else accuracy

    packet1 = LGPacket(ell=1, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
    packet2 = LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)

    N1 = normalization_constant(packet1, accuracy=accuracy)
    N2 = normalization_constant(packet2, accuracy=accuracy)

    impact_b = np.array([0.3, 0.0])
    k3 = np.array([0.8, 0.10, 19.7])
    k4 = np.array([-0.55, -0.08, -19.6])

    S_closed = S_impulse_closed_form(
        k3,
        k4,
        packet1,
        packet2,
        lam1=0.5,
        lam2=0.5,
        lam3=0.5,
        lam4=0.5,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
        accuracy=accuracy,
    )

    S_numeric = S_impulse_numeric_transverse_quad(
        k3,
        k4,
        packet1,
        packet2,
        lam1=0.5,
        lam2=0.5,
        lam3=0.5,
        lam4=0.5,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
        n_phi=n_phi,
        accuracy=accuracy,
    )

    errors = {
        "S closed vs numerical transverse": relative_error(S_closed, S_numeric),
    }

    if verbose:
        _print_errors("S-matrix errors", errors)

    return errors


def run_all_checks(
    n_phi: int = 16,
    accuracy: NumericalAccuracy | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Run all built-in numerical comparisons and return their errors."""
    accuracy = ACCURACY if accuracy is None else accuracy

    results = {
        "normalization": check_normalization(
            accuracy=accuracy,
            verbose=verbose,
        ),
        "laguerre_derivative": check_laguerre_derivative(
            verbose=verbose,
        ),
        "transverse_integral": check_transverse_integral(
            n_phi=n_phi,
            accuracy=accuracy,
            verbose=verbose,
        ),
        "smatrix": check_smatrix(
            n_phi=n_phi,
            accuracy=accuracy,
            verbose=verbose,
        ),
    }

    return results
