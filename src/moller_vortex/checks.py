"""Ordinary numerical check functions for the research workflow.

The functions in this module do not implement pass/fail tests. They compute
comparison errors and return them as plain dictionaries. This is intentional:
in exploratory work it is often more useful to see the actual numerical errors
than to hide them behind a testing framework or an acceptance threshold.
"""

from __future__ import annotations

import numpy as np

from .constants import HBARC_MEV_NM, real_array
from .kinematics import relative_error
from .packets import LGPacket, normalization_constant, spherical_normalization_constant
from .quadrature import ExactTimeQuadrature, ProbabilityQuadrature
from .smatrix import (
    S_exact_time,
    S_exact_time_grid,
    S_impulse_closed_form,
    S_impulse_closed_grid,
    S_impulse_first_order,
    S_impulse_first_order_grid,
    S_impulse_numeric_transverse_quad,
)
from .transverse import (
    laguerre_derivative,
    laguerre_derivative_sum,
    transverse_integral_explicit,
    transverse_integral_numeric_quad,
)


def _print_errors(title: str, errors: dict[str, float]) -> None:
    """Print a compact table of numerical errors.

    Parameters
    ----------
    title:
        Table title printed before the errors.
    errors:
        Mapping from check names to relative errors.

    Returns
    -------
    None
        The table is written to standard output.
    """
    print(title)
    for name, error in errors.items():
        print(f"  {name:<45} {error:.6e}")


def _max_relative_grid_error(candidate, reference) -> float:
    """Return the maximum elementwise relative error for array comparisons."""
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    denominator = np.where(reference == 0, 1.0, np.abs(reference))
    return float(np.max(np.abs(candidate - reference) / denominator))


def check_normalization(
    verbose: bool = True,
) -> dict[str, float]:
    """Return errors for normalization in the spherical analytic limit.

    Parameters
    ----------
    verbose:
        If True, print a compact error table.

    Returns
    -------
    dict[str, float]
        Relative errors keyed by check description.

    The numerical on-axis normalization formula is compared with the closed
    expression valid at sigma_perp = sigma_par. No acceptance threshold is
    applied; the function only returns the numerical relative errors.
    """
    packets = [
        LGPacket(ell=0, sigma_perp=0.70, sigma_par=0.70, kbar_z=4.0),
        LGPacket(ell=2, sigma_perp=0.70, sigma_par=0.70, kbar_z=4.0),
        LGPacket(ell=-3, sigma_perp=0.85, sigma_par=0.85, kbar_z=6.0),
    ]

    errors = {}
    for packet in packets:
        N_numeric = normalization_constant(packet)
        N_closed = spherical_normalization_constant(packet)
        key = f"spherical normalization, ell={packet.ell}"
        errors[key] = relative_error(N_numeric, N_closed)

    if verbose:
        _print_errors("Normalization errors", errors)

    return errors


def check_laguerre_derivative(verbose: bool = True) -> dict[str, float]:
    """Return errors for the Laguerre derivative formula against the direct sum.

    Parameters
    ----------
    verbose:
        If True, print a compact error table.

    Returns
    -------
    dict[str, float]
        Relative errors keyed by derivative orders.

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
    verbose: bool = True,
) -> dict[str, float]:
    """Return errors for closed transverse expressions.

    Parameters
    ----------
    n_phi:
        Number of angular nodes for the direct numerical transverse check.
    verbose:
        If True, print a compact error table.

    Returns
    -------
    dict[str, float]
        Relative errors keyed by OAM pair.

    Each closed expression is compared with direct polar quadrature. No
    acceptance threshold is applied; the function only returns numerical
    relative errors.
    """
    k3_perp = real_array([1.10, 0.45], shape=(2,))
    K_perp = real_array([0.25, -0.18], shape=(2,))
    b_perp = real_array([0.08, -0.04], shape=(2,))

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
        )
        key = f"transverse integral, ell1={ell1}, ell2={ell2}"
        errors[key] = relative_error(analytic, numeric)

    if verbose:
        _print_errors("Transverse-integral errors", errors)

    return errors


def check_vectorized_paths(verbose: bool = True) -> dict[str, float]:
    """Return errors for vectorized S-matrix and probability paths.

    Parameters
    ----------
    verbose:
        If True, print a compact error table.

    Returns
    -------
    dict[str, float]
        Relative errors comparing vectorized code against scalar diagnostic
        evaluations on a small deterministic grid.

    This check is not a physics benchmark.  It protects the implementation
    against array-broadcasting, weighting and batching mistakes in the fast
    paths used by probability scans.
    """
    from .probability import diff_probability

    packet1 = LGPacket(ell=1, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
    packet2 = LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)
    N1 = normalization_constant(packet1)
    N2 = normalization_constant(packet2)
    impact_b = real_array([0.3, -0.1], shape=(2,))

    k3 = real_array(
        [
            [0.80, 0.10, 19.70],
            [1.10, -0.20, 20.20],
            [0.55, 0.35, 19.90],
        ]
    )
    k4 = real_array(
        [
            [-0.55, -0.08, -19.60],
            [-0.75, 0.25, -20.10],
            [-0.35, -0.20, -19.80],
        ]
    )

    closed_grid = S_impulse_closed_grid(
        k3[:, 0],
        k3[:, 1],
        k3[:, 2],
        k4[:, 0],
        k4[:, 1],
        k4[:, 2],
        packet1,
        packet2,
        lam1=0.5,
        lam2=0.5,
        lam3=0.5,
        lam4=0.5,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
    )
    closed_scalar = np.array(
        [
            S_impulse_closed_form(
                k3_i,
                k4_i,
                packet1,
                packet2,
                lam1=0.5,
                lam2=0.5,
                lam3=0.5,
                lam4=0.5,
                impact_b=impact_b,
                N1=N1,
                N2=N2,
            )
            for k3_i, k4_i in zip(k3, k4)
        ]
    )

    first_grid = S_impulse_first_order_grid(
        k3[:, 0],
        k3[:, 1],
        k3[:, 2],
        k4[:, 0],
        k4[:, 1],
        k4[:, 2],
        packet1,
        packet2,
        lam1=0.5,
        lam2=0.5,
        lam3=0.5,
        lam4=0.5,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
    )
    first_scalar = np.array(
        [
            S_impulse_first_order(
                k3_i,
                k4_i,
                packet1,
                packet2,
                lam1=0.5,
                lam2=0.5,
                lam3=0.5,
                lam4=0.5,
                impact_b=impact_b,
                N1=N1,
                N2=N2,
            )
            for k3_i, k4_i in zip(k3, k4)
        ]
    )

    exact_packet1 = LGPacket(
        ell=1,
        sigma_perp=HBARC_MEV_NM / 20.0,
        sigma_par=HBARC_MEV_NM / 5.0,
        kbar_z=10.0,
    )
    exact_packet2 = LGPacket(
        ell=-1,
        sigma_perp=HBARC_MEV_NM / 20.0,
        sigma_par=HBARC_MEV_NM / 1.0,
        kbar_z=-10.0,
    )
    exact_N1 = normalization_constant(exact_packet1)
    exact_N2 = normalization_constant(exact_packet2)
    exact_impact_b = real_array([0.2, -0.1], shape=(2,))
    exact_quadrature = ExactTimeQuadrature(
        n_theta=8,
        n_kappa=5,
        radial_variable="kappa",
        kappa_method="boole",
        kappa_n_sigma=10.0,
    )
    exact_k3 = real_array(
        [
            [0.001, 0.000, 10.00002],
            [0.010, 0.002, 10.00000],
            [0.030, -0.001, 9.99998],
        ]
    )
    exact_k4 = real_array(
        [
            [-0.001, -0.000, -9.99990],
            [-0.010, -0.002, -10.00004],
            [-0.030, 0.001, -10.00000],
        ]
    )
    exact_grid = S_exact_time_grid(
        exact_k3[:, 0],
        exact_k3[:, 1],
        exact_k3[:, 2],
        exact_k4[:, 0],
        exact_k4[:, 1],
        exact_k4[:, 2],
        exact_packet1,
        exact_packet2,
        lam1=0.5,
        lam2=0.5,
        lam3=0.5,
        lam4=0.5,
        impact_b=exact_impact_b,
        N1=exact_N1,
        N2=exact_N2,
        quadrature=exact_quadrature,
        batch_size=2,
    )
    exact_scalar = np.array(
        [
            S_exact_time(
                k3_i,
                k4_i,
                exact_packet1,
                exact_packet2,
                lam1=0.5,
                lam2=0.5,
                lam3=0.5,
                lam4=0.5,
                impact_b=exact_impact_b,
                N1=exact_N1,
                N2=exact_N2,
                quadrature=exact_quadrature,
                return_details=True,
            )[0]
            for k3_i, k4_i in zip(exact_k3, exact_k4)
        ]
    )

    probability_quadrature = ProbabilityQuadrature(
        k3_perp_range=(0.010, 0.030),
        k3z_range=(
            10.0 - 2.0 * exact_packet1.sigma_par,
            10.0 + 2.0 * exact_packet1.sigma_par,
        ),
        k4z_range=(
            -10.0 - 2.0 * exact_packet2.sigma_par,
            -10.0 + 2.0 * exact_packet2.sigma_par,
        ),
        n_k3_perp=5,
        n_phi=4,
        n_k3z=5,
        n_k4z=5,
    )
    probability_kwargs = dict(
        packet1=exact_packet1,
        packet2=exact_packet2,
        quadrature=probability_quadrature,
        impact_b=exact_impact_b,
        N1=exact_N1,
        N2=exact_N2,
    )

    errors = {
        "S closed grid vs scalar": _max_relative_grid_error(
            closed_grid,
            closed_scalar,
        ),
        "S first-order grid vs scalar": _max_relative_grid_error(
            first_grid,
            first_scalar,
        ),
        "S exact-time grid vs scalar details": _max_relative_grid_error(
            exact_grid,
            exact_scalar,
        ),
    }

    for s_matrix, s_kwargs in (
        ("closed", None),
        ("first_order", None),
        (
            "exact_time",
            {"quadrature": exact_quadrature, "batch_size": 8},
        ),
    ):
        fast = diff_probability(
            [0.0, 0.0],
            s_matrix=s_matrix,
            s_matrix_kwargs=s_kwargs,
            **probability_kwargs,
        )
        scalar = diff_probability(
            [0.0, 0.0],
            explicit_spin_sum=True,
            s_matrix=s_matrix,
            s_matrix_kwargs=s_kwargs,
            **probability_kwargs,
        )
        errors[f"probability {s_matrix} fast vs scalar"] = relative_error(
            fast,
            scalar,
        )

    if verbose:
        _print_errors("Vectorized-path errors", errors)

    return errors


def check_smatrix(
    n_phi: int = 16,
    verbose: bool = True,
) -> dict[str, float]:
    """Return error for closed impulse S matrix vs numerical transverse integration.

    Parameters
    ----------
    n_phi:
        Number of angular nodes for numerical transverse quadrature.
    verbose:
        If True, print a compact error table.

    Returns
    -------
    dict[str, float]
        Relative error between closed and numerical-transverse S matrices.
    """
    packet1 = LGPacket(ell=1, sigma_perp=0.18, sigma_par=0.35, kbar_z=20.0)
    packet2 = LGPacket(ell=-1, sigma_perp=0.18, sigma_par=0.35, kbar_z=-20.0)

    N1 = normalization_constant(packet1)
    N2 = normalization_constant(packet2)

    impact_b = real_array([0.3, 0.0], shape=(2,))
    k3 = real_array([0.8, 0.10, 19.7], shape=(3,))
    k4 = real_array([-0.55, -0.08, -19.6], shape=(3,))

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
    )

    errors = {
        "S closed vs numerical transverse": relative_error(S_closed, S_numeric),
    }

    if verbose:
        _print_errors("S-matrix errors", errors)

    return errors


def run_all_checks(
    n_phi: int = 16,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Run all built-in numerical comparisons and return their errors.

    Parameters
    ----------
    n_phi:
        Number of angular nodes used by transverse numerical checks.
    verbose:
        If True, print each check table.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested mapping from check group to relative-error values.
    """
    results = {
        "normalization": check_normalization(
            verbose=verbose,
        ),
        "laguerre_derivative": check_laguerre_derivative(
            verbose=verbose,
        ),
        "transverse_integral": check_transverse_integral(
            n_phi=n_phi,
            verbose=verbose,
        ),
        "vectorized_paths": check_vectorized_paths(
            verbose=verbose,
        ),
        "smatrix": check_smatrix(
            n_phi=n_phi,
            verbose=verbose,
        ),
    }

    return results
