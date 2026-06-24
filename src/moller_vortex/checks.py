"""Ordinary numerical check functions for the research workflow.

The functions in this module do not implement pass/fail tests. They compute
comparison errors and return them as plain dictionaries. This is intentional:
in exploratory work it is often more useful to see the actual numerical errors
than to hide them behind a testing framework or an acceptance threshold.
"""

from __future__ import annotations

import numpy as np

from .constants import HBARC_MEV_NM, real_array, spatial_width_nm_to_momentum_mev
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
)
from .transverse import (
    laguerre_derivative,
    laguerre_derivative_sum,
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
    from .probability import diff_probability, diff_probability_grid

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
        n_theta=64,
        n_kappa=5,
        kappa_method="boole",
        kappa_n_sigma=10.0,
    )
    exact_analytic_quadrature = ExactTimeQuadrature(
        n_theta=8,
        n_kappa=5,
        theta_method="analytic",
        kappa_method="boole",
        kappa_n_sigma=10.0,
    )
    exact_massive_quadrature = ExactTimeQuadrature(
        n_theta=16,
        n_kappa=5,
        theta_method="trapezoid",
        kappa_method="boole",
        kappa_n_sigma=10.0,
    )
    exact_reference_quadrature = ExactTimeQuadrature(
        n_theta=512,
        n_kappa=5,
        theta_method="trapezoid",
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
    exact_analytic_grid = S_exact_time_grid(
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
        quadrature=exact_analytic_quadrature,
        batch_size=2,
    )
    exact_analytic_scalar = np.array(
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
                quadrature=exact_analytic_quadrature,
                return_details=True,
            )[0]
            for k3_i, k4_i in zip(exact_k3, exact_k4)
        ]
    )
    exact_massive_grid = S_exact_time_grid(
        exact_k3[:, 0],
        exact_k3[:, 1],
        exact_k3[:, 2],
        exact_k4[:, 0],
        exact_k4[:, 1],
        exact_k4[:, 2],
        exact_packet1,
        exact_packet2,
        lam1=0.5,
        lam2=-0.5,
        lam3=0.5,
        lam4=-0.5,
        impact_b=exact_impact_b,
        N1=exact_N1,
        N2=exact_N2,
        quadrature=exact_massive_quadrature,
        denominator_mode="minkowski",
        batch_size=2,
        matrix_element="paraxial_massive",
    )
    exact_massive_scalar = np.array(
        [
            S_exact_time(
                k3_i,
                k4_i,
                exact_packet1,
                exact_packet2,
                lam1=0.5,
                lam2=-0.5,
                lam3=0.5,
                lam4=-0.5,
                impact_b=exact_impact_b,
                N1=exact_N1,
                N2=exact_N2,
                quadrature=exact_massive_quadrature,
                denominator_mode="minkowski",
                matrix_element="paraxial_massive",
                return_details=True,
            )[0]
            for k3_i, k4_i in zip(exact_k3, exact_k4)
        ]
    )
    exact_reference_grid = S_exact_time_grid(
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
        quadrature=exact_reference_quadrature,
        batch_size=2,
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
        "S exact-time analytic grid vs scalar": _max_relative_grid_error(
            exact_analytic_grid,
            exact_analytic_scalar,
        ),
        "S exact-time massive grid vs scalar": _max_relative_grid_error(
            exact_massive_grid,
            exact_massive_scalar,
        ),
        "S exact-time analytic theta vs numeric theta": _max_relative_grid_error(
            exact_analytic_grid,
            exact_reference_grid,
        ),
    }

    for label, s_matrix, s_kwargs in (
        ("closed", "closed", None),
        ("first_order", "first_order", None),
        (
            "exact_time",
            "exact_time",
            {"quadrature": exact_analytic_quadrature, "batch_size": 8},
        ),
        (
            "exact_time massive",
            "exact_time",
            {
                "matrix_element": "paraxial_massive",
                "denominator_mode": "minkowski",
                "quadrature": exact_massive_quadrature,
                "batch_size": 4,
            },
        ),
    ):
        fast = diff_probability(
            [0.0, 0.0],
            s_matrix=s_matrix,
            s_matrix_kwargs=s_kwargs,
            **probability_kwargs,
        )
        grid = diff_probability_grid(
            np.array([0.0]),
            np.array([0.0]),
            s_matrix=s_matrix,
            s_matrix_kwargs=s_kwargs,
            **probability_kwargs,
        )[0, 0]
        errors[f"probability {label} grid vs point"] = relative_error(
            grid,
            fast,
        )

    if verbose:
        _print_errors("Vectorized-path errors", errors)

    return errors


def check_massive_ur_limit(verbose: bool = True) -> dict[str, float]:
    """Return errors for the massive exact-time branch in the UR limit.

    The paraxial massive t-channel matrix element must reduce to the old
    ultrarelativistic impulse result at large longitudinal momentum.  This
    check uses Gauss-Legendre radial nodes; low-order Boole rules can be far
    from converged for this exact-time radial integral.
    """
    p = 1000.0
    packet1 = LGPacket(
        ell=5,
        sigma_perp=spatial_width_nm_to_momentum_mev(10.0),
        sigma_par=spatial_width_nm_to_momentum_mev(5.0),
        kbar_z=p,
    )
    packet2 = LGPacket(
        ell=0,
        sigma_perp=spatial_width_nm_to_momentum_mev(2.0),
        sigma_par=spatial_width_nm_to_momentum_mev(1.0),
        kbar_z=-p,
    )
    N1 = normalization_constant(packet1)
    N2 = normalization_constant(packet2)
    impact_b = real_array([5.0 / HBARC_MEV_NM, 0.0], shape=(2,))

    rho = 0.030
    phi = 0.7
    K_perp = real_array([5.0e-5, -5.0e-5], shape=(2,))
    k3_perp = real_array([rho * np.cos(phi), rho * np.sin(phi)], shape=(2,))
    k4_perp = K_perp - k3_perp
    k3 = real_array([k3_perp[0], k3_perp[1], p], shape=(3,))
    k4 = real_array([k4_perp[0], k4_perp[1], -p], shape=(3,))

    kwargs = dict(
        k3=k3,
        k4=k4,
        packet1=packet1,
        packet2=packet2,
        lam1=-0.5,
        lam2=0.5,
        lam3=-0.5,
        lam4=0.5,
        impact_b=impact_b,
        N1=N1,
        N2=N2,
    )
    exact_quadrature = ExactTimeQuadrature(
        n_theta=64,
        n_kappa=33,
        theta_method="trapezoid",
        kappa_method="legendre",
        kappa_n_sigma=10.0,
    )

    closed = S_impulse_closed_form(**kwargs)
    exact_expanded = S_exact_time(
        **kwargs,
        quadrature=exact_quadrature,
        denominator_mode="expanded",
    )
    exact_massive = S_exact_time(
        **kwargs,
        quadrature=exact_quadrature,
        denominator_mode="minkowski",
        matrix_element="paraxial_massive",
    )

    errors = {
        "UR exact-time vs closed impulse": relative_error(
            exact_expanded,
            closed,
        ),
        "massive exact-time vs closed impulse": relative_error(
            exact_massive,
            closed,
        ),
        "massive exact-time vs UR exact-time": relative_error(
            exact_massive,
            exact_expanded,
        ),
    }

    if verbose:
        _print_errors("Massive-UR-limit errors", errors)

    return errors


def run_all_checks(
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Run all built-in numerical comparisons and return their errors.

    Parameters
    ----------
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
        "vectorized_paths": check_vectorized_paths(
            verbose=verbose,
        ),
        "massive_ur_limit": check_massive_ur_limit(
            verbose=verbose,
        ),
    }

    return results
