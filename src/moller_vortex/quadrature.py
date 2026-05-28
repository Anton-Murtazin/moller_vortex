"""Shared deterministic quadrature rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import HBARC_MEV_NM, real_full, real_zeros


_DEFAULT_SIGMA1_PAR = HBARC_MEV_NM / 5.0
_DEFAULT_SIGMA2_PAR = HBARC_MEV_NM / 1.0


def legendre_nodes_and_weights(interval: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Gauss-Legendre nodes and weights on a finite interval.

    Parameters
    ----------
    interval:
        Integration interval ``(a, b)``.
    n:
        Number of quadrature nodes; must be positive.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Nodes and weights mapped to ``interval``.
    """
    a, b = interval
    n = int(n)
    if n <= 0:
        raise ValueError("Gauss-Legendre quadrature requires n > 0.")

    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w
    return nodes, weights


def boole_nodes_and_weights(interval: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build composite Boole-rule nodes and weights.

    Parameters
    ----------
    interval:
        Integration interval ``(a, b)``.
    n:
        Number of nodes. It must satisfy ``n = 4*m + 1`` and be at least 5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Equally spaced nodes and Boole weights on ``interval``.
    """
    a, b = interval
    n = int(n)

    if n < 5:
        raise ValueError("Composite Boole quadrature requires at least 5 nodes.")

    n_intervals = n - 1
    if n_intervals % 4 != 0:
        raise ValueError("Composite Boole quadrature requires n - 1 divisible by 4.")

    nodes = np.linspace(a, b, n)
    h = (b - a) / n_intervals

    weights = real_zeros(n)
    for j in range(0, n_intervals, 4):
        weights[j] += 7.0
        weights[j + 1] += 32.0
        weights[j + 2] += 12.0
        weights[j + 3] += 32.0
        weights[j + 4] += 7.0

    weights *= 2.0 * h / 45.0
    return nodes, weights


def trapezoid_nodes_and_weights(
    interval: tuple[float, float],
    n: int,
    *,
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build trapezoidal-rule nodes and weights.

    Parameters
    ----------
    interval:
        Integration interval ``(a, b)``.
    n:
        Number of nodes.
    endpoint:
        If False, build the endpoint-free periodic rule on ``[a, b)``.
        If True, include both endpoints and half-weight them.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Trapezoidal nodes and weights.
    """
    a, b = interval
    n = int(n)

    if n <= 0:
        raise ValueError("Trapezoidal quadrature requires n > 0.")

    if endpoint:
        if n < 2:
            raise ValueError("Closed trapezoidal quadrature requires n >= 2.")
        nodes = np.linspace(a, b, n, endpoint=True)
        h = (b - a) / (n - 1)
        weights = real_full(n, h)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return nodes, weights

    nodes = np.linspace(a, b, n, endpoint=False)
    weights = real_full(n, (b - a) / n)
    return nodes, weights


def nodes_and_weights(
    interval: tuple[float, float],
    n: int,
    *,
    method: str = "boole",
    endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build quadrature nodes and weights by method name.

    Parameters
    ----------
    interval:
        Integration interval ``(a, b)``.
    n:
        Number of nodes.
    method:
        One of ``"boole"``, ``"legendre"`` or ``"trapezoid"``.
    endpoint:
        Passed to ``trapezoid_nodes_and_weights`` only.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Nodes and weights for the requested rule.
    """
    if method == "boole":
        return boole_nodes_and_weights(interval, n)
    if method == "legendre":
        return legendre_nodes_and_weights(interval, n)
    if method == "trapezoid":
        return trapezoid_nodes_and_weights(interval, n, endpoint=endpoint)
    raise ValueError("Unknown quadrature method: {!r}".format(method))


@dataclass(frozen=True)
class ExactTimeQuadrature:
    """Settings for S_exact_time transverse disk integration.

    Parameters
    ----------
    n_chi, n_theta:
        Node counts for the chi radial variable and azimuthal angle.
    chi_method, theta_method:
        Quadrature method names passed to ``nodes_and_weights``.
    radial_variable:
        ``"kappa"`` integrates the exact-time disk in the physical radial
        variable ``kappa = |q - q0|``.  ``"chi"`` keeps the older
        regularizing substitution ``kappa = R sin(chi)``.
    n_kappa, kappa_method:
        Node count and quadrature method for the direct kappa formula.  If
        either is ``None``, the corresponding chi setting is used only as a
        backward-compatible node-setting alias.
    kappa_n_sigma:
        For ``radial_variable="kappa"``, integrate only over the transverse
        Gaussian support, using this many effective transverse widths around
        the Gaussian center.  Set to ``None`` to use the full disk radius.

    Returns
    -------
    ExactTimeQuadrature
        Immutable configuration object.  Nodes are built by
        ``exact_time_nodes``; the kappa case also needs a physical radial
        interval.
    """

    n_chi: int = 65
    n_theta: int = 32
    n_kappa: int | None = 65 # 129 is better
    chi_method: str = "boole"
    theta_method: str = "trapezoid"
    radial_variable: str = "kappa"
    kappa_method: str = "boole"
    kappa_n_sigma: float | None = 10.0


@dataclass(frozen=True)
class ProbabilityQuadrature:
    """Settings for probability-level integrations.

    Parameters
    ----------
    k3_perp_range, k3z_range, k4z_range:
        Inner integration ranges used by ``diff_probability``.
    K_perp_range:
        Outer transverse-total-momentum range used by ``total_probability``
        and ``Ky_average``.
    n_k3_perp, n_phi, n_k3z, n_k4z, n_K_perp, n_K_phi:
        Node counts for the corresponding axes.
    *_method:
        Quadrature rule names. Finite non-periodic axes default to
        ``"boole"``; periodic angular axes default to ``"trapezoid"``.

    The default inner ranges and node counts are the current working setup
    for the 10 MeV / -10 MeV packet example.  Override any field locally with
    ``dataclasses.replace`` or by passing explicit constructor arguments.

    Returns
    -------
    ProbabilityQuadrature
        Immutable configuration object.  Actual nodes are built by
        ``probability_inner_nodes`` and ``probability_outer_nodes``.
    """

    k3_perp_range: tuple[float, float] | None = (0.010, 0.050)
    k3z_range: tuple[float, float] | None = (
        10.0 - 50.0 * _DEFAULT_SIGMA1_PAR,
        10.0 + 50.0 * _DEFAULT_SIGMA1_PAR,
    )
    k4z_range: tuple[float, float] | None = (
        -10.0 - 50.0 * _DEFAULT_SIGMA2_PAR,
        -10.0 + 50.0 * _DEFAULT_SIGMA2_PAR,
    )
    K_perp_range: tuple[float, float] | None = None
    n_k3_perp: int | None = 17
    n_phi: int | None = 5
    n_k3z: int | None = 17
    n_k4z: int | None = 17
    n_K_perp: int | None = None
    n_K_phi: int | None = None
    k3_perp_method: str = "boole"
    k3z_method: str = "boole"
    k4z_method: str = "boole"
    K_perp_method: str = "boole"
    phi_method: str = "trapezoid"
    K_phi_method: str = "trapezoid"


def _require_interval(name: str, interval: tuple[float, float] | None) -> tuple[float, float]:
    """Return a configured interval or raise a clear error.

    Parameters
    ----------
    name:
        Field name to display in the error message.
    interval:
        Optional interval value from a quadrature config.

    Returns
    -------
    tuple[float, float]
        The validated interval.
    """
    if interval is None:
        raise ValueError(f"ProbabilityQuadrature.{name} must be set.")
    return interval


def _require_count(name: str, n: int | None) -> int:
    """Return a configured node count or raise a clear error.

    Parameters
    ----------
    name:
        Field name to display in the error message.
    n:
        Optional node count from a quadrature config.

    Returns
    -------
    int
        The validated node count.
    """
    if n is None:
        raise ValueError(f"ProbabilityQuadrature.{name} must be set.")
    return n


def probability_inner_nodes(
    quadrature: ProbabilityQuadrature,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Return nodes for the inner fixed-K_perp probability integral.

    Parameters
    ----------
    quadrature:
        Probability quadrature settings.

    Returns
    -------
    tuple
        Nodes and weights in the order ``k3_perp``, ``phi``, ``k3z``,
        ``k4z``.
    """
    k3_perp, phi = probability_transverse_nodes(quadrature)
    k3z = nodes_and_weights(
        _require_interval("k3z_range", quadrature.k3z_range),
        _require_count("n_k3z", quadrature.n_k3z),
        method=quadrature.k3z_method,
    )
    k4z = nodes_and_weights(
        _require_interval("k4z_range", quadrature.k4z_range),
        _require_count("n_k4z", quadrature.n_k4z),
        method=quadrature.k4z_method,
    )
    return k3_perp, phi, k3z, k4z


def probability_transverse_nodes(
    quadrature: ProbabilityQuadrature,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return transverse nodes for fixed longitudinal final momenta.

    Parameters
    ----------
    quadrature:
        Probability quadrature settings with ``k3_perp_range``, ``n_k3_perp``
        and ``n_phi`` configured.

    Returns
    -------
    tuple
        Nodes and weights in the order ``k3_perp``, ``phi``.
    """
    k3_perp = nodes_and_weights(
        _require_interval("k3_perp_range", quadrature.k3_perp_range),
        _require_count("n_k3_perp", quadrature.n_k3_perp),
        method=quadrature.k3_perp_method,
    )
    phi = nodes_and_weights(
        (0.0, 2.0 * np.pi),
        _require_count("n_phi", quadrature.n_phi),
        method=quadrature.phi_method,
        endpoint=False,
    )
    return k3_perp, phi


def probability_outer_nodes(
    quadrature: ProbabilityQuadrature,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return nodes for the outer K_perp integration.

    Parameters
    ----------
    quadrature:
        Probability quadrature settings.

    Returns
    -------
    tuple
        Nodes and weights in the order ``K_perp``, ``K_phi``.
    """
    K_perp = nodes_and_weights(
        _require_interval("K_perp_range", quadrature.K_perp_range),
        _require_count("n_K_perp", quadrature.n_K_perp),
        method=quadrature.K_perp_method,
    )
    K_phi = nodes_and_weights(
        (0.0, 2.0 * np.pi),
        _require_count("n_K_phi", quadrature.n_K_phi),
        method=quadrature.K_phi_method,
        endpoint=False,
    )
    return K_perp, K_phi


def exact_time_nodes(
    quadrature: ExactTimeQuadrature,
    radial_interval: tuple[float, float] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return radial and theta nodes for S_exact_time.

    Parameters
    ----------
    quadrature:
        Exact-time quadrature settings.
    radial_interval:
        Physical ``kappa`` interval for ``radial_variable="kappa"``.  The
        chi formula does not use this parameter.

    Returns
    -------
    tuple
        Nodes and weights in the order ``radial``, ``theta``.
    """
    if quadrature.radial_variable == "chi":
        radial = nodes_and_weights(
            (0.0, 0.5 * np.pi),
            quadrature.n_chi,
            method=quadrature.chi_method,
            endpoint=True,
        )
    elif quadrature.radial_variable == "kappa":
        if radial_interval is None:
            raise ValueError(
                "radial_interval is required for exact-time kappa nodes."
            )
        n_kappa = quadrature.n_kappa
        kappa_method = quadrature.kappa_method
        if n_kappa is None:
            n_kappa = quadrature.n_chi
        if kappa_method is None:
            kappa_method = quadrature.chi_method
        radial = nodes_and_weights(
            radial_interval,
            n_kappa,
            method=kappa_method,
            endpoint=False,
        )
    else:
        raise ValueError("radial_variable must be 'chi' or 'kappa'.")

    theta = nodes_and_weights(
        (0.0, 2.0 * np.pi),
        quadrature.n_theta,
        method=quadrature.theta_method,
        endpoint=False,
    )
    return radial, theta
