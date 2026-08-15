import numpy as np

import moller_vortex as mv


def packets():
    return (
        mv.VortexPacket(ell=1, sigma_perp=0.20, sigma_parallel=0.35, sigma_energy=0.30, k0_z=10.0),
        mv.VortexPacket(ell=-2, sigma_perp=0.25, sigma_parallel=0.40, sigma_energy=0.32, k0_z=-10.0),
    )


def numeric_transverse(k3, K, packet1, packet2, impact, points=96):
    """Independent Cartesian quadrature of Eqs. (29) and (32)."""
    x, wx = mv.Axis(-1.8, 1.8, points).nodes_weights()
    y, wy = mv.Axis(-1.8, 1.8, points).nodes_weights()
    x1 = x[:, None]
    y1 = y[None, :]
    x2 = K[0] - x1
    y2 = K[1] - y1

    def vortex(x_value, y_value, ell):
        if ell == 0:
            return np.ones(np.broadcast(x_value, y_value).shape, dtype=np.complex128)
        sign = 1 if ell > 0 else -1
        return (x_value + 1j * sign * y_value) ** abs(ell)

    gaussian = np.exp(
        -(x1**2 + y1**2) / (2.0 * packet1.sigma_perp**2)
        -(x2**2 + y2**2) / (2.0 * packet2.sigma_perp**2)
        + 1j * (impact[0] * x2 + impact[1] * y2)
    )
    k3_squared = np.dot(k3, k3)
    propagator = (1.0 + 2.0 * (k3[0] * x1 + k3[1] * y1) / k3_squared) / k3_squared
    integrand = (
        vortex(x1, y1, packet1.ell)
        * vortex(x2, y2, packet2.ell)
        * gaussian
        * propagator
    )
    return np.sum(wx[:, None] * wy[None, :] * integrand)


def test_axis_integrates_polynomial():
    x, w = mv.Axis(-2.0, 3.0, 5).nodes_weights()
    assert np.isclose(np.sum(w * x**8), (3.0**9 - (-2.0) ** 9) / 9.0)


def test_normalization_converges():
    packet = mv.VortexPacket(
        ell=3,
        sigma_perp=0.08,
        sigma_parallel=0.20,
        sigma_energy=0.15,
        k0_z=10.0,
    )
    coarse = mv.NormalizationGrid(
        mv.Axis(0.0, 0.8, 40), mv.Axis(8.8, 11.2, 48)
    )
    fine = mv.NormalizationGrid(
        mv.Axis(0.0, 0.8, 64), mv.Axis(8.8, 11.2, 80)
    )
    N_coarse = mv.normalization(packet, coarse)
    N_fine, info = mv.normalization(packet, fine, return_info=True)
    assert np.isclose(N_coarse, N_fine, rtol=2e-9)
    assert np.isclose(N_fine**2 * info["integral_without_norm"], 1.0, rtol=2e-14)

    k_perp, w_perp = fine.k_perp.nodes_weights()
    k_z, w_z = fine.k_z.nodes_weights()
    KP, KZ = np.meshgrid(k_perp, k_z, indexing="ij")
    momenta = np.stack((KP, np.zeros_like(KP), KZ), axis=-1)
    direct = np.sum(
        w_perp[:, None]
        * w_z[None, :]
        * KP
        * np.abs(mv.wave_packet(momenta, packet, norm=N_fine)) ** 2
        / (8.0 * np.pi**2 * mv.energy(momenta))
    )
    assert np.isclose(direct, 1.0, rtol=2e-14)


def test_closed_transverse_matches_independent_quadrature():
    k3 = np.array((1.2, 0.4))
    K = np.array((0.12, -0.07))
    impact = np.array((0.30, -0.20))
    cases = ((1, 2), (-1, -2), (-1, 2), (2, -1), (0, 2), (-2, 0), (0, 0))
    for ell1, ell2 in cases:
        packet1 = mv.VortexPacket(ell1, 0.20, 0.35, 0.30, 10.0)
        packet2 = mv.VortexPacket(ell2, 0.25, 0.40, 0.32, -10.0)
        analytic = mv.transverse_integral(
            k3, K, packet1, packet2, impact=impact
        )
        numeric = numeric_transverse(k3, K, packet1, packet2, impact)
        assert np.isclose(analytic, numeric, rtol=2e-11, atol=2e-13), (ell1, ell2)


def test_s_matrix_vectorization_and_helicity_delta():
    packet1, packet2 = packets()
    norms = (1.2, 1.4)
    k3 = np.array((0.8, 0.1, 9.95))
    k4 = np.array((-0.7, -0.08, -9.96))
    scalar = mv.s_matrix(k3, k4, packet1, packet2, norms=norms)
    vector = mv.s_matrix(
        np.stack((k3, k3)), np.stack((k4, k4)), packet1, packet2, norms=norms
    )
    assert np.allclose(vector, scalar)
    assert mv.s_matrix(
        k3,
        k4,
        packet1,
        packet2,
        norms=norms,
        helicities=(0.5, -0.5, -0.5, -0.5),
    ) == 0.0j


def test_pdf_approximation_numbers():
    packet1 = mv.VortexPacket(0, 0.01, 0.05, 0.05, 10.0)
    packet2 = mv.VortexPacket(0, 0.01, 0.05, 0.05, -10.0)
    report = mv.approximation_parameters(packet1, packet2)
    assert np.isclose(report["packet1"]["central_energy"], 10.01304748)
    assert np.isclose(report["packet1"]["mass_over_energy"], 0.05103331)
    assert np.isclose(report["packet1"]["velocity"], 0.99869695)


def test_probability_and_mean_are_finite():
    packet1 = mv.VortexPacket(0, 0.02, 0.12, 0.10, 4.0)
    packet2 = mv.VortexPacket(0, 0.02, 0.12, 0.10, -4.0)
    grid = mv.ProbabilityGrid(
        k3_perp=mv.Axis(0.20, 0.35, 5),
        k3_phi=mv.Axis(0.0, 2.0 * np.pi, 8, "periodic"),
        k3_z=mv.Axis(3.7, 4.3, 6),
        k4_z=mv.Axis(-4.3, -3.7, 6),
        K_perp=mv.Axis(0.0, 0.08, 5),
        K_phi=mv.Axis(0.0, 2.0 * np.pi, 8, "periodic"),
        batch_size=16,
    )
    total, mean = mv.total_probability(
        packet1, packet2, grid, norms=(1.0, 1.0), return_mean=True
    )
    assert np.isfinite(total) and total > 0.0
    assert mean.shape == (3,)
    assert np.all(np.isfinite(mean))
    assert abs(mean[0]) < 1.0e-12
    assert abs(mean[1]) < 1.0e-12

    serial = mv.differential_probability_grid(
        [-0.01, 0.01],
        [-0.01, 0.01],
        packet1,
        packet2,
        grid,
        norms=(1.0, 1.0),
        workers=1,
    )
    parallel = mv.differential_probability_grid(
        [-0.01, 0.01],
        [-0.01, 0.01],
        packet1,
        packet2,
        grid,
        norms=(1.0, 1.0),
        workers=2,
    )
    assert np.array_equal(serial, parallel)


def test_probability_matches_direct_phase_space_sum():
    packet1 = mv.VortexPacket(0, 0.03, 0.12, 0.10, 4.0)
    packet2 = mv.VortexPacket(0, 0.03, 0.12, 0.10, -4.0)
    grid = mv.ProbabilityGrid(
        k3_perp=mv.Axis(0.20, 0.35, 4),
        k3_phi=mv.Axis(0.0, 2.0 * np.pi, 6, "periodic"),
        k3_z=mv.Axis(3.7, 4.3, 5),
        k4_z=mv.Axis(-4.3, -3.7, 5),
        batch_size=16,
    )
    K = np.array((0.01, -0.01))
    optimized = mv.differential_probability(
        K, packet1, packet2, grid, norms=(1.0, 1.0)
    )

    radius, wr = grid.k3_perp.nodes_weights()
    phi, wp = grid.k3_phi.nodes_weights()
    z3, wz3 = grid.k3_z.nodes_weights()
    z4, wz4 = grid.k4_z.nodes_weights()
    k3x = radius[:, None, None, None] * np.cos(phi)[None, :, None, None]
    k3y = radius[:, None, None, None] * np.sin(phi)[None, :, None, None]
    k3z = z3[None, None, :, None]
    k4x = K[0] - k3x
    k4y = K[1] - k3y
    k4z = z4[None, None, None, :]
    components = np.broadcast_arrays(k3x, k3y, k3z, k4x, k4y, k4z)
    k3 = np.stack(components[:3], axis=-1)
    k4 = np.stack(components[3:], axis=-1)
    S = mv.s_matrix(k3, k4, packet1, packet2, norms=(1.0, 1.0))
    phase_space_weight = (
        wr[:, None, None, None]
        * wp[None, :, None, None]
        * wz3[None, None, :, None]
        * wz4[None, None, None, :]
        * radius[:, None, None, None]
    )
    explicit = np.sum(
        phase_space_weight
        * np.abs(S) ** 2
        / (4.0 * (2.0 * np.pi) ** 6 * mv.energy(k3) * mv.energy(k4))
    )
    assert np.isclose(optimized, explicit, rtol=2.0e-14)
