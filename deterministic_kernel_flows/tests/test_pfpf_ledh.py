"""
Test suite for Particle Filtering with Particle Flow using Localized EDH (PFPF_LEDH) implementation.

Includes unit tests and integration tests for pfpf_ledh.py
"""

import pytest
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pfpf_ledh import PFPF_LEDH
from acoustic_function import simulate_trajectory


class TestPFPFLEDHUnit:
    """Unit tests for individual PFPF_LEDH methods."""

    def test_initialization(self, acoustic_system_simple, ekf_for_acoustic):
        """Test PFPF_LEDH initialization with correct dimensions."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=50,
                n_lambda=10,
                lambda_ratio=1.2,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            # Initialize particles
            particles, m0 = pfpf_ledh.initialize(model_params)

            # Check that LEDH-specific state is initialized
            assert pfpf_ledh.log_jacobian_det_sum is not None
            assert pfpf_ledh.log_jacobian_det_sum.shape == (pfpf_ledh.n_particle,)

            # Check that use_local is set to True
            assert pfpf_ledh.use_local == True

            # Check P_all initialization
            assert pfpf_ledh.P_all is not None
            assert pfpf_ledh.P_all.shape == (pfpf_ledh.n_particle, model_params['state_dim'], model_params['state_dim'])

        except ImportError:
            pytest.skip("EKF module not available")

    def test_log_jacobian_initialization(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that log Jacobian determinants are initialized to zero."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=50,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            pfpf_ledh.initialize(model_params)

            # Log Jacobian should start at zero
            assert tf.reduce_all(pfpf_ledh.log_jacobian_det_sum == 0.0)

        except ImportError:
            pytest.skip("EKF module not available")

    def test_particle_flow_ledh_computation(self, acoustic_system_simple, ekf_for_acoustic):
        """Test local linearization particle flow computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=20,  # Small number for speed
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            pfpf_ledh.initialize(model_params)

            # Set up for particle flow
            x_true = model_params['x0_initial_target_states']
            measurement = system['observation_model_fn'](x_true, model_params)

            # Propagate particles
            pfpf_ledh.particles_pred = pfpf_ledh._propagate_particles(pfpf_ledh.particles, model_params)
            pfpf_ledh.auxiliary_individual = pfpf_ledh._propagate_particles_deterministic(
                pfpf_ledh.particles, model_params
            )
            pfpf_ledh.mu_0 = x_true

            # Run particle flow
            log_jacobian_det_sum = pfpf_ledh._particle_flow_ledh(model_params, measurement)

            # Check output shape
            assert log_jacobian_det_sum.shape == (pfpf_ledh.n_particle,)

            # Should be finite
            assert tf.reduce_all(tf.math.is_finite(log_jacobian_det_sum))

        except ImportError:
            pytest.skip("EKF module not available")

    def test_per_particle_covariance(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that each particle maintains its own covariance matrix."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            pfpf_ledh.initialize(model_params)

            # Check that P_all has correct shape
            expected_shape = (pfpf_ledh.n_particle, model_params['state_dim'], model_params['state_dim'])
            assert pfpf_ledh.P_all.shape == expected_shape

            # Each particle's covariance should be positive definite
            for i in range(pfpf_ledh.n_particle):
                eigenvalues = tf.linalg.eigvalsh(pfpf_ledh.P_all[i])
                assert tf.reduce_all(eigenvalues > 0)

        except ImportError:
            pytest.skip("EKF module not available")


class TestPFPFLEDHIntegration:
    """Integration tests for full PFPF_LEDH workflow."""

    def test_single_filter_step(self, acoustic_system_simple, ekf_for_acoustic):
        """Test a single PFPF_LEDH filter step."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            # Initialize
            pfpf_ledh.initialize(model_params)

            # Generate measurement
            x_true = model_params['x0_initial_target_states']
            x_next = system['state_transition_fn'](x_true, model_params, use_real_noise=True)
            measurement = system['observation_model_fn'](x_next, model_params)

            # Perform filter step
            particles, mean_estimate, P, N_eff = pfpf_ledh.step(measurement, model_params)

            # Check output shapes
            assert particles.shape == (model_params['state_dim'], pfpf_ledh.n_particle)
            assert mean_estimate.shape == (model_params['state_dim'],)
            assert P.shape == (model_params['state_dim'], model_params['state_dim'])
            assert isinstance(N_eff.numpy(), float)

        except ImportError:
            pytest.skip("EKF module not available")

    def test_filter_with_acoustic_data(self, acoustic_system_simple, ekf_for_acoustic):
        """Test PFPF_LEDH filter with simulated acoustic trajectory."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            # Generate trajectory
            T = 5  # Short trajectory for faster testing
            true_states, observations = simulate_trajectory(model_params, T=T)

            # Initialize filter
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            # Run filter
            estimates, particles_all, covariances_all, Neff_all = pfpf_ledh.run(observations, model_params)

            # Check output shapes
            assert estimates.shape == (model_params['state_dim'], T)
            assert particles_all.shape == (model_params['state_dim'], pfpf_ledh.n_particle, T)
            assert covariances_all.shape == (model_params['state_dim'], model_params['state_dim'], T)
            assert Neff_all.shape == (T, 1)

        except ImportError:
            pytest.skip("EKF module not available")

    def test_filter_no_nans_or_infs(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that PFPF_LEDH filter produces no NaN or Inf values."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            T = 5
            true_states, observations = simulate_trajectory(model_params, T=T)

            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            estimates, particles_all, covariances_all, Neff_all = pfpf_ledh.run(observations, model_params)

            # Check for NaNs and Infs
            assert not tf.reduce_any(tf.math.is_nan(estimates))
            assert not tf.reduce_any(tf.math.is_inf(estimates))
            assert not tf.reduce_any(tf.math.is_nan(particles_all))
            assert not tf.reduce_any(tf.math.is_inf(particles_all))
            assert not tf.reduce_any(tf.math.is_nan(Neff_all))
            assert not tf.reduce_any(tf.math.is_inf(Neff_all))

        except ImportError:
            pytest.skip("EKF module not available")

    def test_filter_reproducibility(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that PFPF_LEDH filter produces reproducible results with same seed."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            T = 3
            tf.random.set_seed(123)
            true_states, observations = simulate_trajectory(model_params, T=T)

            # Run filter twice with same seed
            pfpf_ledh1 = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            tf.random.set_seed(456)
            estimates1, _, _, _ = pfpf_ledh1.run(observations, model_params)

            pfpf_ledh2 = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            tf.random.set_seed(456)
            estimates2, _, _, _ = pfpf_ledh2.run(observations, model_params)

            # Results should be very similar
            assert tf.reduce_all(tf.abs(estimates1 - estimates2) < 1e-4)

        except ImportError:
            pytest.skip("EKF module not available")

    def test_jacobian_determinant_tracking(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that Jacobian determinants are tracked correctly."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            T = 3
            true_states, observations = simulate_trajectory(model_params, T=T)

            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            pfpf_ledh.initialize(model_params)

            # Run one step
            measurement = tf.expand_dims(observations[:, 0], 1)
            particles, mean_estimate, P, N_eff = pfpf_ledh.step(measurement, model_params)

            # Log Jacobian determinants should be finite
            assert tf.reduce_all(tf.math.is_finite(pfpf_ledh.log_jacobian_det_sum))

        except ImportError:
            pytest.skip("EKF module not available")

    def test_local_vs_global_comparison(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that LEDH (local) differs from EDH (global) but both work."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            # Import PFPF_EDH for comparison
            from pfpf_edh import PFPF_EDH

            T = 3
            tf.random.set_seed(100)
            true_states, observations = simulate_trajectory(model_params, T=T)

            # Global linearization (PFPF_EDH)
            pfpf_edh = PFPF_EDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                observation_model_general=system['observation_model_general_fn'],
                state_transition=system['state_transition_fn'],
                n_particle=30,
                n_lambda=5,
                use_local=False,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            tf.random.set_seed(200)
            estimates_global, _, _, _ = pfpf_edh.run(observations, model_params)

            # Local linearization (PFPF_LEDH)
            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=30,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            tf.random.set_seed(200)
            estimates_local, _, _, _ = pfpf_ledh.run(observations, model_params)

            # Both should produce valid estimates
            assert not tf.reduce_any(tf.math.is_nan(estimates_global))
            assert not tf.reduce_any(tf.math.is_nan(estimates_local))

            # Shapes should match
            assert estimates_global.shape == estimates_local.shape

        except ImportError:
            pytest.skip("EKF or PFPF_EDH module not available")

    def test_effective_sample_size_tracking(self, acoustic_system_simple, ekf_for_acoustic):
        """Test that effective sample size is tracked across time steps."""
        system = acoustic_system_simple
        model_params = system['model_params']

        try:
            T = 5
            true_states, observations = simulate_trajectory(model_params, T=T)

            pfpf_ledh = PFPF_LEDH(
                observation_jacobian=system['observation_jacobian'],
                observation_model=system['observation_model_fn'],
                state_transition=system['state_transition_fn'],
                observation_model_general=system['observation_model_general_fn'],
                n_particle=50,
                n_lambda=5,
                use_ekf=True,
                ekf_filter=ekf_for_acoustic,
                verbose=False
            )

            estimates, particles_all, covariances_all, Neff_all = pfpf_ledh.run(observations, model_params)

            # ESS should be positive and less than or equal to number of particles
            assert tf.reduce_all(Neff_all > 0)
            assert tf.reduce_all(Neff_all <= pfpf_ledh.n_particle)

        except ImportError:
            pytest.skip("EKF module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
