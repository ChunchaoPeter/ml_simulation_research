"""
Test suite for Exact Daum-Huang (EDH) Filter implementation.

Includes unit tests and integration tests for edh.py
"""

import pytest
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from edh import EDHFilter
from acoustic_function import simulate_trajectory


class TestEDHUnit:
    """Unit tests for individual EDH methods."""

    def test_initialization(self, acoustic_system_simple):
        """Test EDH initialization with correct dimensions."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            lambda_ratio=1.2,
            use_local=False,
            use_ekf=False,
            verbose=False
        )

        assert edh.n_particle == 50
        assert edh.n_lambda == 10
        assert edh.lambda_ratio == 1.2
        assert edh.use_local == False
        assert edh.use_ekf == False

    def test_lambda_steps_computation(self, acoustic_system_simple):
        """Test that lambda steps are correctly computed."""
        system = acoustic_system_simple

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            lambda_ratio=1.2,
            verbose=False
        )

        # Check that lambda values sum to 1
        assert abs(tf.reduce_sum(edh.lambda_steps).numpy() - 1.0) < 1e-5

        # Check that lambda_values are cumulative
        expected_cumsum = tf.cumsum(edh.lambda_steps)
        assert tf.reduce_all(tf.abs(edh.lambda_values - expected_cumsum) < 1e-6)

    def test_particle_initialization(self, acoustic_system_simple):
        """Test particle initialization from Gaussian prior."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=100,
            verbose=False
        )

        particles, m0 = edh.initialize(model_params)

        # Check shapes
        assert particles.shape == (model_params['state_dim'], edh.n_particle)
        assert m0.shape == (model_params['state_dim'], 1)

        # Check that particles are not all identical
        assert tf.reduce_any(particles[:, 0] != particles[:, 1])

    def test_particle_propagation(self, acoustic_system_simple):
        """Test that particles are correctly propagated through motion model."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            verbose=False
        )

        particles, _ = edh.initialize(model_params)
        particles_pred = edh._propagate_particles(particles, model_params)

        # Check shapes
        assert particles_pred.shape == particles.shape

        # Particles should have changed due to propagation and noise
        assert not tf.reduce_all(tf.abs(particles_pred - particles) < 1e-6)

    def test_covariance_estimation(self, acoustic_system_simple):
        """Test particle covariance estimation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=100,
            verbose=False
        )

        particles, _ = edh.initialize(model_params)
        P = edh._estimate_covariance(particles, model_params)

        # Check shape
        assert P.shape == (model_params['state_dim'], model_params['state_dim'])

        # Check positive definiteness
        eigenvalues = tf.linalg.eigvalsh(P)
        assert tf.reduce_all(eigenvalues > 0)

        # Check symmetry
        assert tf.reduce_all(tf.abs(P - tf.transpose(P)) < 1e-5)

    def test_observation_jacobian_computation(self, acoustic_system_simple):
        """Test observation Jacobian computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            verbose=False
        )

        # Create a test state
        x = tf.constant([[10.0], [5.0], [0.1], [0.1], [20.0], [15.0], [-0.1], [0.05]], dtype=tf.float32)

        H = edh.compute_observation_jacobian(x, model_params)

        # Check shape
        assert H.shape == (model_params['n_sensors'], model_params['state_dim'])

        # Check that it's not all zeros
        assert tf.reduce_any(tf.abs(H) > 1e-6)

    def test_flow_parameters_computation(self, acoustic_system_simple):
        """Test computation of flow parameters A and b."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            verbose=False
        )

        particles, _ = edh.initialize(model_params)
        P = edh._estimate_covariance(particles, model_params)
        mean_state = tf.reduce_mean(particles, axis=1, keepdims=True)

        # Generate a measurement
        measurement = system['observation_model_fn'](mean_state, model_params, no_noise=True)

        # Compute flow parameters
        A, b = edh._compute_flow_parameters(
            mean_state,
            mean_state,
            P,
            measurement,
            lam=0.5,
            model_params=model_params
        )

        # Check shapes
        assert A.shape == (model_params['state_dim'], model_params['state_dim'])
        assert b.shape == (model_params['state_dim'],)

    def test_covariance_regularization(self, acoustic_system_simple):
        """Test covariance matrix regularization."""
        system = acoustic_system_simple

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            verbose=False
        )

        # Create a nearly singular matrix
        cov = tf.constant([
            [1.0, 0.999, 0.0, 0.0],
            [0.999, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1e-10, 0.0],
            [0.0, 0.0, 0.0, 1e-10]
        ], dtype=tf.float32)

        # Regularize
        cov_reg = edh._cov_regularize(cov)

        # Check that all eigenvalues are positive
        eigenvalues = tf.linalg.eigvalsh(cov_reg)
        assert tf.reduce_all(eigenvalues > 0)


class TestEDHIntegration:
    """Integration tests for full EDH workflow."""

    def test_single_filter_step(self, acoustic_system_simple):
        """Test a single EDH filter step."""
        system = acoustic_system_simple
        model_params = system['model_params']

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        # Initialize
        edh.initialize(model_params)

        # Generate a measurement from true state
        x_true = model_params['x0_initial_target_states']
        x_next = system['state_transition_fn'](x_true, model_params, use_real_noise=True)
        measurement = system['observation_model_fn'](x_next, model_params)

        # Perform filter step
        particles, mean_estimate, P = edh.step(measurement, model_params)

        # Check output shapes
        assert particles.shape == (model_params['state_dim'], edh.n_particle)
        assert mean_estimate.shape == (model_params['state_dim'],)
        assert P.shape == (model_params['state_dim'], model_params['state_dim'])

    def test_filter_with_acoustic_data(self, acoustic_system_simple):
        """Test filter with simulated acoustic trajectory."""
        system = acoustic_system_simple
        model_params = system['model_params']

        # Generate trajectory
        T = 10
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Initialize filter
        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        # Run filter
        estimates, particles_all, covariances_all = edh.run(observations, model_params)

        # Check output shapes
        assert estimates.shape == (model_params['state_dim'], T)
        assert particles_all.shape == (model_params['state_dim'], edh.n_particle, T)
        assert covariances_all.shape == (model_params['state_dim'], model_params['state_dim'], T)

    def test_filter_no_nans_or_infs(self, acoustic_system_simple):
        """Test that filter produces no NaN or Inf values."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 10
        true_states, observations = simulate_trajectory(model_params, T=T)

        edh = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        estimates, particles_all, covariances_all = edh.run(observations, model_params)

        # Check for NaNs and Infs
        assert not tf.reduce_any(tf.math.is_nan(estimates))
        assert not tf.reduce_any(tf.math.is_inf(estimates))
        assert not tf.reduce_any(tf.math.is_nan(particles_all))
        assert not tf.reduce_any(tf.math.is_inf(particles_all))

    def test_filter_reproducibility(self, acoustic_system_simple):
        """Test that filter produces reproducible results with same seed."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 5
        tf.random.set_seed(123)
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Run filter twice with same seed
        edh1 = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        tf.random.set_seed(456)
        estimates1, _, _ = edh1.run(observations, model_params)

        edh2 = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        tf.random.set_seed(456)
        estimates2, _, _ = edh2.run(observations, model_params)

        # Results should be identical
        assert tf.reduce_all(tf.abs(estimates1 - estimates2) < 1e-5)

    def test_local_vs_global_linearization(self, acoustic_system_simple):
        """Test that local and global linearization both work."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 5
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Global linearization (EDH)
        edh_global = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            use_local=False,
            verbose=False
        )

        tf.random.set_seed(789)
        estimates_global, _, _ = edh_global.run(observations, model_params)

        # Local linearization (LEDH)
        edh_local = EDHFilter(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            n_particle=50,
            n_lambda=10,
            use_local=True,
            verbose=False
        )

        tf.random.set_seed(789)
        estimates_local, _, _ = edh_local.run(observations, model_params)

        # Both should produce valid estimates
        assert not tf.reduce_any(tf.math.is_nan(estimates_global))
        assert not tf.reduce_any(tf.math.is_nan(estimates_local))

        # Estimates may differ but should be in similar range
        assert estimates_global.shape == estimates_local.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
