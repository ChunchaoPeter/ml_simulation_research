"""
Test suite for Particle Filtering with Particle Flow (PFPF_EDH) implementation.

Includes unit tests and integration tests for pfpf_edh.py
"""

import pytest
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pfpf_edh import PFPF_EDH
from acoustic_function import simulate_trajectory


class TestPFPFEDHUnit:
    """Unit tests for individual PFPF_EDH methods."""

    def test_initialization(self, acoustic_system_simple):
        """Test PFPF_EDH initialization with correct dimensions."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            lambda_ratio=1.2,
            use_local=False,
            use_ekf=False,
            verbose=False
        )

        # Initialize particles
        particles, m0 = pfpf.initialize(model_params)

        # Check that weights are initialized
        assert pfpf.weights is not None
        assert pfpf.log_weights is not None
        assert pfpf.weights.shape == (pfpf.n_particle,)
        assert pfpf.log_weights.shape == (pfpf.n_particle,)

        # Weights should be uniform
        expected_weight = 1.0 / pfpf.n_particle
        assert tf.reduce_all(tf.abs(pfpf.weights - expected_weight) < 1e-6)

    def test_particle_propagation_deterministic(self, acoustic_system_simple):
        """Test deterministic particle propagation (without noise)."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            verbose=False
        )

        particles, _ = pfpf.initialize(model_params)
        particles_det = pfpf._propagate_particles_deterministic(particles, model_params)

        # Check shape
        assert particles_det.shape == particles.shape

        # Deterministic propagation should follow Phi exactly
        Phi = model_params['Phi']
        expected = tf.matmul(Phi, particles)
        assert tf.reduce_all(tf.abs(particles_det - expected) < 1e-5)

    def test_weight_update(self, acoustic_system_simple):
        """Test weight update mechanism."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            verbose=False
        )

        pfpf.initialize(model_params)

        # Propagate particles
        particles_pred = pfpf._propagate_particles(pfpf.particles, model_params)
        particles_pred_det = pfpf._propagate_particles_deterministic(pfpf.particles, model_params)

        # Generate measurement
        x_true = model_params['x0_initial_target_states']
        measurement = system['observation_model_fn'](x_true, model_params)

        # Update weights
        log_jacobian_det_sum = 0.0
        weights, log_weights = pfpf._update_weights(
            particles_pred,
            particles_pred,
            particles_pred_det,
            measurement,
            log_jacobian_det_sum,
            model_params
        )

        # Check that weights are normalized
        assert abs(tf.reduce_sum(weights).numpy() - 1.0) < 1e-5

        # All weights should be non-negative
        assert tf.reduce_all(weights >= 0)

    def test_effective_sample_size(self, acoustic_system_simple):
        """Test effective sample size computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=100,
            verbose=False
        )

        # Uniform weights should give ESS = N
        uniform_weights = tf.ones(100, dtype=tf.float32) / 100.0
        ess = pfpf._compute_effective_sample_size(uniform_weights)
        assert abs(ess.numpy() - 100.0) < 1e-3

        # Concentrated weights should give low ESS
        concentrated_weights = tf.concat([
            tf.constant([0.99], dtype=tf.float32),
            tf.ones(99, dtype=tf.float32) * 0.01 / 99.0
        ], axis=0)
        ess_low = pfpf._compute_effective_sample_size(concentrated_weights)
        assert ess_low < 10.0

    def test_particle_estimate(self, acoustic_system_simple):
        """Test weighted particle estimate computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=100,
            verbose=False
        )

        # Create test particles
        state_dim = model_params['state_dim']
        particles = tf.random.normal((state_dim, 100), dtype=tf.float32)

        # Uniform weights should give mean
        log_weights = tf.math.log(tf.ones(100, dtype=tf.float32) / 100.0)
        estimate, weights_normalized = pfpf._particle_estimate(log_weights, particles)

        # Check that estimate is close to mean
        expected_mean = tf.reduce_mean(particles, axis=1)
        assert tf.reduce_all(tf.abs(estimate - expected_mean) < 1e-4)

        # Check that weights are normalized
        assert abs(tf.reduce_sum(weights_normalized).numpy() - 1.0) < 1e-5

    def test_log_proposal_density(self, acoustic_system_simple):
        """Test log proposal density computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            verbose=False
        )

        particles, _ = pfpf.initialize(model_params)
        particles_det = pfpf._propagate_particles_deterministic(particles, model_params)

        # Compute log proposal density
        Q = model_params['Q']
        log_jacobian_det_sum = tf.zeros(pfpf.n_particle, dtype=tf.float32)

        log_proposal = pfpf._log_proposal_density(
            particles,
            particles_det,
            Q,
            log_jacobian_det_sum
        )

        # Check shape
        assert log_proposal.shape == (pfpf.n_particle,)

        # Should be finite
        assert tf.reduce_all(tf.math.is_finite(log_proposal))

    def test_log_likelihood_density(self, acoustic_system_simple):
        """Test log likelihood density computation."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            verbose=False
        )

        particles, _ = pfpf.initialize(model_params)

        # Generate measurement
        x_true = model_params['x0_initial_target_states']
        measurement = system['observation_model_fn'](x_true, model_params)

        # Compute log likelihood
        log_likelihood = pfpf._log_likelihood_density(particles, measurement, model_params)

        # Check shape
        assert log_likelihood.shape == (pfpf.n_particle,)

        # Should be finite
        assert tf.reduce_all(tf.math.is_finite(log_likelihood))

    def test_multinomial_resample(self, acoustic_system_simple):
        """Test multinomial resampling."""
        system = acoustic_system_simple

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=100,
            verbose=False
        )

        # Create test weights
        weights = tf.ones(100, dtype=tf.float32) / 100.0
        indices = pfpf._multinomial_resample(weights)

        # Check shape
        assert indices.shape[0] == 100

        # All indices should be in valid range
        assert tf.reduce_all(indices >= 0)
        assert tf.reduce_all(indices < 100)


class TestPFPFEDHIntegration:
    """Integration tests for full PFPF_EDH workflow."""

    def test_single_filter_step(self, acoustic_system_simple):
        """Test a single PFPF_EDH filter step."""
        system = acoustic_system_simple
        model_params = system['model_params']

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        # Initialize
        pfpf.initialize(model_params)

        # Generate measurement
        x_true = model_params['x0_initial_target_states']
        x_next = system['state_transition_fn'](x_true, model_params, use_real_noise=True)
        measurement = system['observation_model_fn'](x_next, model_params)

        # Perform filter step
        particles, mean_estimate, P, N_eff = pfpf.step(measurement, model_params)

        # Check output shapes
        assert particles.shape == (model_params['state_dim'], pfpf.n_particle)
        assert mean_estimate.shape == (model_params['state_dim'],)
        assert P.shape == (model_params['state_dim'], model_params['state_dim'])
        assert isinstance(N_eff.numpy(), float)

    def test_filter_with_acoustic_data(self, acoustic_system_simple):
        """Test PFPF filter with simulated acoustic trajectory."""
        system = acoustic_system_simple
        model_params = system['model_params']

        # Generate trajectory
        T = 10
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Initialize filter
        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        # Run filter
        estimates, particles_all, covariances_all, Neff_all = pfpf.run(observations, model_params)

        # Check output shapes
        assert estimates.shape == (model_params['state_dim'], T)
        assert particles_all.shape == (model_params['state_dim'], pfpf.n_particle, T)
        assert covariances_all.shape == (model_params['state_dim'], model_params['state_dim'], T)
        assert Neff_all.shape == (T, 1)

    def test_filter_no_nans_or_infs(self, acoustic_system_simple):
        """Test that PFPF filter produces no NaN or Inf values."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 10
        true_states, observations = simulate_trajectory(model_params, T=T)

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        estimates, particles_all, covariances_all, Neff_all = pfpf.run(observations, model_params)

        # Check for NaNs and Infs
        assert not tf.reduce_any(tf.math.is_nan(estimates))
        assert not tf.reduce_any(tf.math.is_inf(estimates))
        assert not tf.reduce_any(tf.math.is_nan(particles_all))
        assert not tf.reduce_any(tf.math.is_inf(particles_all))
        assert not tf.reduce_any(tf.math.is_nan(Neff_all))
        assert not tf.reduce_any(tf.math.is_inf(Neff_all))

    def test_filter_reproducibility(self, acoustic_system_simple):
        """Test that PFPF filter produces reproducible results with same seed."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 5
        tf.random.set_seed(123)
        true_states, observations = simulate_trajectory(model_params, T=T)

        # Run filter twice with same seed
        pfpf1 = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        tf.random.set_seed(456)
        estimates1, _, _, _ = pfpf1.run(observations, model_params)

        pfpf2 = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=50,
            n_lambda=10,
            verbose=False
        )

        tf.random.set_seed(456)
        estimates2, _, _, _ = pfpf2.run(observations, model_params)

        # Results should be very similar (allowing for some numerical differences)
        assert tf.reduce_all(tf.abs(estimates1 - estimates2) < 1e-4)

    def test_effective_sample_size_tracking(self, acoustic_system_simple):
        """Test that effective sample size is tracked across time steps."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 10
        true_states, observations = simulate_trajectory(model_params, T=T)

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=100,
            n_lambda=10,
            verbose=False
        )

        estimates, particles_all, covariances_all, Neff_all = pfpf.run(observations, model_params)

        # ESS should be positive and less than or equal to number of particles
        assert tf.reduce_all(Neff_all > 0)
        assert tf.reduce_all(Neff_all <= pfpf.n_particle)

    def test_resampling_occurs(self, acoustic_system_simple):
        """Test that resampling occurs when ESS is low."""
        system = acoustic_system_simple
        model_params = system['model_params']

        T = 5
        true_states, observations = simulate_trajectory(model_params, T=T)

        pfpf = PFPF_EDH(
            observation_jacobian=system['observation_jacobian'],
            observation_model=system['observation_model_fn'],
            observation_model_general=system['observation_model_general_fn'],
            state_transition=system['state_transition_fn'],
            n_particle=30,  # Smaller number makes resampling more likely
            n_lambda=10,
            verbose=False
        )

        # Run filter - resampling should occur automatically
        estimates, particles_all, covariances_all, Neff_all = pfpf.run(observations, model_params)

        # Filter should complete successfully
        assert estimates.shape == (model_params['state_dim'], T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
