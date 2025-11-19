"""
Test suite for Particle Filter implementation.

Includes unit tests and integration tests for pf.py
Based on pf_demo.ipynb
"""

import pytest
import tensorflow as tf
from pf import ParticleFilter


class TestPFUnit:
    """Unit tests for individual PF methods."""

    def test_initialization(self, range_bearing_system_for_pf):
        """Test PF initialization with correct dimensions."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        assert pf.num_particles == 100
        assert pf.dtype == tf.float64
        assert pf.resample_method == 'multinomial'

    def test_effective_sample_size(self, range_bearing_system_for_pf):
        """Test effective sample size computation."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        # Uniform weights should give ESS = N
        uniform_weights = tf.ones(100, dtype=tf.float64) / 100.0
        ess = pf.effective_sample_size(uniform_weights)
        assert abs(ess.numpy() - 100.0) < 1e-5

    def test_predict_changes_particles(self, range_bearing_system_for_pf):
        """Test that prediction step changes particle positions."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        # Create initial particles
        tf.random.set_seed(42)
        particles = x0_sampler(100)

        # Predict
        particles_pred = pf.predict(particles)

        assert particles_pred.shape == particles.shape
        # Particles should have moved (with high probability)
        assert not tf.reduce_all(tf.abs(particles_pred - particles) < 1e-6)

    def test_update_produces_normalized_weights(self, range_bearing_system_for_pf, assert_allclose):
        """Test that update step produces normalized weights."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        # Create particles and observation
        tf.random.set_seed(42)
        particles = x0_sampler(100)
        z = observation_fn(x0[:, tf.newaxis])

        # Update
        weights = pf.update(z, particles)

        # Weights should sum to 1
        assert_allclose(tf.reduce_sum(weights), tf.constant(1.0, dtype=tf.float64))
        # All weights should be non-negative
        assert tf.reduce_all(weights >= 0)

    def test_resample_produces_valid_indices(self, range_bearing_system_for_pf):
        """Test that resampling produces valid particle indices."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        weights = tf.ones(100, dtype=tf.float64) / 100.0
        indices = pf.multinomial_resample(weights)

        # Check that indices are in valid range
        assert indices.shape[0] == 100
        assert tf.reduce_all(indices >= 0)
        assert tf.reduce_all(indices < 100)


class TestPFIntegration:
    """Integration tests for full PF workflow."""

    def test_filter_with_range_bearing_data(self, range_bearing_system_for_pf):
        """Test filter produces correct output shapes with range-bearing observations."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        # Generate observations
        T = 20
        true_states, observations = model.simulate_trajectory(T=T)

        # Initialize and run filter
        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        filtered_states = pf.filter(observations)

        # Check output shapes
        assert filtered_states.shape == (4, T + 1)
        assert true_states.shape == (4, T + 1)
        assert observations.shape == (2, T)

    def test_filter_with_details(self, range_bearing_system_for_pf):
        """Test filter returns all detailed outputs correctly."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        # Generate observations
        T = 20
        _, observations = model.simulate_trajectory(T=T)

        # Initialize and run filter with details
        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        results = pf.filter(observations, return_details=True)
        (filtered_states, predicted_states, particles_history,
         weights_history, ess_history, ancestry_history) = results

        # Check all output shapes
        assert filtered_states.shape == (4, T + 1)
        assert predicted_states.shape == (4, T)
        assert particles_history.shape == (4, 100, T + 1)
        assert weights_history.shape == (100, T + 1)
        assert ess_history.shape == (T + 1,)
        assert ancestry_history.shape == (100, T + 1)

    def test_filter_no_nans_or_infs(self, range_bearing_system_for_pf):
        """Test that filter produces no NaN or Inf values."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        T = 20
        _, observations = model.simulate_trajectory(T=T)

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        filtered_states = pf.filter(observations)

        assert not tf.reduce_any(tf.math.is_nan(filtered_states))
        assert not tf.reduce_any(tf.math.is_inf(filtered_states))

    def test_filter_reproducibility(self, range_bearing_system_for_pf, assert_equal):
        """Test that filter produces reproducible results with same seed."""
        (model, state_transition_fn, observation_fn,
         process_noise_sampler, observation_likelihood_fn, x0_sampler,
         x0, Sigma0) = range_bearing_system_for_pf

        tf.random.set_seed(123)
        _, observations = model.simulate_trajectory(T=10)

        pf = ParticleFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            process_noise_sampler=process_noise_sampler,
            observation_likelihood_fn=observation_likelihood_fn,
            x0_sampler=x0_sampler,
            num_particles=100,
            dtype=tf.float64
        )

        # Run filter twice with same observations
        tf.random.set_seed(456)
        filtered_states1 = pf.filter(observations)

        tf.random.set_seed(456)
        filtered_states2 = pf.filter(observations)

        assert_equal(filtered_states1, filtered_states2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
