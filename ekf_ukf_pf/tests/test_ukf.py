"""
Test suite for Unscented Kalman Filter implementation.

Includes unit tests and integration tests for ukf.py
"""

import pytest
import tensorflow as tf
from ukf import UnscentedKalmanFilter


class TestUKFUnit:
    """Unit tests for individual UKF methods."""

    def test_initialization(self, range_bearing_system):
        """Test UKF initialization with correct dimensions."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        assert ukf.state_dim == 4
        assert ukf.obs_dim == 2
        assert ukf.x0.shape == (4, 1)

    def test_initialization_with_1d_vector(self, range_bearing_system):
        """Test that 1D initial state vector is converted to column vector."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        assert len(ukf.x0.shape) == 2
        assert ukf.x0.shape == (4, 1)

    def test_sigma_points_generation(self, range_bearing_system):
        """Test sigma points generation produces correct number of points."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        x = tf.constant([[1.0], [0.0], [1.0], [0.0]], dtype=tf.float32)
        Sigma = tf.eye(4, dtype=tf.float32)

        sigma_points = ukf.generate_sigma_points(x, Sigma)

        expected_num_points = 2 * ukf.state_dim + 1
        assert sigma_points.shape == (4, expected_num_points)

    def test_weights_sum_to_one(self, range_bearing_system, assert_allclose):
        """Test that weights for mean sum to 1."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        sum_wm = tf.reduce_sum(ukf.Wm)
        assert_allclose(sum_wm, tf.constant(1.0))

    def test_predict_increases_uncertainty(self, range_bearing_system):
        """Test that prediction step increases uncertainty."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        x = tf.constant([[1.0], [0.5], [1.0], [0.5]], dtype=tf.float32)
        Sigma = tf.eye(4, dtype=tf.float32)

        _, Sigma_pred, _ = ukf.predict(x, Sigma)

        assert tf.linalg.trace(Sigma_pred) > tf.linalg.trace(Sigma)

    def test_update_reduces_uncertainty(self, range_bearing_system):
        """Test that update step reduces uncertainty."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        x_pred = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        Sigma_pred = tf.eye(4, dtype=tf.float32) * 2.0

        sigma_points_pred = ukf.generate_sigma_points(x_pred, Sigma_pred)
        z = observation_fn(x_pred)

        _, Sigma_post = ukf.update(z, x_pred, Sigma_pred, sigma_points_pred)

        assert tf.linalg.trace(Sigma_post) < tf.linalg.trace(Sigma_pred)

    def test_covariance_positive_definite(self, range_bearing_system):
        """Test that covariances remain positive definite."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0
        )

        x = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        Sigma = tf.eye(4, dtype=tf.float32)

        x_pred, Sigma_pred, sigma_points_pred = ukf.predict(x, Sigma)
        assert tf.reduce_all(tf.linalg.eigvalsh(Sigma_pred) > 0)

        z = observation_fn(x_pred)
        _, Sigma_post = ukf.update(z, x_pred, Sigma_pred, sigma_points_pred)
        assert tf.reduce_all(tf.linalg.eigvalsh(Sigma_post) > 0)

    def test_different_ukf_parameters(self, range_bearing_system):
        """Test UKF with different alpha, beta, kappa values."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        param_sets = [
            {'alpha': 1.0, 'beta': 2.0, 'kappa': 0.0},
            {'alpha': 0.5, 'beta': 2.0, 'kappa': 1.0},
            {'alpha': 1.0, 'beta': 0.0, 'kappa': 3.0},
        ]

        T = 20
        _, observations = model.simulate_trajectory(T=T)

        for params in param_sets:
            ukf = UnscentedKalmanFilter(
                state_transition_fn=state_transition_fn,
                observation_fn=observation_fn,
                Q=model.Q,
                R=model.R,
                x0=x0,
                Sigma0=Sigma0,
                **params
            )

            filtered_states, predicted_states = ukf.filter(observations)

            assert not tf.reduce_any(tf.math.is_nan(filtered_states))
            assert not tf.reduce_any(tf.math.is_inf(filtered_states))
            assert filtered_states.shape == (4, T + 1)


class TestUKFIntegration:
    """Integration tests for full UKF workflow."""

    def test_filter_with_range_bearing_data(self, range_bearing_system):
        """Test filter with range-bearing observations using true initial state."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        T = 30
        true_states, observations = model.simulate_trajectory(T=T)

        # Use true initial state
        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=true_states[:, 0],
            Sigma0=Sigma0
        )

        filtered_states, predicted_states = ukf.filter(observations)

        assert filtered_states.shape == (4, T + 1)
        assert predicted_states.shape == (4, T)
        assert true_states.shape == (4, T + 1)
        assert observations.shape == (2, T)


    def test_filter_reduces_error(self, range_bearing_system):
        """Test that filtered estimates are better than predicted using true initial state."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        T = 50
        true_states, observations = model.simulate_trajectory(T=T)

        # Use true initial state
        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=true_states[:, 0],
            Sigma0=Sigma0
        )

        filtered_states, predicted_states = ukf.filter(observations)

        predicted_error = tf.reduce_mean(
            tf.sqrt((predicted_states[0, :] - true_states[0, 1:])**2 +
                   (predicted_states[2, :] - true_states[2, 1:])**2)
        )
        filtered_error = tf.reduce_mean(
            tf.sqrt((filtered_states[0, 1:] - true_states[0, 1:])**2 +
                   (filtered_states[2, 1:] - true_states[2, 1:])**2)
        )

        assert filtered_error < predicted_error

    def test_filter_reproducibility(self, range_bearing_system, assert_equal):
        """Test that filter produces reproducible results with true initial state."""
        model, state_transition_fn, observation_fn, x0, Sigma0 = range_bearing_system

        tf.random.set_seed(123)
        true_states, observations = model.simulate_trajectory(T=10)

        # Use true initial state
        ukf = UnscentedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=true_states[:, 0],
            Sigma0=Sigma0
        )

        filtered_states1, predicted_states1 = ukf.filter(observations)
        filtered_states2, predicted_states2 = ukf.filter(observations)

        assert_equal(filtered_states1, filtered_states2)
        assert_equal(predicted_states1, predicted_states2)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
