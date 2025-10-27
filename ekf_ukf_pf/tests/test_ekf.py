"""
Test suite for Extended Kalman Filter implementation.

Includes unit tests and integration tests for ekf.py
"""

import pytest
import tensorflow as tf
from ekf import ExtendedKalmanFilter


class TestEKFUnit:
    """Unit tests for individual EKF methods."""

    def test_initialization(self, range_bearing_system_with_jacobians):
        """Test EKF initialization with correct dimensions."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        assert ekf.state_dim == 4
        assert ekf.obs_dim == 2
        assert ekf.x0.shape == (4, 1)

    def test_initialization_with_1d_vector(self, range_bearing_system_with_jacobians):
        """Test that 1D initial state vector is converted to column vector."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        assert len(ekf.x0.shape) == 2
        assert ekf.x0.shape == (4, 1)

    def test_state_jacobian_computation(self, range_bearing_system_with_jacobians, assert_allclose):
        """Test state transition Jacobian computation."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        x = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        F = ekf.compute_state_jacobian(x)

        assert F.shape == (4, 4)
        assert_allclose(F, model.A)

    def test_observation_jacobian_computation(self, range_bearing_system_with_jacobians, assert_allclose):
        """Test observation Jacobian computation."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        x = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        H = ekf.compute_observation_jacobian(x)

        assert H.shape == (2, 4)
        H_expected = model.compute_observation_jacobian(x)
        assert_allclose(H, H_expected)

    def test_predict_increases_uncertainty(self, range_bearing_system_with_jacobians):
        """Test that prediction step increases uncertainty."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        x = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        Sigma = tf.eye(4, dtype=tf.float32)

        _, Sigma_pred = ekf.predict(x, Sigma)

        assert tf.linalg.trace(Sigma_pred) > tf.linalg.trace(Sigma)

    def test_update_reduces_uncertainty(self, range_bearing_system_with_jacobians):
        """Test that update step reduces uncertainty."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        x_pred = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        Sigma_pred = tf.eye(4, dtype=tf.float32) * 2.0
        z = observation_fn(x_pred)

        _, Sigma_post = ekf.update(z, x_pred, Sigma_pred)

        assert tf.linalg.trace(Sigma_post) < tf.linalg.trace(Sigma_pred)

    def test_joseph_form_vs_standard_form(self, range_bearing_system_with_jacobians, assert_allclose):
        """Test that Joseph and standard forms produce similar covariances."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf_joseph = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn,
            use_joseph_form=True
        )

        ekf_standard = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn,
            use_joseph_form=False
        )

        x_pred_j, Sigma_pred_j = ekf_joseph.predict(ekf_joseph.x0, ekf_joseph.Sigma0)
        x_pred_s, Sigma_pred_s = ekf_standard.predict(ekf_standard.x0, ekf_standard.Sigma0)

        z = observation_fn(x_pred_j)

        x_post_j, Sigma_post_j = ekf_joseph.update(z, x_pred_j, Sigma_pred_j)
        x_post_s, Sigma_post_s = ekf_standard.update(z, x_pred_s, Sigma_pred_s)

        assert_allclose(x_post_j, x_post_s)
        assert_allclose(Sigma_post_j, Sigma_post_s, rtol=1e-3, atol=1e-3)

    def test_covariance_positive_definite(self, range_bearing_system_with_jacobians):
        """Test that covariances remain positive definite."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        x = tf.constant([[10.0], [1.0], [20.0], [1.0]], dtype=tf.float32)
        Sigma = tf.eye(4, dtype=tf.float32)

        x_pred, Sigma_pred = ekf.predict(x, Sigma)
        assert tf.reduce_all(tf.linalg.eigvalsh(Sigma_pred) > 0)

        z = observation_fn(x_pred)
        _, Sigma_post = ekf.update(z, x_pred, Sigma_pred)
        assert tf.reduce_all(tf.linalg.eigvalsh(Sigma_post) > 0)


class TestEKFIntegration:
    """Integration tests for full EKF workflow."""

    def test_filter_with_range_bearing_data(self, range_bearing_system_with_jacobians):
        """Test filter produces correct output shapes and with range-bearing observations."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        T = 30
        true_states, observations = model.simulate_trajectory(T=T)
        filtered_states, predicted_states = ekf.filter(observations)

        assert filtered_states.shape == (4, T + 1)
        assert predicted_states.shape == (4, T)
        assert true_states.shape == (4, T + 1)
        assert observations.shape == (2, T)


    def test_filter_reproducibility(self, range_bearing_system_with_jacobians, assert_equal):
        """Test that filter produces reproducible results."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        tf.random.set_seed(123)
        _, observations = model.simulate_trajectory(T=10)

        filtered_states1, predicted_states1 = ekf.filter(observations)
        filtered_states2, predicted_states2 = ekf.filter(observations)

        assert_equal(filtered_states1, filtered_states2)
        assert_equal(predicted_states1, predicted_states2)

    def test_no_nan_or_inf(self, range_bearing_system_with_jacobians):
        """Test that filter doesn't produce NaN or Inf values."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=x0,
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )

        T = 100
        _, observations = model.simulate_trajectory(T=T)
        filtered_states, predicted_states = ekf.filter(observations)

        assert not tf.reduce_any(tf.math.is_nan(filtered_states))
        assert not tf.reduce_any(tf.math.is_inf(filtered_states))
        assert not tf.reduce_any(tf.math.is_nan(predicted_states))
        assert not tf.reduce_any(tf.math.is_inf(predicted_states))

    def test_full_workflow(self, range_bearing_system_with_jacobians):
        """Test complete workflow: simulate -> filter -> evaluate."""
        (model, state_transition_fn, observation_fn,
         state_jacobian_fn, obs_jacobian_fn, x0, Sigma0) = range_bearing_system_with_jacobians

        T = 30
        true_states, observations = model.simulate_trajectory(T=T)

        ekf = ExtendedKalmanFilter(
            state_transition_fn=state_transition_fn,
            observation_fn=observation_fn,
            Q=model.Q,
            R=model.R,
            x0=true_states[:, 0],
            Sigma0=Sigma0,
            state_transition_jacobian_fn=state_jacobian_fn,
            observation_jacobian_fn=obs_jacobian_fn
        )
        filtered_states, predicted_states = ekf.filter(observations)

        pred_error = tf.reduce_mean(
            tf.sqrt((predicted_states[0, :] - true_states[0, 1:])**2 +
                   (predicted_states[2, :] - true_states[2, 1:])**2)
        )
        filt_error = tf.reduce_mean(
            tf.sqrt((filtered_states[0, 1:] - true_states[0, 1:])**2 +
                   (filtered_states[2, 1:] - true_states[2, 1:])**2)
        )

        assert true_states.shape == (4, T + 1)
        assert observations.shape == (2, T)
        assert filtered_states.shape == (4, T + 1)
        assert predicted_states.shape == (4, T)
        assert filt_error < 100.0
        assert pred_error < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
