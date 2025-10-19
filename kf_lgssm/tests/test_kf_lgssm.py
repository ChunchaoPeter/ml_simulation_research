"""
Test suite for Kalman Filter implementation.

Includes unit tests and integration tests for kf_lgssm.py
"""

import pytest
import tensorflow as tf
from kf_lgssm import KalmanFilter
from lgss_sample import sample


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-5):
    """Assert that two tensors are element-wise equal within tolerance."""
    diff = tf.abs(actual - expected)
    tolerance = atol + rtol * tf.abs(expected)
    assert tf.reduce_all(diff <= tolerance), (
        f"Arrays not close enough.\n"
        f"Max difference: {tf.reduce_max(diff)}\n"
        f"Max tolerance: {tf.reduce_max(tolerance)}"
    )


def assert_equal(actual, expected):
    """Assert that two tensors are exactly equal."""
    assert tf.reduce_all(tf.equal(actual, expected)), (
        f"Arrays not equal.\n"
        f"Actual: {actual}\n"
        f"Expected: {expected}"
    )


class TestKalmanFilterUnit:
    """Unit tests for individual Kalman Filter methods."""

    def test_initialization(self, simple_2d_system):
        """Test KalmanFilter initialization with correct dimensions."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        assert kf.state_dim == 2
        assert kf.obs_dim == 2
        assert kf.x0.shape == (2, 1)

    def test_initialization_with_1d_vector(self, simple_2d_system):
        """Test that 1D initial state vector is converted to column vector."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        assert len(kf.x0.shape) == 2
        assert kf.x0.shape == (2, 1)

    def test_predict_without_control(self, simple_2d_system):
        """Test prediction step matches mathematical formula."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        Sigma = tf.eye(2, dtype=tf.float32)

        x_pred, Sigma_pred = kf.predict(x, Sigma)

        # Check x_pred = F @ x
        expected_x_pred = tf.linalg.matmul(F, x)
        assert_allclose(x_pred, expected_x_pred)

        # Check Sigma_pred = F @ Sigma @ F^T + Q
        expected_Sigma = tf.linalg.matmul(tf.linalg.matmul(F, Sigma), F, transpose_b=True) + Q
        assert_allclose(Sigma_pred, expected_Sigma)

    def test_predict_with_control(self, system_with_control):
        """Test prediction step with control input."""
        F, H, Q, R, x0, Sigma0, B = system_with_control
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, B=B)

        x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        Sigma = tf.eye(2, dtype=tf.float32)
        u = tf.constant([[5.0]], dtype=tf.float32)

        x_pred, Sigma_pred = kf.predict(x, Sigma, u)

        # Check x_pred = F @ x + B @ u
        expected_x_pred = tf.matmul(F, x) + tf.matmul(B, u)
        assert_allclose(x_pred, expected_x_pred)

    def test_update_reduces_uncertainty(self, simple_2d_system):
        """Test that update step reduces covariance (uncertainty)."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        z = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        x_pred = tf.constant([[0.5], [1.5]], dtype=tf.float32)
        Sigma_pred = tf.constant([[2.0, 0.0], [0.0, 2.0]], dtype=tf.float32)

        x_post, Sigma_post = kf.update(z, x_pred, Sigma_pred)

        # Posterior covariance should have smaller trace than predicted
        trace_pred = tf.linalg.trace(Sigma_pred)
        trace_post = tf.linalg.trace(Sigma_post)

        assert trace_post < trace_pred

    def test_update_joseph_form(self, simple_2d_system):
        """Test Joseph form produces symmetric covariance."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, use_joseph_form=True)

        z = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        x_pred = tf.constant([[0.5], [1.5]], dtype=tf.float32)
        Sigma_pred = tf.eye(2, dtype=tf.float32)

        x_post, Sigma_post = kf.update(z, x_pred, Sigma_pred)

        # Check symmetry
        assert_allclose(Sigma_post, tf.transpose(Sigma_post))

    def test_update_standard_form(self, simple_2d_system):
        """Test standard form produces symmetric covariance."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, use_joseph_form=False)

        z = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        x_pred = tf.constant([[0.5], [1.5]], dtype=tf.float32)
        Sigma_pred = tf.eye(2, dtype=tf.float32)

        x_post, Sigma_post = kf.update(z, x_pred, Sigma_pred)

        # Check symmetry
        assert_allclose(Sigma_post, tf.transpose(Sigma_post))

    def test_covariance_positive_definite(self, simple_2d_system):
        """Test that covariances remain positive definite."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        Sigma = tf.eye(2, dtype=tf.float32)

        # Predict
        x_pred, Sigma_pred = kf.predict(x, Sigma)
        eigenvalues_pred = tf.linalg.eigvalsh(Sigma_pred)
        assert tf.reduce_all(eigenvalues_pred > 0)

        # Update
        z = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        x_post, Sigma_post = kf.update(z, x_pred, Sigma_pred)
        eigenvalues_post = tf.linalg.eigvalsh(Sigma_post)
        assert tf.reduce_all(eigenvalues_post > 0)


class TestKalmanFilterIntegration:
    """Integration tests for full Kalman filter workflow (mainly testing filter() method)."""

    def test_filter_output_shapes(self, simple_2d_system):
        """Test filter produces correct output shapes."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 10
        observations = tf.random.normal([2, T], dtype=tf.float32)

        filtered_states, predicted_states = kf.filter(observations)

        assert filtered_states.shape == (2, T + 1)
        assert predicted_states.shape == (2, T)

    def test_filter_with_sampled_data(self, simple_2d_system):
        """Test filter with data from sample function."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 30
        true_states, observations = sample(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, T=T, seed=42)

        filtered_states, predicted_states = kf.filter(observations)

        # Check shapes
        assert filtered_states.shape == (2, T + 1)
        assert predicted_states.shape == (2, T)
        assert true_states.shape == (2, T + 1)
        assert observations.shape == (2, T)

    def test_filter_reduces_error(self, simple_2d_system):
        """Test that filtered estimates are better than predicted."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 50
        true_states, observations = sample(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, T=T, seed=42)

        filtered_states, predicted_states = kf.filter(observations)

        # Compare errors (skip initial state)
        predicted_error = tf.reduce_mean(tf.norm(predicted_states - true_states[:, 1:], axis=0))
        filtered_error = tf.reduce_mean(tf.norm(filtered_states[:, 1:] - true_states[:, 1:], axis=0))

        assert filtered_error < predicted_error

    def test_filter_with_control_input(self, system_with_control):
        """Test filter with control inputs."""
        F, H, Q, R, x0, Sigma0, B = system_with_control
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, B=B)

        T = 10
        observations = tf.random.normal([2, T], dtype=tf.float32)
        controls = tf.random.normal([1, T], dtype=tf.float32)

        filtered_states, predicted_states = kf.filter(observations, controls)

        assert filtered_states.shape == (2, T + 1)
        assert predicted_states.shape == (2, T)

    def test_filter_initial_state(self, simple_2d_system):
        """Test that first filtered state matches initial state."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 10
        observations = tf.random.normal([2, T], dtype=tf.float32)

        filtered_states, predicted_states = kf.filter(observations)

        assert_allclose(filtered_states[:, 0:1], kf.x0)

    def test_filter_reproducibility(self, simple_2d_system):
        """Test that filter produces reproducible results."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 10
        observations = tf.random.normal([2, T], dtype=tf.float32, seed=123)

        filtered_states1, predicted_states1 = kf.filter(observations)
        filtered_states2, predicted_states2 = kf.filter(observations)

        assert_equal(filtered_states1, filtered_states2)
        assert_equal(predicted_states1, predicted_states2)

    def test_forms_comparison(self, simple_2d_system):
        """Test that Joseph and standard forms produce similar results."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system

        kf_joseph = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, use_joseph_form=True)
        kf_standard = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, use_joseph_form=False)

        T = 20
        true_states, observations = sample(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, T=T, seed=42)

        filtered_joseph, _ = kf_joseph.filter(observations)
        filtered_standard, _ = kf_standard.filter(observations)

        # States should be very similar
        assert_allclose(filtered_joseph, filtered_standard, rtol=1e-3, atol=1e-3)

    def test_full_workflow(self, simple_2d_system):
        """Test complete workflow: sample -> filter -> evaluate."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system

        # Sample data
        T = 30
        true_states, observations = sample(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, T=T, seed=42)

        # Filter
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)
        filtered_states, predicted_states = kf.filter(observations)

        # Evaluate
        pred_error = tf.reduce_mean(tf.norm(predicted_states - true_states[:, 1:], axis=0))
        filt_error = tf.reduce_mean(tf.norm(filtered_states[:, 1:] - true_states[:, 1:], axis=0))

        # Check all shapes and error reduction
        assert true_states.shape == (2, T + 1)
        assert observations.shape == (2, T)
        assert filtered_states.shape == (2, T + 1)
        assert predicted_states.shape == (2, T)
        assert filt_error < pred_error

    def test_1d_system(self, simple_1d_system):
        """Test with 1D system."""
        F, H, Q, R, x0, Sigma0 = simple_1d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 10
        observations = tf.random.normal([1, T], dtype=tf.float32)

        filtered_states, predicted_states = kf.filter(observations)

        assert filtered_states.shape == (1, T + 1)
        assert predicted_states.shape == (1, T)

    def test_no_nan_or_inf(self, simple_2d_system):
        """Test that filter doesn't produce NaN or Inf values."""
        F, H, Q, R, x0, Sigma0 = simple_2d_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 100
        observations = tf.random.normal([2, T], dtype=tf.float32)

        filtered_states, predicted_states = kf.filter(observations)

        assert not tf.reduce_any(tf.math.is_nan(filtered_states))
        assert not tf.reduce_any(tf.math.is_inf(filtered_states))
        assert not tf.reduce_any(tf.math.is_nan(predicted_states))
        assert not tf.reduce_any(tf.math.is_inf(predicted_states))

    def test_10d_very_high_dimensional_system(self, very_high_dimensional_system):
        """Test filter with 10D state space (stress test for high dimensions)."""
        F, H, Q, R, x0, Sigma0 = very_high_dimensional_system
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

        T = 50
        # Sample data
        true_states, observations = sample(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, T=T, seed=42)

        # Filter
        filtered_states, predicted_states = kf.filter(observations)

        # Check shapes: 10 states, 5 observations
        assert filtered_states.shape == (10, T + 1)
        assert predicted_states.shape == (10, T)
        assert true_states.shape == (10, T + 1)
        assert observations.shape == (5, T)

        # Check error reduction
        predicted_error = tf.reduce_mean(tf.norm(predicted_states - true_states[:, 1:], axis=0))
        filtered_error = tf.reduce_mean(tf.norm(filtered_states[:, 1:] - true_states[:, 1:], axis=0))
        assert filtered_error < predicted_error

        # Check no NaN or Inf in high dimensional case
        assert not tf.reduce_any(tf.math.is_nan(filtered_states))
        assert not tf.reduce_any(tf.math.is_inf(filtered_states))

    def test_multidimensional_with_varying_dimensions(self):
        """Test systems with different state/observation dimension combinations."""
        test_cases = [
            (3, 2),  # 3 states, 2 observations
            (5, 3),  # 5 states, 3 observations
            (6, 4),  # 6 states, 4 observations
        ]

        for state_dim, obs_dim in test_cases:
            # Create system
            F = tf.eye(state_dim, dtype=tf.float32)
            H = tf.concat([tf.eye(obs_dim, dtype=tf.float32),
                          tf.zeros([obs_dim, state_dim - obs_dim], dtype=tf.float32)], axis=1)
            Q = tf.eye(state_dim, dtype=tf.float32) * 0.1
            R = tf.eye(obs_dim, dtype=tf.float32) * 0.5
            x0 = tf.zeros([state_dim], dtype=tf.float32)
            Sigma0 = tf.eye(state_dim, dtype=tf.float32)

            kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0)

            T = 10
            observations = tf.random.normal([obs_dim, T], dtype=tf.float32)

            filtered_states, predicted_states = kf.filter(observations)

            assert filtered_states.shape == (state_dim, T + 1)
            assert predicted_states.shape == (state_dim, T)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
