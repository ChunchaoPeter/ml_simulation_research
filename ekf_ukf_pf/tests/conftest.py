"""
Shared pytest fixtures and helper functions for EKF, UKF, and Range-Bearing Model tests.
"""

import pytest
import tensorflow as tf
from range_bearing_model import RangeBearingModel

# Enable eager execution for coverage testing
# This disables @tf.function graph compilation, allowing coverage.py to trace execution
tf.config.run_functions_eagerly(True)

@pytest.fixture
def assert_allclose():
    """Assert that two tensors are element-wise equal within tolerance."""
    def _fn(actual, expected, rtol=1e-5, atol=1e-5):
        diff = tf.abs(actual - expected)
        tolerance = atol + rtol * tf.abs(expected)
        assert tf.reduce_all(diff <= tolerance), (
            f"Arrays not close enough.\n"
            f"Max difference: {tf.reduce_max(diff)}\n"
            f"Max tolerance: {tf.reduce_max(tolerance)}"
        )
    return _fn

@pytest.fixture
def assert_equal():
    """Assert that two tensors are exactly equal."""
    def _fn(actual, expected):
        assert tf.reduce_all(tf.equal(actual, expected)), (
            f"Arrays not equal.\n"
            f"Actual: {actual}\n"
            f"Expected: {expected}"
        )
    return _fn


def _create_range_bearing_base():
    """
    Helper function to create common range-bearing system components.
    Returns: model, state_transition_fn, observation_fn, x0, Sigma0
    """
    # Create model
    model = RangeBearingModel(
        dt=1.0,
        process_noise_std_pos=0.1,
        process_noise_std_vel=0.1,
        range_noise_std=5.0,
        bearing_noise_std=0.01,
        seed=42
    )

    # Define state transition function
    def state_transition_fn(x, u=None):
        return tf.linalg.matmul(model.A, x)

    # Define observation function
    def observation_fn(x):
        x_pos, y_pos = x[0, 0], x[2, 0]
        r = tf.sqrt(x_pos**2 + y_pos**2)
        theta = tf.atan2(y_pos, x_pos)
        return tf.stack([r, theta])[:, tf.newaxis]

    # Initial state
    x0 = tf.constant([10.0, 1.0, 20.0, 1.0], dtype=tf.float32)

    # Initial covariance
    Sigma0 = tf.eye(4, dtype=tf.float32) * 10.0

    return model, state_transition_fn, observation_fn, x0, Sigma0


@pytest.fixture
def range_bearing_system():
    """
    Range-bearing tracking system for UKF testing.
    State: [x, x_dot, y, y_dot] (2D position and velocity)
    Observation: [range, bearing] (polar coordinates from origin)
    """
    return _create_range_bearing_base()


@pytest.fixture
def range_bearing_system_with_jacobians():
    """
    Range-bearing tracking system with Jacobian functions for EKF testing.
    """
    model, state_transition_fn, observation_fn, x0, Sigma0 = _create_range_bearing_base()

    # Define state Jacobian function (returns A matrix)
    def state_jacobian_fn(x, u=None):
        return model.A

    # Define observation Jacobian function
    def observation_jacobian_fn(x):
        return model.compute_observation_jacobian(x)

    return model, state_transition_fn, observation_fn, state_jacobian_fn, observation_jacobian_fn, x0, Sigma0
