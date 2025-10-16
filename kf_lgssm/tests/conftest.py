"""
Shared pytest fixtures for Kalman Filter tests.
"""

import pytest
import tensorflow as tf


@pytest.fixture
def simple_2d_system():
    """
    2D constant velocity system for testing.
    State: [position, velocity]
    """
    F = tf.constant([[1.0, 1.0],
                     [0.0, 1.0]], dtype=tf.float32)
    H = tf.constant([[1.0, 0.0],
                     [0.0, 1.0]], dtype=tf.float32)
    Q = tf.constant([[0.1, 0.0],
                     [0.0, 0.1]], dtype=tf.float32)
    R = tf.constant([[0.5, 0.0],
                     [0.0, 0.5]], dtype=tf.float32)
    x0 = tf.constant([0.0, 1.0], dtype=tf.float32)
    Sigma0 = tf.constant([[1.0, 0.0],
                          [0.0, 1.0]], dtype=tf.float32)

    return F, H, Q, R, x0, Sigma0


@pytest.fixture
def simple_1d_system():
    """1D system for testing edge cases."""
    F = tf.constant([[1.0]], dtype=tf.float32)
    H = tf.constant([[1.0]], dtype=tf.float32)
    Q = tf.constant([[0.1]], dtype=tf.float32)
    R = tf.constant([[0.5]], dtype=tf.float32)
    x0 = tf.constant([0.0], dtype=tf.float32)
    Sigma0 = tf.constant([[1.0]], dtype=tf.float32)

    return F, H, Q, R, x0, Sigma0


@pytest.fixture
def system_with_control():
    """2D system with control input."""
    F = tf.constant([[1.0, 1.0],
                     [0.0, 1.0]], dtype=tf.float32)
    H = tf.constant([[1.0, 0.0],
                     [0.0, 1.0]], dtype=tf.float32)
    Q = tf.constant([[0.1, 0.0],
                     [0.0, 0.1]], dtype=tf.float32)
    R = tf.constant([[0.5, 0.0],
                     [0.0, 0.5]], dtype=tf.float32)
    x0 = tf.constant([0.0, 1.0], dtype=tf.float32)
    Sigma0 = tf.constant([[1.0, 0.0],
                          [0.0, 1.0]], dtype=tf.float32)
    B = tf.constant([[0.5], [1.0]], dtype=tf.float32)

    return F, H, Q, R, x0, Sigma0, B


@pytest.fixture
def very_high_dimensional_system():
    """
    10D system for stress testing multidimensional capabilities.
    State dimension: 10
    Observation dimension: 5
    """
    state_dim = 10
    obs_dim = 5

    # Random but stable system matrix
    tf.random.set_seed(42)
    F = tf.eye(state_dim, dtype=tf.float32) + tf.random.normal([state_dim, state_dim], 0.0, 0.05, dtype=tf.float32)

    # Observe first 5 dimensions
    H = tf.concat([tf.eye(obs_dim, dtype=tf.float32), tf.zeros([obs_dim, state_dim - obs_dim], dtype=tf.float32)], axis=1)

    # Process noise
    Q = tf.eye(state_dim, dtype=tf.float32) * 0.1

    # Observation noise
    R = tf.eye(obs_dim, dtype=tf.float32) * 0.5

    # Initial state
    x0 = tf.zeros([state_dim], dtype=tf.float32)

    # Initial covariance
    Sigma0 = tf.eye(state_dim, dtype=tf.float32)

    return F, H, Q, R, x0, Sigma0
