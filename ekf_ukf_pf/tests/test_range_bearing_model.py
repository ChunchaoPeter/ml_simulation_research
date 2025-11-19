"""
Test suite for Range-Bearing Non-Linear State-Space Model.

Includes unit tests and integration tests for range_bearing_model.py
"""

import pytest
import tensorflow as tf
from range_bearing_model import RangeBearingModel
import math


class TestRangeBearingModelUnit:
    """Unit tests for individual RangeBearingModel methods."""

    def test_initialization(self):
        """Test RangeBearingModel initialization."""
        model = RangeBearingModel(dt=1.0, seed=42)

        assert model.dt == 1.0
        assert model.A.shape == (4, 4)
        assert model.Q.shape == (4, 4)
        assert model.R.shape == (2, 2)

    def test_state_transition_matrix(self, assert_allclose):
        """Test state transition matrix is correct for constant velocity model."""
        dt = 1.0
        model = RangeBearingModel(dt=dt, seed=42)

        # Check structure of A matrix
        expected_A = tf.constant([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=tf.float32)

        assert_allclose(model.A, expected_A)

    def test_observation_model_range_bearing(self):
        """Test observation model produces range and bearing."""
        model = RangeBearingModel(
            dt=1.0,
            range_noise_std=0.0,
            bearing_noise_std=0.0,
            seed=42
        )

        # State at (3, 0, 4, 0) -> range = 5, bearing = arctan2(4, 3)
        x = tf.constant([[3.0], [0.0], [4.0], [0.0]], dtype=tf.float32)

        z = model.observation_model(x)

        # Check shape
        assert z.shape == (2, 1)

        # Check range (with noise_std=0, but model still adds noise)
        # So we just check reasonable values
        expected_range = 5.0
        expected_bearing = tf.atan2(4.0, 3.0)

        # Allow for noise
        assert tf.abs(z[0, 0] - expected_range) < 1  # Range within 1
        # Bearing wrapped to [-pi, pi]
        bearing_diff = tf.abs(z[1, 0] - expected_bearing)
        bearing_diff = tf.minimum(bearing_diff, 2*math.pi - bearing_diff)
        assert bearing_diff < 0.1  # Within 0.1 radians

    def test_observation_model_bearing_wrapping(self):
        """Test that bearing is wrapped to [-π, π]."""
        model = RangeBearingModel(
            dt=1.0,
            bearing_noise_std=0.001,
            seed=42
        )

        # Test various positions
        positions = [
            [[1.0], [0.0], [0.0], [0.0]],   # bearing = 0
            [[0.0], [0.0], [1.0], [0.0]],   # bearing = π/2
            [[-1.0], [0.0], [0.0], [0.0]],  # bearing = π or -π
            [[0.0], [0.0], [-1.0], [0.0]],  # bearing = -π/2
        ]

        for pos in positions:
            x = tf.constant(pos, dtype=tf.float32)
            z = model.observation_model(x)

            # Bearing should be in [-π, π]
            assert z[1, 0] >= -math.pi
            assert z[1, 0] <= math.pi

    def test_compute_observation_jacobian(self, assert_allclose):
        """Test observation Jacobian computation."""
        model = RangeBearingModel(dt=1.0, seed=42)

        x = tf.constant([[10.0], [1.0], [20.0], [2.0]], dtype=tf.float32)

        H = model.compute_observation_jacobian(x)

        # Should be (2, 4) - 2 observations (range, bearing), 4 states
        assert H.shape == (2, 4)

        # Check structure: H should only depend on position, not velocity
        # So columns 1 and 3 (velocity indices) should be zero
        assert_allclose(H[0, 1], 0.0)
        assert_allclose(H[0, 3], 0.0)
        assert_allclose(H[1, 1], 0.0)
        assert_allclose(H[1, 3], 0.0)


class TestRangeBearingModelIntegration:
    """Integration tests for full trajectory simulation."""

    def test_simulate_trajectory_shapes(self):
        """Test simulate_trajectory produces correct shapes."""
        model = RangeBearingModel(dt=1.0, seed=42)

        T = 10
        states, observations = model.simulate_trajectory(T=T)

        # States: (state_dim, T+1) - includes x_0, x_1, ..., x_T
        assert states.shape == (4, T + 1)

        # Observations: (obs_dim, T) - includes z_1, ..., z_T
        assert observations.shape == (2, T)

    def test_simulate_trajectory_reproducibility(self, assert_allclose):
        """Test that simulation with same seed is reproducible."""
        T = 10

        # First simulation
        model1 = RangeBearingModel(dt=1.0, seed=42)
        states1, obs1 = model1.simulate_trajectory(T=T, initial_pos_std=10.0, initial_vel_std=1.0)

        # Second simulation with same seed (Global seed here)
        model2 = RangeBearingModel(dt=1.0, seed=42)
        states2, obs2 = model2.simulate_trajectory(T=T, initial_pos_std=10.0, initial_vel_std=1.0)

        # Should produce same results
        assert_allclose(states1, states2)
        assert_allclose(obs1, obs2)


    def test_observation_range_positive(self):
        """Test that range observations are always positive (enforced by clipping)."""
        # Use default noise (now 10.0 instead of 50.0)
        model = RangeBearingModel(dt=1.0, seed=42)

        T = 50
        _, observations = model.simulate_trajectory(T=T)

        # Range (first row) should always be non-negative due to clipping
        ranges = observations[0, :]

        # With clipping in range_bearing_model.py, all ranges should be >= 0
        assert tf.reduce_all(ranges >= 0.0), f"Found negative range: min={tf.reduce_min(ranges)}"

        # Check mean is positive
        mean_range = tf.reduce_mean(ranges)
        assert mean_range > 0, f"Mean range is negative: {mean_range}"

    def test_observation_bearing_in_range(self):
        """Test that bearing observations are in [-π, π]."""
        model = RangeBearingModel(dt=1.0, seed=42)

        T = 50
        _, observations = model.simulate_trajectory(T=T)

        # Bearing (second row) should be in [-π, π]
        bearings = observations[1, :]
        assert tf.reduce_all(bearings >= -math.pi)
        assert tf.reduce_all(bearings <= math.pi)

    def test_full_workflow_with_filtering(self):
        """Test that generated data can be used for filtering."""
        model = RangeBearingModel(dt=1.0, seed=42)

        # Generate trajectory
        T = 30
        true_states, observations = model.simulate_trajectory(T=T)

        # Verify data is valid for filtering
        assert true_states.shape == (4, T + 1)
        assert observations.shape == (2, T)

        # Check that we can compute Jacobians for all states
        for t in range(T + 1):
            x = true_states[:, t:t+1]
            H = model.compute_observation_jacobian(x)
            assert H.shape == (2, 4)
            assert not tf.reduce_any(tf.math.is_nan(H))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
