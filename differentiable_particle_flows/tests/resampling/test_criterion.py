"""Tests for dpf/resampling/criterion.py: NeffCriterion.

Tests the ESS-based resampling criterion:

    ESS_t = 1 / sum_i (w_t^i)^2

Resampling is triggered when ESS_t < threshold_ratio * N.

    - Uniform weights:    ESS = N (all particles equally useful) -> no resample.
    - Degenerate weights: ESS ~ 1 (one particle dominates)       -> resample.

Test class:
    TestNeffCriterion
        - Uniform weights (ESS=N) with threshold=0.5 -> no resampling.
        - Degenerate weights (ESS~1) with threshold=0.5 -> resampling triggered.
        - Per-batch independence: mixed batch (one uniform, one degenerate)
          produces different flags per batch element.
        - Custom threshold: threshold_ratio > 1.0 forces resampling even
          for uniform weights.
"""

import tensorflow as tf
import numpy as np
from dpf.base import State
from dpf.resampling.criterion import NeffCriterion

BATCH = 2
N = 10
D = 2


def _make_state_uniform():
    """State with uniform weights → ESS = N."""
    particles = tf.constant(np.random.randn(BATCH, N, D), dtype=tf.float64)
    log_weights = tf.fill([BATCH, N], -tf.math.log(tf.cast(N, tf.float64)))
    return State(particles=particles, log_weights=log_weights)


def _make_state_degenerate():
    """State with all weight on particle 0 → ESS ≈ 1."""
    particles = tf.constant(np.random.randn(BATCH, N, D), dtype=tf.float64)
    lw = np.full((BATCH, N), -1e10)
    lw[:, 0] = 0.0
    log_weights = tf.constant(lw, dtype=tf.float64)
    return State(particles=particles, log_weights=log_weights)


class TestNeffCriterion:
    def test_high_ess_no_resample(self):
        """Uniform weights → ESS=N > 0.5*N → no resampling."""
        criterion = NeffCriterion(threshold_ratio=0.5)
        state = _make_state_uniform()
        flags = criterion.apply(state)
        assert flags.shape == (BATCH,)
        assert not np.any(flags.numpy())

    def test_low_ess_resample(self):
        """Degenerate weights → ESS≈1 < 0.5*N → resample."""
        criterion = NeffCriterion(threshold_ratio=0.5)
        state = _make_state_degenerate()
        flags = criterion.apply(state)
        assert flags.shape == (BATCH,)
        assert np.all(flags.numpy())

    def test_per_batch(self):
        """Mixed batch: one uniform, one degenerate → different flags."""
        particles = tf.constant(np.random.randn(2, N, D), dtype=tf.float64)
        # Batch 0: uniform → ESS = N
        lw0 = np.full(N, -np.log(N))
        # Batch 1: degenerate → ESS ≈ 1
        lw1 = np.full(N, -1e10)
        lw1[0] = 0.0
        log_weights = tf.constant(np.stack([lw0, lw1]), dtype=tf.float64)
        state = State(particles=particles, log_weights=log_weights)

        criterion = NeffCriterion(threshold_ratio=0.5)
        flags = criterion.apply(state)
        assert not flags.numpy()[0]  # uniform → no resample
        assert flags.numpy()[1]      # degenerate → resample

    def test_custom_threshold(self):
        """Use threshold > 1.0 to force resampling."""
        state = _make_state_uniform()
        criterion_force = NeffCriterion(threshold_ratio=1.1)
        flags_force = criterion_force.apply(state)
        assert np.all(flags_force.numpy())
