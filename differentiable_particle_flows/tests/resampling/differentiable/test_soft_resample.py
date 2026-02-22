"""Tests for SoftResampler: _get_cdf_weights, _compute_new_log_weights,
and gradient flow through weights.
"""

import tensorflow as tf
import numpy as np
from dpf.base import State
from dpf.resampling.differentiable.soft_resample import SoftResampler

BATCH = 2
N = 5
ALPHA = 0.7
W = np.array([[0.4, 0.3, 0.15, 0.1, 0.05],
              [0.2, 0.2, 0.2, 0.2, 0.2]])


def _make_state(weights=W):
    weights = np.asarray(weights, dtype=np.float64)
    log_weights = tf.constant(np.log(weights), dtype=tf.float64)
    batch, n = weights.shape
    particles = tf.constant(
        np.arange(n, dtype=np.float64)[np.newaxis, :, np.newaxis]
        * np.ones((batch, 1, 1)),
        dtype=tf.float64,
    )
    return State(particles=particles, log_weights=log_weights)


class TestGetCdfWeights:

    def test_matches_formula_and_sums_to_one(self):
        """CDF weights = normalized(W / w_soft) where w_soft = alpha*W + (1-alpha)/N.
        Must sum to 1 per batch."""
        w_soft = ALPHA * W + (1.0 - ALPHA) / N
        corrected = W / w_soft
        expected = corrected / corrected.sum(axis=-1, keepdims=True)

        resampler = SoftResampler(alpha=ALPHA)
        cdf_weights = resampler._get_cdf_weights(_make_state()).numpy()

        np.testing.assert_allclose(cdf_weights, expected, atol=1e-10)
        np.testing.assert_allclose(
            cdf_weights.sum(axis=-1), [1.0, 1.0], atol=1e-12
        )

    def test_uniform_weights_stay_uniform(self):
        """Equal weights: w_soft = 1/N, so W/w_soft = 1 => uniform output."""
        uniform = np.ones((BATCH, N)) / N
        resampler = SoftResampler(alpha=0.6)
        cdf_weights = resampler._get_cdf_weights(_make_state(uniform)).numpy()
        np.testing.assert_allclose(cdf_weights, np.ones((BATCH, N)) / N, atol=1e-12)


class TestComputeNewLogWeights:

    def test_gathers_corrected_weights_at_indices(self):
        """New log weights = corrected_log_weights[ancestor_indices] per batch."""
        resampler = SoftResampler(alpha=ALPHA)
        state = _make_state()
        resampler._get_cdf_weights(state)
        corrected = resampler._corrected_log_weights.numpy()

        indices = tf.constant([[2, 0, 0, 4, 1],
                               [3, 3, 1, 0, 2]])
        flags_2d = tf.constant([[True], [True]])            # [batch, 1]
        new_log_w = resampler._compute_new_log_weights(
            state, indices, N, flags_2d
        ).numpy()

        np.testing.assert_allclose(
            new_log_w[0], corrected[0][[2, 0, 0, 4, 1]], atol=1e-12
        )
        np.testing.assert_allclose(
            new_log_w[1], corrected[1][[3, 3, 1, 0, 2]], atol=1e-12
        )

    def test_flags_false_preserves_original(self):
        """When flags=False, original log weights are kept unchanged."""
        resampler = SoftResampler(alpha=ALPHA)
        state = _make_state()
        resampler._get_cdf_weights(state)

        indices = tf.constant([[0, 1, 2, 3, 4],
                               [0, 1, 2, 3, 4]])
        flags_2d = tf.constant([[False], [False]])          # [batch, 1]
        new_log_w = resampler._compute_new_log_weights(
            state, indices, N, flags_2d
        )
        np.testing.assert_allclose(
            new_log_w.numpy(), state.log_weights.numpy(), atol=1e-12
        )


class TestGradientFlow:

    def test_gradient_nonzero_through_full_pipeline(self):
        """d(loss)/d(log_weights) must be non-zero after soft resampling,
        confirming the differentiable path through corrected weights."""
        log_w = tf.Variable(np.log(W), dtype=tf.float64)
        particles = tf.constant(
            np.arange(N, dtype=np.float64)[np.newaxis, :, np.newaxis]
            * np.ones((BATCH, 1, 1)),
            dtype=tf.float64,
        )
        resampler = SoftResampler(alpha=0.5, seed=42)

        with tf.GradientTape() as tape:
            state = State(particles=particles, log_weights=log_w)
            new_state = resampler.apply(state, tf.constant([True, True]))
            loss = tf.reduce_sum(new_state.log_weights)

        grad = tape.gradient(loss, log_w)
        assert grad is not None, "Gradient is None — no gradient path exists"
        assert grad.shape == (BATCH, N)
        assert tf.reduce_any(grad != 0.0).numpy(), "All gradients are zero"

    def test_gradient_through_get_cdf_weights(self):
        """d(cdf_weights)/d(log_weights) must be non-zero, verifying
        the soft weight computation is differentiable."""
        log_w = tf.Variable(np.log(W), dtype=tf.float64)
        particles = tf.constant(
            np.arange(N, dtype=np.float64)[np.newaxis, :, np.newaxis]
            * np.ones((BATCH, 1, 1)),
            dtype=tf.float64,
        )
        resampler = SoftResampler(alpha=0.5)

        with tf.GradientTape() as tape:
            state = State(particles=particles, log_weights=log_w)
            cdf_weights = resampler._get_cdf_weights(state)
            loss = tf.reduce_sum(cdf_weights)

        grad = tape.gradient(loss, log_w)
        assert grad is not None, "No gradient through _get_cdf_weights"
        assert grad.shape == (BATCH, N)
        assert tf.reduce_any(grad != 0.0).numpy()
