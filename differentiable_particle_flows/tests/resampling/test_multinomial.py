"""Tests for dpf/resampling/standard/multinomial.py: MultinomialResampler.

Tests multinomial resampling:

    For i = 1 to N:
        Draw A^i ~ Categorical(w_t^1, ..., w_t^N)
        Set X_t^i = X_t^{A^i}
    Set w_t^i = 1/N  for all i

Each particle is selected independently with probability proportional to
its normalized weight. After resampling, weights are reset to uniform.

Test class:
    TestMultinomialResampler
        - No-op when flags=False: particles and weights unchanged.
        - Weight reset: after resampling, all log_weights = -log(N).
        - Per-batch flags: only flagged batch elements are resampled;
          unflagged elements keep original particles and weights.
        - Ancestor indices: resampler populates ancestor_indices field
          with shape [batch, n_particles].
        - Statistical correctness: over R=500 rounds, empirical selection
          frequencies converge to the true weights (law of large numbers).
          Verified with known weights [0.4, 0.3, 0.15, 0.1, 0.05].
"""

import tensorflow as tf
import numpy as np
from dpf.base import State
from dpf.resampling.standard import MultinomialResampler
from dpf.constants import DEFAULT_SEED

BATCH = 2
N = 50  # More particles for statistical tests
D = 2


def _make_state_uniform():
    particles = tf.constant(np.random.randn(BATCH, N, D), dtype=tf.float64)
    log_weights = tf.fill([BATCH, N], -tf.math.log(tf.cast(N, tf.float64)))
    return State(particles=particles, log_weights=log_weights)


def _make_state_degenerate():
    """All weight on particle 0."""
    particles = tf.constant(np.random.randn(BATCH, N, D), dtype=tf.float64)
    lw = np.full((BATCH, N), -1e10)
    lw[:, 0] = 0.0
    log_weights = tf.constant(lw, dtype=tf.float64)
    return State(particles=particles, log_weights=log_weights)


class TestMultinomialResampler:
    def test_no_sampling_results(self):
        resampler = MultinomialResampler()
        state = _make_state_uniform()
        flags = tf.constant([False, False])
        new_state = resampler.apply(state, flags)
        np.testing.assert_allclose(
            new_state.particles.numpy(), state.particles.numpy()
        )
        np.testing.assert_allclose(
            new_state.log_weights.numpy(), state.log_weights.numpy()
        )

    def test_uniform_weights_after_resampling(self):
        """After resampling, weights should be reset to uniform."""
        resampler = MultinomialResampler()
        state = _make_state_degenerate()
        flags = tf.constant([True, True])
        new_state = resampler.apply(state, flags)
        expected_log_w = -np.log(N)
        np.testing.assert_allclose(
            new_state.log_weights.numpy(), expected_log_w, atol=1e-10
        )


    def test_mixed_particles_flags(self):
        resampler = MultinomialResampler()
        state = _make_state_uniform()
        flags = tf.constant([False, True])
        new_state = resampler.apply(state, flags)
        np.testing.assert_allclose(
            new_state.particles.numpy()[0], state.particles.numpy()[0]
        )
        assert not np.allclose(
            new_state.particles.numpy()[1],
            state.particles.numpy()[1],
        )

    def test_mixed_weight_flags(self):
        """Batch 0: resample, Batch 1: keep. Check independently."""
        resampler = MultinomialResampler()
        state = _make_state_degenerate()
        flags = tf.constant([True, False])
        new_state = resampler.apply(state, flags)
        # Batch 0: weights should be uniform (resampled)
        np.testing.assert_allclose(
            new_state.log_weights[0].numpy(), -np.log(N), atol=1e-10
        )
        # Batch 1: weights should be unchanged (not resampled)
        np.testing.assert_allclose(
            new_state.log_weights[1].numpy(), state.log_weights[1].numpy()
        )

    def test_returns_state(self):
        resampler = MultinomialResampler()
        state = _make_state_uniform()
        flags = tf.constant([True, True])
        new_state = resampler.apply(state, flags)
        assert isinstance(new_state, State)


    def test_ancestor_indices_set(self):
        """After resampling, ancestor_indices should be populated."""
        resampler = MultinomialResampler()
        state = _make_state_uniform()
        flags = tf.constant([True, True])
        new_state = resampler.apply(state, flags)
        assert new_state.ancestor_indices is not None
        assert new_state.ancestor_indices.shape == (BATCH, N)

    def test_ancestor_indices_identity_when_no_resampling(self):
        """When flags=False, ancestor_indices should be identity [0, 1, ..., N-1]."""
        resampler = MultinomialResampler()
        state = _make_state_uniform()
        flags = tf.constant([False, False])
        new_state = resampler.apply(state, flags)
        expected = np.broadcast_to(np.arange(N)[np.newaxis, :], (BATCH, N))
        np.testing.assert_array_equal(
            new_state.ancestor_indices.numpy(), expected
        )

    def test_ancestor_indices_mixed_flags(self):
        """Batch 0 (no resample): identity indices. Batch 1 (resample): changed."""
        resampler = MultinomialResampler()
        state = _make_state_degenerate()
        flags = tf.constant([False, True])
        new_state = resampler.apply(state, flags)
        # Batch 0: identity (no resampling, each particle is its own ancestor)
        np.testing.assert_array_equal(
            new_state.ancestor_indices.numpy()[0], np.arange(N)
        )
        # Batch 1: resampled (degenerate => all indices should be 0)
        np.testing.assert_array_equal(
            new_state.ancestor_indices.numpy()[1], np.zeros(N, dtype=np.int32)
        )

    def test_multinomial_sampling_frequencies(self):
        """Empirical selection frequencies should match the weights.

        Multinomial resampling satisfies E[N_k] = N * w_k, where N_k is the
        number of times particle k is selected. Over R independent resampling
        rounds, the empirical frequency of selecting particle k should converge
        to w_k by the law of large numbers.

        We set up 5 particles with known weights [0.4, 0.3, 0.15, 0.1, 0.05],
        run R=500 resampling rounds (each producing N=5 ancestor indices), and
        check that the empirical proportions are within tolerance of the true
        weights.
        """
        n = 5
        target_weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        log_weights = tf.constant(
            np.log(target_weights)[np.newaxis, :],  # [1, 5]
            dtype=tf.float64,
        )
        particles = tf.constant(
            np.arange(n)[np.newaxis, :, np.newaxis],  # [1, 5, 1]
            dtype=tf.float64,
        )
        state = State(particles=particles, log_weights=log_weights)
        flags = tf.constant([True])

        R = 500
        counts = np.zeros(n)
        resampler = MultinomialResampler(seed=42)
        for _ in range(R):
            new_state = resampler.apply(state, flags)
            indices = new_state.ancestor_indices.numpy()[0]  # [N]
            for idx in indices:
                counts[idx] += 1

        # Empirical frequencies over R*N total draws
        empirical_freq = counts / counts.sum()

        # With R=500 rounds and N=5, we have 2500 total samples.
        # Allow tolerance of 0.05 for each weight bucket.
        np.testing.assert_allclose(
            empirical_freq, target_weights, atol=0.05,
            err_msg=(
                f"Empirical frequencies {empirical_freq} do not match "
                f"target weights {target_weights}"
            ),
        )
