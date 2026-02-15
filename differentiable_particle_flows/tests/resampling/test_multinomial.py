"""Tests for dpf/resampling/standard/multinomial.py: MultinomialResampler."""

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

    # def test_high_weight_particle_duplicated(self):
    #     """Particle with w~1 should appear in most resampled slots."""
    #     resampler = MultinomialResampler()
    #     state = _make_state_degenerate()
    #     flags = tf.constant([True, True])
    #     new_state = resampler.apply(state, flags)
    #     # Particle 0 has w~1, so most resampled particles should be particle 0
    #     particle_0 = state.particles[:, 0:1, :].numpy()  # [batch, 1, D]
    #     resampled = new_state.particles.numpy()           # [batch, N, D]
    #     for b in range(BATCH):
    #         matches = np.all(
    #             np.isclose(resampled[b], particle_0[b], atol=1e-12), axis=-1
    #         )
    #         # With w~1, nearly all N particles should be copies of particle 0
    #         assert np.sum(matches) >= N - 2

    # def test_returns_state(self):
    #     resampler = MultinomialResampler()
    #     state = _make_state_uniform()
    #     flags = tf.constant([True, True])
    #     new_state = resampler.apply(state, flags)
    #     assert isinstance(new_state, State)


    # def test_ancestor_indices_set(self):
    #     """After resampling, ancestor_indices should be populated."""
    #     resampler = MultinomialResampler()
    #     state = _make_state_uniform()
    #     flags = tf.constant([True, True])
    #     new_state = resampler.apply(state, flags)
    #     assert new_state.ancestor_indices is not None
    #     assert new_state.ancestor_indices.shape == (BATCH, N)
