"""Tests for dpf/resampling/differentiable/regularised_ot_resample.py

Tests the full RegularisedOTResampler integration with the dpf State:
  - Shape preservation through the resampling pipeline.
  - Uniform weights after resampling.
  - Conditional resampling via flags (per-batch).
  - Gradient flow through resampling w.r.t. particles and log_weights.
  - End-to-end gradient through the SMC filter with OT resampling.

Reference: Corenflos et al. (2021), Algorithm 3 (DET Resampling).
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from dpf.base import State
from dpf import SMC
from dpf.observation.linear import LinearObservationModel
from dpf.transition import RandomWalkModel
from dpf.proposal.bootstrap import BootstrapProposal
from dpf.resampling.criterion import NeffCriterion
from dpf.resampling.differentiable.regularised_ot_resample import (
    RegularisedOTResampler,
)

tfd = tfp.distributions
DTYPE = tf.float64

B = 2       # batch size
N = 25      # particles
D = 2       # state dimension


def _make_state(batch=B, n_particles=N, dim=D, seed=42):
    """Create a State with random particles and non-uniform weights."""
    np.random.seed(seed)
    particles = tf.constant(np.random.randn(batch, n_particles, dim), dtype=DTYPE)
    raw_w = np.random.uniform(0.1, 1.0, [batch, n_particles])
    raw_w /= raw_w.sum(axis=1, keepdims=True)
    log_weights = tf.constant(np.log(raw_w), dtype=DTYPE)
    return State(particles=particles, log_weights=log_weights)


def _make_resampler(**kwargs):
    defaults = dict(epsilon=0.5, scaling=0.75, max_iter=100, convergence_threshold=1e-3)
    defaults.update(kwargs)
    return RegularisedOTResampler(**defaults)


# ────────────────────────────────────────────────────────────
# RegularisedOTResampler
# ────────────────────────────────────────────────────────────

class TestRegularisedOTResampler:

    def test_shape_preservation(self):
        """Particles and weights shapes should be preserved."""
        resampler = _make_resampler()
        state = _make_state()
        flags = tf.constant([True, True])
        new_state = resampler.apply(state, flags)
        assert new_state.particles.shape == (B, N, D)
        assert new_state.log_weights.shape == (B, N)

    def test_uniform_weights_after_resampling(self):
        """After OT resampling, weights should be reset to uniform 1/N."""
        resampler = _make_resampler()
        state = _make_state()
        flags = tf.constant([True, True])
        new_state = resampler.apply(state, flags)
        expected_log_w = -np.log(N)
        np.testing.assert_allclose(
            new_state.log_weights.numpy(), expected_log_w, atol=1e-2
        )

    def test_flags_false_no_op(self):
        """When flags=False, particles and weights should be unchanged."""
        resampler = _make_resampler()
        state = _make_state()
        flags = tf.constant([False, False])
        new_state = resampler.apply(state, flags)
        np.testing.assert_allclose(
            new_state.particles.numpy(), state.particles.numpy()
        )
        np.testing.assert_allclose(
            new_state.log_weights.numpy(), state.log_weights.numpy()
        )


# ────────────────────────────────────────────────────────────
# Gradient flow through the full resampler
# ────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradient_wrt_particles(self):
        """d(loss)/d(particles) should be non-zero through OT resampling."""
        state_base = _make_state()
        particles_var = tf.Variable(state_base.particles)
        resampler = _make_resampler()
        flags = tf.constant([True, True])

        with tf.GradientTape() as tape:
            state = State(particles=particles_var, log_weights=state_base.log_weights)
            new_state = resampler.apply(state, flags)
            loss = tf.reduce_sum(new_state.particles)

        grad = tape.gradient(loss, particles_var)
        assert grad is not None, "Gradient w.r.t. particles is None"
        assert tf.reduce_all(tf.math.is_finite(grad)).numpy(), \
            "Gradient w.r.t. particles has non-finite values"
        assert tf.reduce_any(grad != 0.).numpy(), "Gradient w.r.t. particles is all zeros"

    def test_gradient_wrt_log_weights(self):
        """d(loss)/d(log_weights) should be non-zero through OT resampling."""
        state_base = _make_state()
        logw_var = tf.Variable(state_base.log_weights)
        resampler = _make_resampler()
        flags = tf.constant([True, True])

        with tf.GradientTape() as tape:
            state = State(particles=state_base.particles, log_weights=logw_var)
            new_state = resampler.apply(state, flags)
            loss = tf.reduce_sum(new_state.particles)

        grad = tape.gradient(loss, logw_var)
        assert grad is not None, "Gradient w.r.t. log_weights is None"
        assert tf.reduce_all(tf.math.is_finite(grad)).numpy(), \
            "Gradient w.r.t. log_weights has non-finite values"
        assert tf.reduce_any(grad != 0.).numpy(), "Gradient w.r.t. log_weights is all zeros"


# ────────────────────────────────────────────────────────────
# End-to-end SMC integration
# ────────────────────────────────────────────────────────────

class TestSMCIntegration:

    def test_smc_gradient_wrt_parameter(self):
        """Gradient of log-likelihood w.r.t. a model parameter should exist.

        This is the key test: with OT resampling, d(log p(y))/d(theta)
        should be non-zero, confirming end-to-end differentiability.
        """
        np.random.seed(42)
        n_particles = 25
        T = 5
        batch_size = 2

        F = tf.eye(1, dtype=DTYPE)
        H = tf.eye(1, dtype=DTYPE)

        initial_particles = tf.constant(
            np.random.randn(batch_size, n_particles, 1), dtype=DTYPE)
        initial_state = State(
            particles=initial_particles,
            log_weights=tf.fill([batch_size, n_particles],
                                -tf.math.log(tf.cast(n_particles, DTYPE))))
        obs = tf.constant(np.random.randn(T, batch_size, 1), dtype=DTYPE)

        # Trainable parameter: log(sigma_obs)
        theta = tf.Variable(tf.constant(0.0, dtype=DTYPE))

        with tf.GradientTape() as tape:
            obs_noise = tfd.MultivariateNormalTriL(
                loc=tf.zeros([1], dtype=DTYPE),
                scale_tril=tf.reshape(tf.exp(theta), [1, 1]))
            trans_noise = tfd.MultivariateNormalTriL(
                loc=tf.zeros([1], dtype=DTYPE),
                scale_tril=tf.constant([[0.5]], dtype=DTYPE))
            tm = RandomWalkModel(F, trans_noise)
            smc = SMC(
                observation_model=LinearObservationModel(H, obs_noise),
                transition_model=tm,
                proposal_model=BootstrapProposal(tm),
                resampling_criterion=NeffCriterion(threshold_ratio=0.5),
                resampling_method=_make_resampler(),
            )
            final_state = smc(initial_state, obs)
            loss = -tf.reduce_mean(final_state.log_likelihoods)

        grad = tape.gradient(loss, theta)
        assert grad is not None, "Gradient w.r.t. theta is None — no gradient path"
        assert tf.math.is_finite(grad).numpy(), f"Gradient is not finite: {grad.numpy()}"
        assert grad.numpy() != 0.0, "Gradient is zero — differentiability broken"
