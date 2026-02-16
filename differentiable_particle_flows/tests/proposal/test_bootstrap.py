"""Tests for dpf/proposal/bootstrap.py: BootstrapProposal."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from dpf.base import State
from dpf.transition.linear_gaussian_state import LinearGaussianTransition
from dpf.proposal.bootstrap import BootstrapProposal

tfd = tfp.distributions

BATCH = 2
N = 10
D = 2


def _make_proposal(d=D):
    F = tf.eye(d, dtype=tf.float64)
    noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([d], dtype=tf.float64),
        scale_tril=0.1 * tf.eye(d, dtype=tf.float64),
    )
    transition = LinearGaussianTransition(F, noise)
    return BootstrapProposal(transition)


def _make_state(batch=BATCH, n=N, d=D):
    particles = tf.constant(np.random.randn(batch, n, d), dtype=tf.float64)
    log_weights = tf.fill([batch, n], -tf.math.log(tf.cast(n, tf.float64)))
    return State(particles=particles, log_weights=log_weights)


class TestBootstrapProposal:
    def test_shape_preserved(self):
        proposal = _make_proposal()
        state = _make_state()
        obs = tf.constant(np.random.randn(BATCH, D), dtype=tf.float64)
        new_state = proposal.propose(state, inputs=None, observation=obs)
        assert new_state.particles.shape == state.particles.shape

    def test_particles_change(self):
        """Bootstrap proposal propagates through transition → particles change."""
        proposal = _make_proposal()
        state = _make_state()
        obs = tf.constant(np.random.randn(BATCH, D), dtype=tf.float64)
        new_state = proposal.propose(state, inputs=None, observation=obs)
        assert not np.allclose(new_state.particles.numpy(), state.particles.numpy())

    def test_loglikelihood_shape(self):
        proposal = _make_proposal()
        state = _make_state()
        obs = tf.constant(np.random.randn(BATCH, D), dtype=tf.float64)
        proposed = proposal.propose(state, inputs=None, observation=obs)
        ll = proposal.loglikelihood(proposed, state, inputs=None, observation=obs)
        assert ll.shape == (BATCH, N)
