"""Tests for dpf/transition/linear_gaussian_state.py: LinearGaussianTransition.

Tests the linear Gaussian state transition model:

    x_t = F @ x_{t-1} + w_t,    w_t ~ N(0, Q)

where F is the state transition matrix and Q = Q_chol @ Q_chol^T is the
process noise covariance.

All tests use a general (non-identity) configuration to exercise the full
matrix algebra:
    F = [[0.9, 0.1], [0.0, 0.8]]     (2D coupled dynamics)
    Q_chol = [[0.5, 0.0], [0.2, 0.3]] (non-isotropic, correlated noise)

Test classes:
    TestGeneralTransitionFunction
        - Verifies transition_function computes f(x) = F @ x via matvec,
          cross-checked against explicit matmul for a single particle.
        - Confirms non-identity F produces output different from input.

    TestGeneralSample
        - Shape preservation: sample() returns [batch, n_particles, state_dim].
        - Statistical correctness: over many draws, E[x_t | x_{t-1}] -> F @ x_{t-1}
          by the law of large numbers (Monte Carlo test).

    TestGeneralLoglikelihood
        - Shape: log p(x_t | x_{t-1}) has shape [batch, n_particles].
        - Optimality: log-density is maximized when proposed = F @ prior
          (zero-residual case).
        - Analytical match: log p(x_t | x_{t-1}) = log N(x_t - F x_{t-1}; 0, Q),
          verified against an independently constructed MVN distribution.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from dpf.base import State
from dpf.transition.linear_gaussian_state import LinearGaussianTransition

tfd = tfp.distributions

BATCH = 2
N = 10
D = 2


def _make_general_model():
    """Create a LinearGaussianTransition with a non-identity F and non-isotropic Q.

    Uses a 2D nearly-constant-velocity model:
        F = [[0.9, 0.1],
             [0.0, 0.8]]
        Q_chol = [[0.5, 0.0],
                   [0.2, 0.3]]
    """
    F = tf.constant([[0.9, 0.1],
                     [0.0, 0.8]], dtype=tf.float64)
    Q_chol = tf.constant([[0.5, 0.0],
                          [0.2, 0.3]], dtype=tf.float64)
    noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], dtype=tf.float64),
        scale_tril=Q_chol,
    )
    return LinearGaussianTransition(F, noise), F, Q_chol


def _make_state(batch=BATCH, n=N, d=D):
    particles = tf.constant(np.random.randn(batch, n, d), dtype=tf.float64)
    log_weights = tf.fill([batch, n], -tf.math.log(tf.cast(n, tf.float64)))
    return State(particles=particles, log_weights=log_weights)


# -----------------------------------------------------------------------
# General (non-identity F, non-isotropic Q) linear Gaussian tests
# -----------------------------------------------------------------------

class TestGeneralTransitionFunction:
    def test_general_f_applies_matrix(self):
        """With non-identity F, transition_function should compute F @ x."""
        model, F, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        result = model.transition_function(state.particles)
        expected = tf.linalg.matvec(F, state.particles)

        ## check one case manually to verify the matvec is doing what we expect
        one_matmul_case = tf.linalg.matmul(F, state.particles[1][6][:,None])
        one_matvecl_case = expected[1][6][:,None]
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-12)
        np.testing.assert_allclose(one_matmul_case.numpy(), one_matvecl_case.numpy(), atol=1e-12)


    def test_general_f_differs_from_input(self):
        """Non-identity F should change the particles."""
        model, _, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        result = model.transition_function(state.particles)
        assert not np.allclose(result.numpy(), state.particles.numpy())


class TestGeneralSample:
    def test_output_shape(self):
        model, _, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        new_state = model.sample(state)
        assert new_state.particles.shape == state.particles.shape

    def test_mean_is_f_times_x(self):
        """Over many samples, the sample mean should approach F @ x."""
        model, F, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state(batch=1, n=1)
        expected_mean = tf.linalg.matvec(F, state.particles)  # [1, 1, 2]
        # Draw many samples and average
        samples = []
        for _ in range(1000):
            new_state = model.sample(state)
            samples.append(new_state.particles.numpy())
        sample_mean = np.mean(samples, axis=0)
        np.testing.assert_allclose(
            sample_mean, expected_mean.numpy(), atol=0.05
        )


class TestGeneralLoglikelihood:
    def test_output_shape(self):
        model, _, _ = _make_general_model()
        np.random.seed(42)
        prior = _make_state()
        proposed = _make_state()
        ll = model.loglikelihood(prior, proposed)
        assert ll.shape == (BATCH, N)

    def test_log_prob_at_mean_is_maximum(self):
        """log p(F@x | x) should be the maximum (zero residual)."""
        model, F, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        mean_state = State(
            particles=model.transition_function(state.particles),
            log_weights=state.log_weights,
        )
        ll_mean = model.loglikelihood(state, mean_state)
        # Perturbed state should have lower log-likelihood
        perturbed_state = State(
            particles=mean_state.particles + 1.0,
            log_weights=state.log_weights,
        )
        ll_perturbed = model.loglikelihood(state, perturbed_state)
        assert np.all(ll_mean.numpy() > ll_perturbed.numpy())

    def test_log_prob_matches_analytical(self):
        """Verify log-likelihood matches manual MVN computation."""
        model, F, Q_chol = _make_general_model()
        np.random.seed(42)
        state = _make_state(batch=1, n=1)
        proposed = _make_state(batch=1, n=1)
        ll = model.loglikelihood(state, proposed)
        # Manual: diff = proposed - F @ prior, then log_prob under N(0, Q)
        mean = tf.linalg.matvec(F, state.particles)
        diff = proposed.particles - mean  # [1, 1, 2]
        Q = Q_chol @ tf.transpose(Q_chol)
        dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros([2], dtype=tf.float64),
            covariance_matrix=Q,
        )
        expected_ll = dist.log_prob(diff)  # [1, 1]
        np.testing.assert_allclose(
            ll.numpy(), expected_ll.numpy(), atol=1e-10
        )
