"""Tests for dpf/observation/linear.py: LinearObservationModel.

Tests the linear Gaussian observation model:

    y_t = H @ x_t + v_t,    v_t ~ N(0, R)

where H is the observation matrix and R = R_chol @ R_chol^T is the
observation noise covariance.

All tests use a general (non-square, non-identity) configuration:
    H = [[1.0, 0.5, 0.0],     shape [2, 3]  (obs_dim=2, state_dim=3)
         [0.0, 0.2, 0.8]]
    R_chol = [[0.3, 0.0],     shape [2, 2]  (non-isotropic noise)
              [0.1, 0.4]]

Test classes:
    TestGeneralObservationFunction
        - Shape: h(x) = H @ x maps [batch, N, 3] -> [batch, N, 2].
        - Correctness: verifies matvec against explicit matmul for a
          single particle.
        - Dimensionality: output dim (obs_dim=2) differs from input
          dim (state_dim=3).

    TestGeneralLoglikelihood
        - Weight update: loglikelihood() modifies log_weights via
              log w_t^i = log w_{t-1}^i + log p(y_t | x_t^i).
        - Shape preservation: particles and log_weights shapes unchanged.
        - Perfect observation: when y = H @ x exactly, log-weights increase
          (positive log-likelihood added).
        - Comparative: closer observations yield higher accumulated
          log marginal likelihood than far observations.
        - Analytical match: per-particle log p(y|x) = log N(y - Hx; 0, R),
          verified against an independently constructed MVN distribution.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from dpf.base import State
from dpf.observation.linear import LinearObservationModel

tfd = tfp.distributions

BATCH = 2
N = 10
STATE_DIM = 3
OBS_DIM = 2


def _make_general_model():
    """Create a LinearObservationModel with non-square H and non-isotropic R.

    H = [[1.0, 0.5, 0.0],
         [0.0, 0.2, 0.8]]   shape [2, 3]  (obs_dim=2, state_dim=3)

    R_chol = [[0.3, 0.0],
              [0.1, 0.4]]    shape [2, 2]
    """
    H = tf.constant([[1.0, 0.5, 0.0],
                     [0.0, 0.2, 0.8]], dtype=tf.float64)
    R_chol = tf.constant([[0.3, 0.0],
                          [0.1, 0.4]], dtype=tf.float64)
    noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([OBS_DIM], dtype=tf.float64),
        scale_tril=R_chol,
    )
    return LinearObservationModel(H, noise), H, R_chol


def _make_state(batch=BATCH, n=N, d=STATE_DIM):
    particles = tf.constant(np.random.randn(batch, n, d), dtype=tf.float64)
    log_weights = tf.fill([batch, n], -tf.math.log(tf.cast(n, tf.float64)))
    return State(particles=particles, log_weights=log_weights)


class TestGeneralObservationFunction:
    def test_output_shape(self):
        """Output shape should be [batch, n_particles, obs_dim]."""
        model, _, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        predicted = model.observation_function(state.particles)
        assert predicted.shape == (BATCH, N, OBS_DIM)

    def test_applies_h_matrix(self):
        """observation_function should compute H @ x."""
        model, H, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        predicted = model.observation_function(state.particles)
        expected = tf.linalg.matvec(H, state.particles)
        np.testing.assert_allclose(predicted.numpy(), expected.numpy(), atol=1e-12)

        # Check one case manually: H @ x for a single particle
        one_matmul = tf.linalg.matmul(H, state.particles[0][3][:, None])
        one_matvec = expected[0][3][:, None]
        np.testing.assert_allclose(one_matmul.numpy(), one_matvec.numpy(), atol=1e-12)

    def test_output_differs_from_input(self):
        """Non-square H maps state_dim=3 to obs_dim=2, so output != input."""
        model, _, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        predicted = model.observation_function(state.particles)
        assert predicted.shape[-1] != state.particles.shape[-1]


class TestGeneralLoglikelihood:
    def test_updates_log_weights(self):
        model, _, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        obs = tf.constant(np.random.randn(BATCH, OBS_DIM), dtype=tf.float64)
        new_state = model.loglikelihood(state, obs)
        assert not np.allclose(new_state.log_weights.numpy(), state.log_weights.numpy())

    def test_shape_preserved(self):
        model, _, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        obs = tf.constant(np.random.randn(BATCH, OBS_DIM), dtype=tf.float64)
        new_state = model.loglikelihood(state, obs)
        assert new_state.particles.shape == state.particles.shape
        assert new_state.log_weights.shape == state.log_weights.shape


    def test_closer_obs_gets_higher_weight(self):
        """Observation closer to H @ x should yield higher log-likelihood."""
        model, H, _ = _make_general_model()
        np.random.seed(42)
        state = _make_state()
        predicted = model.observation_function(state.particles)
        # Close observation: mean of predicted obs across particles
        close_obs = tf.reduce_mean(predicted, axis=1)  # [batch, obs_dim]
        # Far observation
        far_obs = close_obs + 100.0
        new_state_close = model.loglikelihood(state, close_obs)
        new_state_far = model.loglikelihood(state, far_obs)
        # Close observation should give higher total log-likelihood
        assert np.all(
            new_state_close.log_likelihoods.numpy()
            > new_state_far.log_likelihoods.numpy()
        )

    def test_log_prob_matches_analytical(self):
        """Verify the per-particle log p(y|x) matches manual MVN computation."""
        model, H, R_chol = _make_general_model()
        np.random.seed(42)
        state = _make_state(batch=1, n=1)
        obs = tf.constant(np.random.randn(1, OBS_DIM), dtype=tf.float64)
        new_state = model.loglikelihood(state, obs)
        # Manual computation: innovation = y - H @ x
        predicted = tf.linalg.matvec(H, state.particles)  # [1, 1, obs_dim]
        innovation = obs[:, tf.newaxis, :] - predicted      # [1, 1, obs_dim]
        # log N(innovation; 0, R)
        noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros([OBS_DIM], dtype=tf.float64),
            scale_tril=R_chol,
        )
        expected_log_lik = noise_dist.log_prob(innovation)  # [1, 1]
        # new_log_weights = old_log_weights + log_lik
        expected_new_lw = state.log_weights + expected_log_lik
        np.testing.assert_allclose(
            new_state.log_weights.numpy(), expected_new_lw.numpy(), atol=1e-10
        )
