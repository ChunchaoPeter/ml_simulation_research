"""Tests for dpf/smc.py: SMC orchestrator.

Tests the Standard Particle Filter (Corenflos et al. 2021) which
approximates the filtering distribution p(x_t | y_{1:t}) using a
weighted particle set {X_t^i, w_t^i}_{i=1}^{N}.

Each time step t executes, steps 5-10:
    Steps 5-6:  Normalize weights w_{t-1}^i propto omega_{t-1}^i,
                resample X_tilde_{t-1}^i if ESS < threshold
    Step 7:     Propose X_t^i ~ q_phi(. | X_tilde_{t-1}^i, y_t)
    Step 8:     Compute omega_t^i = p_theta(X_t^i, y_t | X_tilde_{t-1}^i)
                                    / q_phi(X_t^i | X_tilde_{t-1}^i, y_t)
    Steps 9-10: Estimate p_hat(y_t | y_{1:t-1}) = (1/N) sum omega_t^i,
                accumulate ell(theta) += log p_hat(y_t | y_{1:t-1})

With bootstrap proposal (q = f), weights simplify to the observation
likelihood: omega_t^i = g(y_t | X_t^i).

After T steps, returns (step 12):
    ell(theta) = log p_hat_theta(y_{1:T})

Test classes:
    TestOneStep
        - Single-step filtering (steps 5-10 once): one_step() returns a
          valid State with correct particle and weight shapes.

    TestFullRun
        - Full sequence (steps 4-11): running over T observations produces
          correct final state shapes [batch, N, D].
        - Series output: return_series=True returns time-indexed tensors
          with shapes [T, batch, N, D], [T, batch, N], [T, batch].
        - Time tracking: final state.t equals T.
        - Non-trivial filtering: accumulated log-likelihoods are non-zero
          (step 10 actually accumulates values).

    TestLogMarginalLikelihood
        - Step 12: log_marginal_likelihood() returns shape [batch].
        - Finite values: no NaN or Inf in the output.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from dpf.base import State
from dpf.smc import SMC
from dpf.observation.linear import LinearObservationModel
from dpf.transition.linear_gaussian_state import LinearGaussianTransition as RandomWalkModel
from dpf.proposal.bootstrap import BootstrapProposal
from dpf.resampling.criterion import NeffCriterion
from dpf.resampling.standard import MultinomialResampler

tfd = tfp.distributions

BATCH = 3
N = 20
D = 1
T = 5


def _make_smc(d=D):
    """Assemble a standard particle filter for a 1D linear Gaussian model."""
    F = tf.eye(d, dtype=tf.float64)
    H = tf.eye(d, dtype=tf.float64)
    q_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([d], dtype=tf.float64),
        scale_tril=0.5 * tf.eye(d, dtype=tf.float64),
    )
    r_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([d], dtype=tf.float64),
        scale_tril=1.0 * tf.eye(d, dtype=tf.float64),
    )
    obs_model = LinearObservationModel(H, r_noise)
    trans_model = RandomWalkModel(F, q_noise)
    proposal = BootstrapProposal(trans_model)
    criterion = NeffCriterion(threshold_ratio=0.5)
    resampler = MultinomialResampler()
    return SMC(obs_model, trans_model, proposal, criterion, resampler)


def _make_initial_state(batch=BATCH, n=N, d=D):
    particles = tf.constant(np.random.randn(batch, n, d), dtype=tf.float64)
    log_weights = tf.fill([batch, n], -tf.math.log(tf.cast(n, tf.float64)))
    return State(particles=particles, log_weights=log_weights)


def _make_observations(t=T, batch=BATCH, d=D):
    return tf.constant(np.random.randn(t, batch, d), dtype=tf.float64)


class TestOneStep:
    """Test one_step(): a single pass through Algorithm 1, steps 5-10."""

    def test_shape_preserved(self):
        """Steps 5-10 should preserve particle shape [batch, N, D]."""
        smc = _make_smc()
        state = _make_initial_state()
        obs = tf.constant(np.random.randn(BATCH, D), dtype=tf.float64)
        new_state = smc.one_step(state, obs)
        assert new_state.particles.shape == (BATCH, N, D)
        assert new_state.log_weights.shape == (BATCH, N)


class TestFullRun:
    """Test __call__(): full filtering loop (steps 4-11)."""

    def test_final_state_shapes(self):
        """After T steps, final state has shapes [batch, N, D] and [batch]."""
        smc = _make_smc()
        state = _make_initial_state()
        observations = _make_observations()
        final_state = smc(state, observations)
        assert final_state.particles.shape == (BATCH, N, D)
        assert final_state.log_likelihoods.shape == (BATCH,)

    def test_return_series_shapes(self):
        """return_series=True stores all T time steps with leading time axis."""
        smc = _make_smc()
        state = _make_initial_state()
        observations = _make_observations()
        final_state, series = smc(state, observations, return_series=True)
        assert series['particles'].shape == (T, BATCH, N, D)
        assert series['log_weights'].shape == (T, BATCH, N)
        assert series['log_likelihoods'].shape == (T, BATCH)
        assert series['particles'][-1] == final_state.particles
        assert series['log_weights'][-1] == final_state.log_weights
        assert series['log_likelihoods'][-1] == final_state.log_likelihoods
        """After the loop (step 11: end for), state.t should equal T."""
        assert final_state.t == T


    def test_log_likelihoods_nonzero(self):
        """Step 10 accumulates ell(theta); after T steps it should be non-zero."""
        smc = _make_smc()
        state = _make_initial_state()
        observations = _make_observations()
        final_state = smc(state, observations)
        assert not np.allclose(final_state.log_likelihoods.numpy(), 0.0)


class TestLogMarginalLikelihood:
    """Test log_marginal_likelihood(): returns ell(theta) (step 12)."""

    def test_shape(self):
        """ell(theta) = log p_hat(y_{1:T}) should have shape [batch]."""
        smc = _make_smc()
        state = _make_initial_state()
        observations = _make_observations()
        lml = smc.log_marginal_likelihood(state, observations)
        assert lml.shape == (BATCH,)

    def test_finite_values(self):
        """Log marginal likelihood should be finite (no NaN or Inf)."""
        smc = _make_smc()
        state = _make_initial_state()
        observations = _make_observations()
        lml = smc.log_marginal_likelihood(state, observations)
        assert np.all(np.isfinite(lml.numpy()))
