"""
Linear observation model: y_t = H @ x_t + v_t, v_t ~ N(0, R).

The simplest observation model. Useful for testing and as a reference
implementation for the ObservationModelBase interface.
"""

import attr
import tensorflow as tf
import tensorflow_probability as tfp
from dpf.base import State
from dpf.observation.base import ObservationModelBase

tfd = tfp.distributions


class LinearObservationModel(ObservationModelBase):
    """
    Linear Gaussian observation model.

    y_t = H @ x_t + v_t,  where v_t ~ observation_noise_dist

    Attributes:
        observation_matrix: H matrix, shape [obs_dim, state_dim].
        observation_noise: TFP distribution for observation noise (e.g., MultivariateNormalTriL).
    """

    def __init__(self, observation_matrix: tf.Tensor,
                 observation_noise: tfd.Distribution,
                 name: str = 'LinearObservationModel'):
        """
        Args:
            observation_matrix: H matrix, shape [obs_dim, state_dim].
            observation_noise: TFP distribution for v_t.
                Must support .log_prob() with shape [batch, n_particles, obs_dim].
        """
        super().__init__(name=name)
        self.observation_matrix = tf.cast(observation_matrix, tf.float64)
        self.observation_noise = observation_noise

    def observation_function(self, particles: tf.Tensor) -> tf.Tensor:
        """Compute h(x) = H @ x for each particle.

        Args:
            particles: shape [batch, n_particles, state_dim].

        Returns:
            Predicted observations, shape [batch, n_particles, obs_dim].
        """
        # H: [obs_dim, state_dim], particles: [batch, n_particles, state_dim]
        # Result: [batch, n_particles, obs_dim]
        return tf.linalg.matvec(self.observation_matrix, particles)

    # def observation_jacobian(self, particles: tf.Tensor) -> tf.Tensor:
    #     """Analytical Jacobian: dh/dx = H (constant for linear model).

    #     Args:
    #         particles: shape [batch, n_particles, state_dim].

    #     Returns:
    #         Jacobian, shape [batch, n_particles, obs_dim, state_dim].
    #         (H broadcast to all batch elements and particles.)
    #     """
    #     TODO test it when we need it
    #     batch_size = tf.shape(particles)[0]
    #     n_particles = tf.shape(particles)[1]
    #     # Broadcast H to [batch, n_particles, obs_dim, state_dim]
    #     return tf.broadcast_to(
    #         self.observation_matrix[tf.newaxis, tf.newaxis, :, :],
    #         [batch_size, n_particles,
    #          self.observation_matrix.shape[0],
    #          self.observation_matrix.shape[1]]
    #     )

    def loglikelihood(self, state: State, observation: tf.Tensor) -> State:
        """Compute log p(y_t | x_t) and update weights.

        Args:
            state: Current state.
            observation: shape [batch, obs_dim].

        One observation $y_t$ per time step per batch so it is [batch, obs_dim].

        Returns:
            Updated State with new log_weights.
        """
        # Predicted observations: h(x_t^i) = H @ x_t^i, shape [batch, n_particles, obs_dim]
        predicted = self.observation_function(state.particles)

        # Innovation (residual): e_t^i = y_t - H @ x_t^i
        # observation [batch, obs_dim] is broadcast across n_particles
        innovation = observation[:, tf.newaxis, :] - predicted

        # Observation log-likelihood per particle.
        #
        # Since y_t = H x_t + v_t with v_t ~ N(0, R), the conditional is:
        #   p(y_t | x_t) = N(y_t; H x_t, R)
        #
        # The Gaussian density depends only on the residual from the mean:
        #   N(y; mu, R) = N(y - mu; 0, R)
        #
        # Therefore:
        #   log p(y_t | x_t^i) = log N(y_t - H x_t^i; 0, R)
        #                       = log N(e_t^i; 0, R)
        log_liks = self.observation_noise.log_prob(innovation)  # [batch, n_particles]

        # Weight update.
        # The algorithm writes it in weight space:
        #   w_tilde_t^i = w_{t-1}^i * g(z_t | X_t^i)
        #
        # We work in log space for numerical stability (weights can be ~1e-300):
        #   log w_tilde_t^i = log w_{t-1}^i + log g(z_t | X_t^i)
        new_log_weights = state.log_weights + log_liks

        # Accumulated log marginal likelihood:
        #   log p(y_t | y_{1:t-1}) ≈ log(1/N * sum_i exp(log p(y_t | x_t^i)))
        n = tf.cast(tf.shape(log_liks)[-1], tf.float64)
        step_log_ml = tf.reduce_logsumexp(log_liks, axis=-1) - tf.math.log(n)
        new_log_likelihoods = state.log_likelihoods + step_log_ml

        return attr.evolve(
            state,
            log_weights=new_log_weights,
            weights=tf.nn.softmax(new_log_weights, axis=-1),
            log_likelihoods=new_log_likelihoods,
            ess=1.0 / tf.reduce_sum(
                tf.nn.softmax(new_log_weights, axis=-1) ** 2, axis=-1
            ),
        )
