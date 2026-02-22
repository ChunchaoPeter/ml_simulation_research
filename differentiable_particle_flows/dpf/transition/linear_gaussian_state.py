"""
Random walk (linear Gaussian) state transition model.

x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)

where F is the state transition matrix and Q is the process noise covariance.
"""

import attr
import tensorflow as tf
import tensorflow_probability as tfp
from dpf.base import State
from dpf.transition.base import TransitionModelBase

tfd = tfp.distributions


class LinearGaussianTransition(TransitionModelBase):
    """
    Linear Gaussian state transition: x_t = F @ x_{t-1} + w_t.

    Attributes:
        transition_matrix: F matrix, shape [state_dim, state_dim].
        transition_noise: TFP distribution for w_t.
    """

    def __init__(self, transition_matrix: tf.Tensor,
                 transition_noise: tfd.Distribution,
                 name: str = 'RandomWalkModel'):
        """
        Args:
            transition_matrix: F matrix, shape [state_dim, state_dim].
            transition_noise: TFP distribution for w_t.
                Must support .sample() and .log_prob() with matching shapes.
        """
        super().__init__(name=name)
        self.transition_matrix = tf.cast(transition_matrix, tf.float64)
        self.transition_noise = transition_noise

    def transition_function(self, particles: tf.Tensor,
                             inputs: tf.Tensor = None) -> tf.Tensor:
        """Deterministic transition: f(x) = F @ x.

        Args:
            particles: shape [batch, n_particles, state_dim].

        Returns:
            shape [batch, n_particles, state_dim].
        """
        # tf.linalg.matvec(self.transition_matrix, particles) is more efficient than tf.linalg.matmul for this case
        # tf.linalg.matvec(self.transition_matrix, particles) computes F @ x for each particle efficiently, broadcasting over batch and n_particles dimensions.
        # tf.linalg.matmul(self.transition_matrix, particles) would require reshaping and broadcasting, which is less efficient.

        return tf.linalg.matvec(self.transition_matrix, particles)

    # def transition_jacobian(self, particles: tf.Tensor,
    #                          inputs: tf.Tensor = None) -> tf.Tensor:
    #     """Analytical Jacobian: df/dx = F (constant).

    #     Returns:
    #         F broadcast to shape [batch, n_particles, state_dim, state_dim].
    #     """
    #     TODO test it when we need it
    #     batch_size = tf.shape(particles)[0]
    #     n_particles = tf.shape(particles)[1]
    #     return tf.broadcast_to(
    #         self.transition_matrix[tf.newaxis, tf.newaxis, :, :],
    #         [batch_size, n_particles,
    #          self.transition_matrix.shape[0],
    #          self.transition_matrix.shape[1]]
    #     )

    def sample(self, state: State, inputs: tf.Tensor = None) -> State:
        """Propagate: x_t = F @ x_{t-1} + w_t.

        Args:
            state: Current state.

        Returns:
            New State with propagated particles.
        """
        mean = self.transition_function(state.particles)
        # sample noise with shape [batch, n_particles, state_dim]
        # transition_noise.sample() should support sample_shape=[batch, n_particles]
        # tf.shape(state.particles)[:2] gives [batch, n_particles]
        noise = self.transition_noise.sample(
            sample_shape=tf.shape(state.particles)[:2]
        )
        new_particles = mean + tf.cast(noise, mean.dtype)
        return attr.evolve(state, particles=new_particles)

    def loglikelihood(self, prior_state: State, proposed_state: State,
                      inputs: tf.Tensor = None) -> tf.Tensor:
        """Evaluate log p(x_t | x_{t-1}) = log N(x_t; F @ x_{t-1}, Q).

        Returns:
            Log density, shape [batch, n_particles].
        """
        mean = self.transition_function(prior_state.particles)
        diff = proposed_state.particles - mean
        return self.transition_noise.log_prob(diff)
