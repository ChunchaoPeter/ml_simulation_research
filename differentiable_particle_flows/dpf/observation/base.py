"""
Abstract base class for observation models.

Observation models define p(y_t | x_t) — the likelihood of an observation
given the hidden state.

All observation models must implement:
- loglikelihood(state, observation): Compute log p(y_t | x_t) for all particles.
- observation_function(particles): Compute h(x_t) — the predicted observation.

Extend this class to implement custom observation models for specific problems
(e.g., nonlinear sensors, bearings-only tracking, image observations).
"""

import abc
import tensorflow as tf
from dpf.base import State, Module


class ObservationModelBase(Module, metaclass=abc.ABCMeta):
    """
    Abstract observation model: p(y_t | x_t).

    Subclasses must implement:
        - loglikelihood(state, observation) -> updated State with new log_weights
        - observation_function(particles) -> predicted observations

    The loglikelihood method should:
        1. Compute predicted observations h(x) for all particles.
        2. Compute log p(y | x) using the observation noise distribution.
        3. Return an updated State with incremented log_weights.
    """

    @abc.abstractmethod
    def loglikelihood(self, state: State, observation: tf.Tensor) -> State:
        """
        Compute observation log-likelihood and update particle weights.

        Args:
            state: Current particle filter state.
                state.particles shape: [batch, n_particles, state_dim].
            observation: Observation at current time step.
                shape: [batch, obs_dim].

        Returns:
            New State with updated log_weights incorporating the
            observation likelihood. log_weights += log p(y_t | x_t^i).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def observation_function(self, particles: tf.Tensor) -> tf.Tensor:
        """
        Compute the observation function h(x) for each particle.

        Args:
            particles: Particle positions, shape [batch, n_particles, state_dim].

        Returns:
            Predicted observations, shape [batch, n_particles, obs_dim].
        """
        raise NotImplementedError

    # def observation_jacobian(self, particles: tf.Tensor) -> tf.Tensor:
    #     """
    #     Compute the Jacobian dh/dx of the observation function.

    #     Default: uses tf.GradientTape automatic differentiation.
    #     Override for analytical Jacobians (more efficient and stable).

    #     Args:
    #         particles: shape [batch, n_particles, state_dim].

    #     Returns:
    #         Jacobian, shape [batch, n_particles, obs_dim, state_dim].
    #     """
    #     # TODO: Implement auto-diff Jacobian computation via tf.GradientTape.
    #     # For linear models, override with the constant Jacobian matrix.
    #     raise NotImplementedError(
    #         "Default auto-diff Jacobian not yet implemented. "
    #         "Override observation_jacobian() with an analytical Jacobian."
    #     )
