"""
Abstract base class for proposal distributions.

Proposal models define q(x_t | x_{t-1}, y_t) — how new particles are
generated at each time step. The quality of the proposal directly affects
the variance of importance weights and thus filtering accuracy.
"""

import abc
import tensorflow as tf
from dpf.base import State, Module


class ProposalModelBase(Module, metaclass=abc.ABCMeta):
    """
    Abstract proposal distribution: q(x_t | x_{t-1}, y_t).

    Subclasses must implement:
        - propose(state, inputs, observation) -> State with proposed particles
        - loglikelihood(proposed, prior, inputs, observation) -> log q(x_t | ...)
    """

    @abc.abstractmethod
    def propose(self, state: State, inputs: tf.Tensor,
                observation: tf.Tensor, seed=None) -> State:
        """
        Propose new particles given current state and observation.

        Args:
            state: Current state at time t-1.
            inputs: Optional external inputs, shape [batch, input_dim].
            observation: Current observation y_t, shape [batch, obs_dim].
            seed: Random seed for reproducibility.

        Returns:
            New State with proposed particles at time t.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loglikelihood(self, proposed_state: State, prior_state: State,
                      inputs: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        """
        Evaluate the proposal log-density: log q(x_t | x_{t-1}, y_t).

        Required for importance weight computation when q ≠ p(x_t | x_{t-1}).

        Args:
            proposed_state: Proposed state at time t.
            prior_state: State at time t-1.
            inputs: Optional external inputs.
            observation: Current observation y_t.

        Returns:
            Log proposal density, shape [batch, n_particles].
        """
        raise NotImplementedError
