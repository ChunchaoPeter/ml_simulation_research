"""
Abstract base class for state transition models.

Transition models define p(x_t | x_{t-1}) — how the hidden state evolves
over time. They provide two key operations:
    - sample(): Draw x_t ~ p(x_t | x_{t-1}) for particle propagation.
    - loglikelihood(): Evaluate log p(x_t | x_{t-1}) for weight computation.
"""

import abc
import tensorflow as tf
from dpf.base import State, Module


class TransitionModelBase(Module, metaclass=abc.ABCMeta):
    """
    Abstract state transition model: p(x_t | x_{t-1}).

    Subclasses must implement:
        - sample(state, inputs) -> new State with propagated particles
        - loglikelihood(prior_state, proposed_state, inputs) -> log p(x_t | x_{t-1})
    """

    @abc.abstractmethod
    def sample(self, state: State, inputs: tf.Tensor = None) -> State:
        """
        Propagate particles through the state transition with noise.

        x_t^i ~ p(x_t | x_{t-1}^i) for each particle i.

        Args:
            state: Current particle filter state.
                state.particles shape: [batch, n_particles, state_dim].
            inputs: Optional external inputs/controls, shape [batch, input_dim].

        Returns:
            New State with propagated particles.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loglikelihood(self, prior_state: State, proposed_state: State,
                      inputs: tf.Tensor = None) -> tf.Tensor:
        """
        Evaluate the transition log-density.

        log p(x_t | x_{t-1}) for each particle pair.

        Args:
            prior_state: State at t-1.
            proposed_state: State at t (proposed particles).
            inputs: Optional external inputs.

        Returns:
            Log transition density, shape [batch, n_particles].
        """
        raise NotImplementedError
