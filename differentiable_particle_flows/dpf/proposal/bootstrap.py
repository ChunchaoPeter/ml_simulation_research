"""
Bootstrap proposal: q(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1}).

The simplest proposal — just the transition prior. Ignores the current
observation y_t.

This is the standard choice in basic particle filters (SIR/SIS).
"""

import tensorflow as tf
from dpf.base import State
from dpf.proposal.base import ProposalModelBase
from dpf.transition.base import TransitionModelBase


class BootstrapProposal(ProposalModelBase):
    """
    Bootstrap proposal: propose from the transition prior.

    q(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1})

    Since the proposal equals the transition, the importance weight
    simplifies to: w_t ∝ p(y_t | x_t).

    Attributes:
        transition_model: The state transition model used as the proposal.
    """

    def __init__(self, transition_model: TransitionModelBase,
                 name: str = 'BootstrapProposal'):
        super().__init__(name=name)
        self.transition_model = transition_model

    def propose(self, state: State, inputs: tf.Tensor = None,
                observation: tf.Tensor = None, seed=None) -> State:
        """Propose by sampling from the transition model.

        Args:
            state: Current state at t-1.
            inputs: Optional controls.
            observation: Unused (bootstrap ignores observation).

        Returns:
            State with particles propagated through transition.
        """
        return self.transition_model.sample(state, inputs)

    def loglikelihood(self, proposed_state: State, prior_state: State,
                      inputs: tf.Tensor = None,
                      observation: tf.Tensor = None) -> tf.Tensor:
        """Evaluate log q = log p(x_t | x_{t-1}).

        For bootstrap proposal, this equals the transition log-density.

        Returns:
            Log proposal density, shape [batch, n_particles].
        """
        return self.transition_model.loglikelihood(
            prior_state, proposed_state, inputs
        )
