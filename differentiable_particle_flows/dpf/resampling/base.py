"""
Abstract base class for resamplers.

ResamplerBase defines the interface: apply(state, flags, seed) -> State.
Subclasses implement specific resampling strategies (e.g., MultinomialResampler).
"""

import abc
import tensorflow as tf
from dpf.base import State, Module


class ResamplerBase(Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for all resamplers.

    Subclasses must implement apply(), which takes a State and resampling
    flags (one per batch element), and returns a new State with resampled
    particles and reset weights.
    """

    @abc.abstractmethod
    def apply(self, state: State, flags: tf.Tensor,
              seed=None) -> State:
        """
        Apply resampling to the particle filter state.

        Args:
            state: Current particle filter state.
            flags: Boolean tensor, shape [batch_size].
                True = resample this batch element; False = keep unchanged.
            seed: Optional random seed.

        Returns:
            New State with resampled particles and uniform weights
            (for batch elements where flags=True), or unchanged state
            (where flags=False).
        """
        raise NotImplementedError
