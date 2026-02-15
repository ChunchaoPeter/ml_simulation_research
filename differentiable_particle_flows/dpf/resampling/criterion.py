"""
Resampling criterion: decide WHEN to resample.

Separated from the resampling METHOD (how to resample) following the
single responsibility principle. The criterion evaluates ESS and returns
a per-batch boolean flag.

"""

import abc
import tensorflow as tf
from dpf.base import State, Module
from dpf.constants import DEFAULT_ESS_THRESHOLD


class ResamplingCriterionBase(Module, metaclass=abc.ABCMeta):
    """
    Abstract base for resampling criteria.

    Decides per batch element whether resampling should occur.
    """

    @abc.abstractmethod
    def apply(self, state: State) -> tf.Tensor:
        """
        Evaluate the resampling criterion.

        Args:
            state: Current particle filter state.

        Returns:
            Boolean tensor, shape [batch_size].
            True = resample; False = do not resample.
        """
        raise NotImplementedError


class NeffCriterion(ResamplingCriterionBase):
    """
    Resample when effective sample size (ESS) drops below a threshold.

    ESS_threshold = threshold_ratio * N_particles.
    Resample when ESS < ESS_threshold.

    This is the most common criterion in practice. It avoids unnecessary
    resampling (which introduces extra variance) while preventing weight
    degeneracy.

    Attributes:
        threshold_ratio: Fraction of N (default 0.5).
    
    Return: Boolean tensor, shape [batch_size]. True = resample; False = do not resample.
    For example batch_size=2, flags=[False, True] means resample the second batch element but not the first.
            tf.Tensor: shape=(2,), dtype=bool, numpy=array([ False,  True])> 
    """

    def __init__(self, threshold_ratio: float = DEFAULT_ESS_THRESHOLD,
                 name: str = 'NeffCriterion'):
        super().__init__(name=name)
        self.threshold_ratio = threshold_ratio

    def apply(self, state: State) -> tf.Tensor:
        """Resample where ESS < threshold * N.

        Returns:
            Boolean tensor, shape [batch_size].
        """
        n_particles = tf.cast(state.n_particles, tf.float64)
        threshold = self.threshold_ratio * n_particles
        return state.ess < threshold
