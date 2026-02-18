"""
Multinomial resampling via CDF inversion.

Problem — Weight Degeneracy
---------------------------
After several filtering steps, the importance weights degenerate: a few
particles carry nearly all the weight while the rest are negligible.
Quantified by the effective sample size (ESS):

    ESS = 1 / sum_{i=1}^{N} (w_t^i)^2

where w_t^i are the normalized weights. When ESS is low (e.g. ESS < 0.5*N),
the particle approximation is poor and resampling is triggered.

Multinomial Resampling — Mathematical Formulation
--------------------------------------------------
Given N particles {x_t^i}_{i=1}^{N} with normalized weights {w_t^i}_{i=1}^{N}
where sum_i w_t^i = 1, multinomial resampling draws N new ancestor indices:

    a_t^i ~ Categorical(w_t^1, w_t^2, ..., w_t^N),    i = 1, ..., N

Each index a_t^i is drawn independently with:

    P(a_t^i = k) = w_t^k

The resampled particles are then:

    x_t^i <-- x_t^{a_t^i}

and their weights are reset to uniform:

    w_t^i = 1/N

This is equivalent to sampling with replacement from the empirical
distribution defined by the weighted particles.

CDF Inversion Implementation
-----------------------------
Rather than sampling from Categorical directly, we use CDF inversion
which is more efficient and vectorizable:

    1. Compute the CDF:  C_k = sum_{j=1}^{k} w_t^j,   k = 1, ..., N

    2. Draw N iid uniform samples and sort them:
       u_1, ..., u_N  ~iid~  Uniform(0, 1)
       u_(1) <= u_(2) <= ... <= u_(N)    (order statistics)

    3. Invert the CDF to find indices:
       a^i = min{ k : C_k >= u_(i) }    (via tf.searchsorted)

    4. Gather resampled particles:
       x_new^i = x^{a^i}                (via tf.gather)


Conditional Resampling
----------------------
Resampling is applied per batch element using boolean flags from the
resampling criterion (NeffCriterion). For batch element b:
- If flags[b] = True:  resample particles, reset weights to 1/N
- If flags[b] = False: keep particles and weights unchanged

"""

import tensorflow as tf
from dpf.base import State
from dpf.resampling.resampling_base import CdfInversionResamplerBase
from dpf.constants import DEFAULT_SEED


class MultinomialResampler(CdfInversionResamplerBase):
    """
    Multinomial resampler using sorted uniform spacings and CDF inversion.

    Uses tf.random.Generator for reproducible random number generation.
    Each call to apply() draws from the generator, advancing its internal
    state. Two resamplers created with the same seed will produce the same
    sequence of resampled particles.

    Args:
        seed: Random seed for reproducibility. Default: DEFAULT_SEED from constants.

    Usage:
        resampler = MultinomialResampler(seed=42)
        flags = criterion.apply(state)        # [batch], True where ESS is low
        new_state = resampler.apply(state, flags)
    """

    def __init__(self, seed: int = DEFAULT_SEED, name: str = 'MultinomialResampler'):
        super().__init__(seed=seed, name=name)

    def _get_cdf_weights(self, state: State) -> tf.Tensor:
        """Use the original normalized weights for CDF construction."""
        return state.weights

    def _compute_new_log_weights(self, state: State, indices: tf.Tensor,
                                 n_particles: int,
                                 flags_2d: tf.Tensor) -> tf.Tensor:
        """Reset weights to uniform -log(N) for resampled batch elements."""
        n_f = tf.cast(n_particles, tf.float64)
        uniform_log_weights = tf.fill(
            tf.shape(state.log_weights), -tf.math.log(n_f)
        )
        return tf.where(flags_2d, uniform_log_weights, state.log_weights)
