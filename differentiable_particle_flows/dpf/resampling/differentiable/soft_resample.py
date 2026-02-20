"""Soft resampling (Karkus et al., 2018): differentiable resampling via
mixture with uniform.

Soft weights:      w_soft^i = alpha * w^i + (1 - alpha) / N
Corrected weights: log w_new^i = log w^{a^i} - log w_soft^{a^i}

Gradients flow through the corrected weights (continuous function of w).
The index selection (searchsorted) remains non-differentiable.

Reference: Karkus, P., Hsu, D., & Lee, W. S. (2018).
Particle Filter Networks with Application to Visual Localization. CoRL.
"""

import tensorflow as tf
from dpf.base import State
from dpf.resampling.resampling_base import CdfInversionResamplerBase
from dpf.constants import DEFAULT_SEED


class SoftResampler(CdfInversionResamplerBase):
    """Soft resampler via mixture with uniform for differentiable resampling.

    Args:
        alpha: Mixing coefficient in (0, 1]. Default: 0.5.
        seed: Random seed. Default: DEFAULT_SEED.
    """

    def __init__(self, alpha: float = 0.5, seed: int = DEFAULT_SEED,
                 name: str = 'SoftResampler'):
        super().__init__(seed=seed, name=name)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def _get_cdf_weights(self, state: State) -> tf.Tensor:
        """Compute soft weights w_soft = alpha * w + (1-alpha)/N in log space.

        Also stashes corrected log weights (log w - log w_soft) for
        _compute_new_log_weights.
        """
        log_weights = state.log_weights                      # log W^i
        n_particles = state.n_particles
        n_f = tf.cast(n_particles, tf.float64)
        log_alpha = tf.math.log(tf.cast(self._alpha, tf.float64))
        uniform_log_weights = tf.fill(                       # log(1/N)
            tf.shape(log_weights), -tf.math.log(n_f)
        )

        # log w_soft^i = log(alpha * W^i + (1-alpha)/N)
        # Computed via logsumexp for numerical stability:
        #   logsumexp(log(alpha) + log W^i, log(1-alpha) + log(1/N))
        if self._alpha < 1.0:
            log_one_minus_alpha = tf.math.log(
                tf.cast(1.0 - self._alpha, tf.float64)
            )
            q_log_weights = tf.reduce_logsumexp(
                tf.stack([
                    log_weights + log_alpha,
                    uniform_log_weights + log_one_minus_alpha,
                ], axis=-1),
                axis=-1,
            )
            # Normalize: log w_soft^i -= log(sum_j w_soft^j)
            q_log_weights = q_log_weights - tf.reduce_logsumexp(
                q_log_weights, axis=-1, keepdims=True
            )
            # log w_new^i = log W^i - log w_soft^i
            self._corrected_log_weights = log_weights - q_log_weights
            return tf.exp(self._corrected_log_weights)
        else:
            # alpha=1: w_soft = w, corrected weights cancel to uniform
            q_log_weights = log_weights
            return tf.exp(q_log_weights)


    def _compute_new_log_weights(self, state: State, indices: tf.Tensor,
                                 n_particles: int,
                                 flags_2d: tf.Tensor) -> tf.Tensor:
        """Gather corrected log weights at ancestor indices."""
        # Gather Eq. 48 corrected weights at ancestor indices a^i.
        # Indices are non-differentiable constants; gradients flow through
        # the gathered weight values (w/w_soft depends continuously on w).
        resampled_log_weights = tf.gather(
            self._corrected_log_weights, indices, batch_dims=1
        )
        return tf.where(flags_2d, resampled_log_weights, state.log_weights)
