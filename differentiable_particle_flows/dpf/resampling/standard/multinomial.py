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

import attr
import tensorflow as tf
from dpf.base import State
from dpf.resampling.base import ResamplerBase
from dpf.constants import DEFAULT_SEED


class MultinomialResampler(ResamplerBase):
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
        super().__init__(name=name)
        self._seed = seed
        self._rng = tf.random.Generator.from_seed(seed)

    def apply(self, state: State, flags: tf.Tensor,
              seed=DEFAULT_SEED) -> State:
        """
        Resample particles via multinomial CDF inversion.

        Steps:
            1. Build CDF from normalized weights.
            2. Generate sorted uniform spacings (multinomial sampling).
            3. Invert CDF to find resampled indices (searchsorted).
            4. Gather particles at those indices (with stop_gradient).
            5. Conditionally apply per batch element based on flags.
            6. Reset weights to uniform for resampled elements.

        Args:
            state: Current particle filter state.
            flags: Boolean resampling flags, shape [batch_size].
                True = resample this batch element; False = keep unchanged.

        Returns:
            New State with resampled particles and uniform weights
            (for flagged batch elements).
        """
        batch_size = state.batch_size
        n_particles = state.n_particles

        # Step 1: Normalized weights and CDF
        # Retrieve the normalized weights w_t^i (already summing to 1 via
        # softmax in State). Then build the cumulative distribution function:
        #   C_k = sum_{j=1}^{k} w_t^j,   k = 1, ..., N
        # so C_N = 1. The CDF maps each particle index k to the probability
        # mass accumulated up to and including particle k.
        weights = state.weights                           # [batch, N]
        cdf = tf.cumsum(weights, axis=-1)                 # [batch, N]

        # Step 2: Sorted uniform spacings (multinomial via CDF inversion)
        # Draw N iid samples from Uniform(0,1) and sort them:
        #   u_1, ..., u_N  ~iid~  Uniform(0, 1)
        #   u_(1) <= u_(2) <= ... <= u_(N)   (order statistics)
        # Sorting before searchsorted lets us scan the CDF left-to-right.

        # Uses tf.random.Generator for reproducible, stateful RNG.
        u = self._rng.uniform(
            [batch_size, n_particles], dtype=tf.float64
        )
        spacings = tf.sort(u, axis=-1)                    # [batch, N]

        # Step 3: CDF inversion — find ancestor indices
        # For each sorted uniform u_(i), find the smallest index k such that
        # the CDF exceeds u_(i):
        #   a^i = min{ k : C_k >= u_(i) }
        # tf.searchsorted performs binary search on each row of the CDF.

        indices = tf.searchsorted(cdf, spacings)          # [batch, N]
        indices = tf.minimum(indices, n_particles - 1)

        # Step 4: Gather resampled particles (non-differentiable)
        # Use the ancestor indices to select particles:
        #   x_new^i = x^{a^i}
        # batch_dims=1 means: for each batch element b, gather from
        # state.particles[b] using indices[b], producing resampled_particles[b].
        # tf.stop_gradient blocks gradient flow because the index selection is
        # a discrete, non-differentiable operation. 
        resampled_particles = tf.stop_gradient(
            tf.gather(state.particles, indices, batch_dims=1)
        )

        # Step 5: Conditionally apply resampling per batch element
        # The resampling criterion (NeffCriterion) produces a boolean flag per
        # batch element. We only replace particles where flags[b] = True.
        # flags is [batch], so we expand it to [batch, 1, 1] to broadcast
        # against particles [batch, N, d]. tf.where selects element-wise:
        #   new_particles[b] = resampled_particles[b]  if flags[b] = True
        #                      state.particles[b]      if flags[b] = False
        flags_expanded = flags[:, tf.newaxis, tf.newaxis]  # [batch, 1, 1]
        new_particles = tf.where(flags_expanded, resampled_particles,
                                  state.particles)

        # Step 6: Reset weights to uniform for resampled elements
        # After resampling, each selected particle represents an equal share
        # of the posterior, so we reset to uniform weights:
        #   w_t^i = 1/N   =>   log w_t^i = -log(N)
        # For non-resampled batch elements (flags=False), we keep the original
        # log_weights unchanged. flags_2d is [batch, 1] to broadcast against
        # log_weights [batch, N].
        n_f = tf.cast(n_particles, tf.float64)
        uniform_log_weights = tf.fill(
            tf.shape(state.log_weights), -tf.math.log(n_f)
        )
        flags_2d = flags[:, tf.newaxis]
        new_log_weights = tf.where(flags_2d, uniform_log_weights,
                                    state.log_weights)
        # return a new state 
        # It creates a new instance, copying all fields from state, but replacing only the ones specified.
        return attr.evolve(
            state,
            particles=new_particles,
            log_weights=new_log_weights,
            weights=tf.nn.softmax(new_log_weights, axis=-1),
            ancestor_indices=indices,
            ess=1.0 / tf.reduce_sum(
                tf.nn.softmax(new_log_weights, axis=-1) ** 2, axis=-1
            ),
        )
