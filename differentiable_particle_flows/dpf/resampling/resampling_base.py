"""
CDF inversion resampling base class (Template Method pattern).

Extracts the shared CDF inversion pipeline used by non-differentiable
resamplers (e.g., MultinomialResampler). Subclasses only need to implement
two hooks:

    1. _get_cdf_weights(state) -> tf.Tensor
       Which weights to use for building the CDF.

    2. _compute_new_log_weights(state, indices, n_particles, flags_2d) -> tf.Tensor
       How to compute post-resampling log weights.

The template apply() method handles: CDF cumsum, sorted uniform spacings,
searchsorted, gather with stop_gradient, conditional flags for particles
and indices, weight computation via hook, softmax, ESS, and attr.evolve.

Note: This base class wraps particle gathering in stop_gradient, making it
unsuitable for differentiable resamplers (e.g., SoftResampler) which need
gradient flow through corrected weights.
"""

import abc
import attr
import tensorflow as tf
from dpf.base import State
from dpf.resampling.base import ResamplerBase
from dpf.constants import DEFAULT_SEED


class CdfInversionResamplerBase(ResamplerBase, abc.ABC):
    """
    Abstract base class for CDF-inversion-based resamplers.

    Implements the full CDF inversion pipeline as a Template Method,
    delegating two decisions to subclasses:
    - Which weights build the CDF (_get_cdf_weights)
    - How post-resampling weights are computed (_compute_new_log_weights)

    This base class is designed for non-differentiable resamplers. It wraps
    particle gathering in stop_gradient. Differentiable resamplers (e.g.,
    SoftResampler) should inherit from ResamplerBase directly.

    Args:
        seed: Random seed for reproducibility. Default: DEFAULT_SEED.
        name: Module name.
    """

    def __init__(self, seed: int = DEFAULT_SEED, name: str = 'CdfInversionResampler'):
        super().__init__(name=name)
        self._seed = seed
        self._rng = tf.random.Generator.from_seed(seed)

    @abc.abstractmethod
    def _get_cdf_weights(self, state: State) -> tf.Tensor:
        """Return normalized weights for CDF construction, shape [batch, N].

        These weights must be non-negative and sum to 1 along axis=-1.

        Args:
            state: Current particle filter state.

        Returns:
            Tensor of shape [batch_size, n_particles] used to build the CDF.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_new_log_weights(self, state: State, indices: tf.Tensor,
                                 n_particles: int,
                                 flags_2d: tf.Tensor) -> tf.Tensor:
        """Compute post-resampling log weights, shape [batch, N].

        Args:
            state: Original particle filter state (before resampling).
            indices: Ancestor indices from CDF inversion, shape [batch, N].
            n_particles: Number of particles N.
            flags_2d: Boolean flags expanded to [batch, 1] for broadcasting.

        Returns:
            New log weights tensor of shape [batch_size, n_particles].
        """
        raise NotImplementedError

    def apply(self, state: State, flags: tf.Tensor,
              seed=DEFAULT_SEED) -> State:
        """
        Resample particles via CDF inversion (Template Method).

        Steps:
            1. Get CDF weights from subclass hook.
            2. Build CDF via cumsum.
            3. Generate sorted uniform spacings.
            4. Invert CDF to find ancestor indices (searchsorted).
            5. Gather particles at those indices (with stop_gradient).
            6. Conditionally apply per batch element based on flags.
            7. Compute new log weights from subclass hook.
            8. Compute softmax weights, ESS, and return new State.

        Args:
            state: Current particle filter state.
            flags: Boolean resampling flags, shape [batch_size].
                True = resample this batch element; False = keep unchanged.
            seed: Unused, kept for interface compatibility.

        Returns:
            New State with resampled particles and updated weights.
        """
        batch_size = state.batch_size
        n_particles = state.n_particles

        # Step 1: Get weights for CDF construction (subclass hook)
        # Retrieve the normalized weights w_t^i (already summing to 1 via
        # softmax in State). Then build the cumulative distribution function:
        #   C_k = sum_{j=1}^{k} w_t^j,   k = 1, ..., N
        # so C_N = 1. The CDF maps each particle index k to the probability
        # mass accumulated up to and including particle k.
        cdf_weights = self._get_cdf_weights(state)          # [batch, N]

        # Step 2: Build CDF
        cdf = tf.cumsum(cdf_weights, axis=-1)                # [batch, N]

        # Step 3: Sorted uniform spacings (multinomial via CDF inversion)
        # Draw N iid samples from Uniform(0,1) and sort them:
        #   u_1, ..., u_N  ~iid~  Uniform(0, 1)
        #   u_(1) <= u_(2) <= ... <= u_(N)   (order statistics)
        # Sorting before searchsorted lets us scan the CDF left-to-right.

        # Uses tf.random.Generator for reproducible, stateful RNG.
        u = self._rng.uniform(
            [batch_size, n_particles], dtype=tf.float64
        )
        spacings = tf.sort(u, axis=-1)                       # [batch, N]

        # Step 4: CDF inversion — find ancestor indices
        # For each sorted uniform u_(i), find the smallest index k such that
        # the CDF exceeds u_(i):
        #   a^i = min{ k : C_k >= u_(i) }
        # tf.searchsorted performs binary search on each row of the CDF.
        indices = tf.searchsorted(cdf, spacings)             # [batch, N]
        indices = tf.minimum(indices, n_particles - 1)

        # Step 5: Gather resampled particles (non-differentiable)
        # Use the ancestor indices to select particles:
        #   x_new^i = x^{a^i}
        # batch_dims=1 means: for each batch element b, gather from
        # state.particles[b] using indices[b], producing resampled_particles[b].
        # tf.stop_gradient blocks gradient flow because the index selection is
        # a discrete, non-differentiable operation. 
        resampled_particles = tf.stop_gradient(
            tf.gather(state.particles, indices, batch_dims=1)
        )

        # Step 6: Conditionally apply resampling per batch element
        # The resampling criterion (NeffCriterion) produces a boolean flag per
        # batch element. We only replace particles where flags[b] = True.
        # flags is [batch], so we expand it to [batch, 1, 1] to broadcast
        # against particles [batch, N, d]. tf.where selects element-wise:
        #   new_particles[b] = resampled_particles[b]  if flags[b] = True
        #                      state.particles[b]      if flags[b] = False
        flags_expanded = flags[:, tf.newaxis, tf.newaxis]    # [batch, 1, 1]
        new_particles = tf.where(flags_expanded, resampled_particles,
                                 state.particles)

        # Conditionally apply ancestor indices per batch element.
        # When flags[b] = False, each particle is its own ancestor: identity
        # [0, 1, ..., N-1].
        identity_indices = tf.broadcast_to(
            tf.range(n_particles)[tf.newaxis, :], [batch_size, n_particles]
        )
        flags_2d = flags[:, tf.newaxis]                      # [batch, 1]
        new_indices = tf.where(flags_2d, indices, identity_indices)

        # Step 7: Compute new log weights (subclass hook)
        new_log_weights = self._compute_new_log_weights(
            state, n_particles, flags_2d
        )

        # Step 8: Compute softmax weights, ESS, and return new State
        new_weights = tf.nn.softmax(new_log_weights, axis=-1)

        return attr.evolve(
            state,
            particles=new_particles,
            log_weights=new_log_weights,
            weights=new_weights,
            ancestor_indices=new_indices,
            ess=1.0 / tf.reduce_sum(new_weights ** 2, axis=-1),
        )
