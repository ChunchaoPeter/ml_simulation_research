"""
Core data structures for the standard particle filter.

Provides:
- State: Immutable dataclass holding particle filter state (particles, weights, etc.)
- StateSeries: TensorArray-based accumulator for states across time steps.
- Module: Base class extending tf.Module for all framework components.

Design follows the filterflow pattern:
- State is immutable — all updates create new instances via attr.evolve().
- Validators enforce tensor shape invariants at construction time.
"""

import tensorflow as tf
import attr

from dpf.constants import DEFAULT_DTYPE


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _dim_3_validator(instance, attribute, value):
    """Validate that tensor has exactly 3 dimensions: [batch, n_particles, state_dim]."""
    if value is not None and len(value.shape) != 3:
        raise ValueError(
            f"{attribute.name} must be 3D [batch, n_particles, state_dim], "
            f"got shape {value.shape}"
        )


def _dim_2_validator(instance, attribute, value):
    """Validate that tensor has exactly 2 dimensions: [batch, n_particles]."""
    if value is not None and len(value.shape) != 2:
        raise ValueError(
            f"{attribute.name} must be 2D [batch, n_particles], "
            f"got shape {value.shape}"
        )


def _dim_1_validator(instance, attribute, value):
    """Validate that tensor has exactly 1 dimension: [batch]."""
    if value is not None and len(value.shape) != 1:
        raise ValueError(
            f"{attribute.name} must be 1D [batch], got shape {value.shape}"
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@attr.s(frozen=True)
class State:
    """
    Immutable particle filter state at a single time step.

    All fields are TensorFlow tensors. The leading dimension is always the
    batch dimension, allowing vectorized computation over multiple independent
    particle filter runs.

    Attributes:
        particles: Particle positions X_t^i, shape [batch_size, n_particles, state_dim].
        log_weights: Log importance weights log(omega_t^i), shape [batch_size, n_particles].
        weights: Normalized weights w_t^i (derived from log_weights via softmax),
            shape [batch_size, n_particles].
        log_likelihoods: Accumulated log marginal likelihood ell_hat(theta),
            shape [batch_size]. (Algorithm 1, step 10.)
        ancestor_indices: Resampling ancestry (which parent each particle came from),
            shape [batch_size, n_particles]. (Algorithm 1, step 6.)
        t: Current time step, scalar int.
        ess: Effective sample size 1/sum(w_i^2), shape [batch_size].
    """
    particles = attr.ib(validator=_dim_3_validator)
    log_weights = attr.ib(validator=_dim_2_validator)
    weights = attr.ib(default=None, validator=attr.validators.optional(_dim_2_validator))
    log_likelihoods = attr.ib(default=None, validator=attr.validators.optional(_dim_1_validator))
    ancestor_indices = attr.ib(default=None)
    t = attr.ib(default=None)
    ess = attr.ib(default=None)

    def __attrs_post_init__(self):
        """Initialize derived fields if not provided."""
        if self.weights is None:
            object.__setattr__(
                self, 'weights', tf.nn.softmax(self.log_weights, axis=-1)
            )
        if self.log_likelihoods is None:
            batch_size = self.particles.shape[0]
            object.__setattr__(
                self, 'log_likelihoods',
                tf.zeros([batch_size], dtype=self.particles.dtype)
            )
        if self.ess is None:
            object.__setattr__(self, 'ess', self._compute_ess())

    def _compute_ess(self) -> tf.Tensor:
        """Compute effective sample size: ESS = 1 / sum(w_i^2)."""
        return 1.0 / tf.reduce_sum(self.weights ** 2, axis=-1)

    @property
    def batch_size(self) -> int:
        return self.particles.shape[0]

    @property
    def n_particles(self) -> int:
        return self.particles.shape[1]

    @property
    def state_dim(self) -> int:
        return self.particles.shape[2]


# ---------------------------------------------------------------------------
# StateSeries
# ---------------------------------------------------------------------------

class StateSeries:
    """
    Accumulates State objects across time steps using tf.TensorArray.

    Used by SMC.__call__ when return_series=True. Collects particles,
    log_weights, and log_likelihoods at each time step t = 0, ..., T-1,
    then stacks them into tensors with a leading time dimension.

    Attributes:
        _particles_ta: TensorArray for particles across time.
        _log_weights_ta: TensorArray for log_weights across time.
        _log_likelihoods_ta: TensorArray for log_likelihoods across time.
    """

    def __init__(self, max_time_steps: int, dtype=DEFAULT_DTYPE):
        """
        Args:
            max_time_steps: Maximum number of time steps (T).
            dtype: TensorFlow dtype for arrays.
        """
        self._max_T = max_time_steps
        self._particles_ta = tf.TensorArray(dtype=dtype, size=max_time_steps,
                                             dynamic_size=False)
        self._log_weights_ta = tf.TensorArray(dtype=dtype, size=max_time_steps,
                                               dynamic_size=False)
        self._log_likelihoods_ta = tf.TensorArray(dtype=dtype, size=max_time_steps,
                                                    dynamic_size=False)

    def write(self, t: int, state: State) -> 'StateSeries':
        """Write a state at time step t. Returns self for chaining."""
        self._particles_ta = self._particles_ta.write(t, state.particles)
        self._log_weights_ta = self._log_weights_ta.write(t, state.log_weights)
        self._log_likelihoods_ta = self._log_likelihoods_ta.write(
            t, state.log_likelihoods
        )
        return self

    def stack(self):
        """Stack all time steps into tensors.

        Returns:
            Dict with keys 'particles', 'log_weights', 'log_likelihoods',
            each with a leading time dimension.
        """
        return {
            'particles': self._particles_ta.stack(),
            'log_weights': self._log_weights_ta.stack(),
            'log_likelihoods': self._log_likelihoods_ta.stack(),
        }


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class Module(tf.Module):
    """
    Base class for all framework components.

    Extends tf.Module to provide automatic variable tracking and
    a consistent interface for all components.
    """

    def __init__(self, name: str = None):
        super().__init__(name=name)
