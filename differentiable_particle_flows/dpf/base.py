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

@attr.s(frozen=True)
class StateSeries:
    """
    Immutable accumulator for State objects across time steps.

    Uses tf.TensorArray internally. Immutable like State — write() returns
    a NEW StateSeries rather than mutating self. This follows the functional
    pattern that tf.TensorArray itself uses (ta.write() returns a new ta)
    and is safe for @tf.function tracing.

    Used by SMC.__call__ when return_series=True. Collects particles,
    log_weights, and log_likelihoods at each time step t = 0, ..., T-1,
    then stacks them into tensors with a leading time dimension.

    Attributes:
        _particles_ta: TensorArray for particles across time.
        _log_weights_ta: TensorArray for log_weights across time.
        _log_likelihoods_ta: TensorArray for log_likelihoods across time.
    """
    _particles_ta = attr.ib()
    _log_weights_ta = attr.ib()
    _log_likelihoods_ta = attr.ib()

    @classmethod
    def create(cls, max_time_steps: int, dtype=DEFAULT_DTYPE) -> 'StateSeries':
        """Create an empty StateSeries with pre-allocated TensorArrays.

        Args:
            max_time_steps: Maximum number of time steps (T).
            dtype: TensorFlow dtype for arrays.

        Returns:
            New empty StateSeries.
        """
        return cls(
            particles_ta=tf.TensorArray(dtype=dtype, size=max_time_steps,
                                        dynamic_size=False),
            log_weights_ta=tf.TensorArray(dtype=dtype, size=max_time_steps,
                                          dynamic_size=False),
            log_likelihoods_ta=tf.TensorArray(dtype=dtype, size=max_time_steps,
                                              dynamic_size=False),
        )

    def write(self, t: int, state: State) -> 'StateSeries':
        """Write a state at time step t.

        Returns a NEW StateSeries (immutable — does not modify self).
        This mirrors tf.TensorArray.write() which also returns a new array.

        Args:
            t: Time step index.
            state: State to record.

        Returns:
            New StateSeries with the state written at position t.
        """
        return attr.evolve(
            self,
            particles_ta=self._particles_ta.write(t, state.particles),
            log_weights_ta=self._log_weights_ta.write(t, state.log_weights),
            log_likelihoods_ta=self._log_likelihoods_ta.write(
                t, state.log_likelihoods
            ),
        )

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
