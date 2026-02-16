"""Tests for dpf/base.py: State, StateSeries, validators.

Tests the core data structures used throughout the particle filter:

    State: Immutable (frozen attrs) dataclass holding the particle cloud at
        one time step. Fields:
        - particles:       [batch, n_particles, state_dim]   (3D)
        - log_weights:     [batch, n_particles]              (2D)
        - weights:         [batch, n_particles]              (derived via softmax)
        - log_likelihoods: [batch]                           (1D, accumulated)
        - ess:             [batch]                           (derived: 1/sum(w^2))
        - ancestor_indices, t: optional metadata

    StateSeries: Immutable TensorArray-based accumulator that records State
        at each time step t=0..T-1, then stacks into time-indexed tensors.

Test classes:
    TestDim3Validator / TestDim2Validator / TestDim1Validator
        - Unit tests for shape validators in isolation (accept valid, reject
          wrong rank, allow None).

    TestValidators
        - Integration tests: validators fire correctly during State
          construction (wrong-rank particles, log_weights, weights,
          log_likelihoods all raise ValueError).

    TestStateCreation
        - Shapes, derived fields (weights, log_likelihoods, ess).
        - Properties: batch_size, n_particles, state_dim.
        - Weights sum to 1, ESS=N for uniform, ESS~1 for degenerate.
        - Log-likelihoods initialize to 0.
        - Optional fields (ancestor_indices, t) via attr.evolve.

    TestStateImmutability
        - Frozen: direct assignment raises FrozenInstanceError.
        - attr.evolve creates a new State without mutating the original.

    test_state_fields_in_tf_function
        - State tensors work inside @tf.function (graph-mode compatibility).

    TestStateSeries
        - write/stack round-trip: correct shapes [T, batch, N, D].
        - Distinct values at different time steps.
        - Immutability: frozen attrs.
"""

import pytest
import tensorflow as tf
import numpy as np
import attr
from dpf.base import State, StateSeries, _dim_1_validator, _dim_2_validator, _dim_3_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyAttr:
    """Minimal mock of attrs Attribute for validator unit tests."""
    def __init__(self, name: str):
        self.name = name


BATCH = 2
N = 10
D = 2


def _make_state(batch=BATCH, n=N, d=D, uniform=True):
    """Helper to create a State with given dimensions."""
    particles = tf.constant(
        np.random.randn(batch, n, d), dtype=tf.float64
    )
    if uniform:
        log_weights = tf.fill([batch, n], -tf.math.log(tf.cast(n, tf.float64)))
    else:
        # Degenerate: all weight on particle 0
        lw = np.full((batch, n), -1e10)
        lw[:, 0] = 0.0
        log_weights = tf.constant(lw, dtype=tf.float64)
    return State(particles=particles, log_weights=log_weights)


# ---------------------------------------------------------------------------
# Validator unit tests (test each validator function in isolation)
# ---------------------------------------------------------------------------

class TestDim3Validator:
    def test_accepts_valid_3d_tensor(self):
        x = tf.zeros([2, 10, 4])
        _dim_3_validator(None, DummyAttr("particles"), x)

    def test_rejects_2d_tensor(self):
        x = tf.zeros([2, 10])
        with pytest.raises(ValueError, match="must be 3D"):
            _dim_3_validator(None, DummyAttr("particles"), x)

    def test_allows_none(self):
        _dim_3_validator(None, DummyAttr("particles"), None)


class TestDim2Validator:
    def test_accepts_valid_2d_tensor(self):
        x = tf.zeros([3, 5])
        _dim_2_validator(None, DummyAttr("log_weights"), x)

    def test_rejects_3d_tensor(self):
        x = tf.zeros([3, 5, 1])
        with pytest.raises(ValueError, match="must be 2D"):
            _dim_2_validator(None, DummyAttr("log_weights"), x)

    def test_allows_none(self):
        _dim_2_validator(None, DummyAttr("log_weights"), None)


class TestDim1Validator:
    def test_accepts_valid_1d_tensor(self):
        x = tf.zeros([4])
        _dim_1_validator(None, DummyAttr("log_likelihoods"), x)

    def test_rejects_2d_tensor(self):
        x = tf.zeros([4, 1])
        with pytest.raises(ValueError, match="must be 1D"):
            _dim_1_validator(None, DummyAttr("log_likelihoods"), x)

    def test_allows_none(self):
        _dim_1_validator(None, DummyAttr("log_likelihoods"), None)


# ---------------------------------------------------------------------------
# Validator integration tests (validators fire through State construction)
# ---------------------------------------------------------------------------

class TestValidators:
    def test_particles_wrong_dim_raises(self):
        bad_particles = tf.zeros([BATCH, N], dtype=tf.float64)  # 2D, needs 3D
        log_weights = tf.zeros([BATCH, N], dtype=tf.float64)
        with pytest.raises(ValueError, match="3D"):
            State(particles=bad_particles, log_weights=log_weights)

    def test_log_weights_wrong_dim_raises(self):
        particles = tf.zeros([BATCH, N, D], dtype=tf.float64)
        bad_log_weights = tf.zeros([BATCH, N, 1], dtype=tf.float64)  # 3D, needs 2D
        with pytest.raises(ValueError, match="2D"):
            State(particles=particles, log_weights=bad_log_weights)

    def test_weights_wrong_dim_raises(self):
        particles = tf.zeros([BATCH, N, D], dtype=tf.float64)
        log_weights = tf.zeros([BATCH, N], dtype=tf.float64)
        bad_weights = tf.zeros([BATCH, N, 1], dtype=tf.float64)  # 3D, needs 2D
        with pytest.raises(ValueError, match="2D"):
            State(particles=particles, log_weights=log_weights, weights=bad_weights)

    def test_log_likelihoods_wrong_dim_raises(self):
        particles = tf.zeros([BATCH, N, D], dtype=tf.float64)
        log_weights = tf.zeros([BATCH, N], dtype=tf.float64)
        weights = tf.zeros([BATCH, N], dtype=tf.float64) 
        log_likelihoods = tf.zeros([BATCH, N], dtype=tf.float64)  # 2D, needs 1D
        with pytest.raises(ValueError, match="1D"):
            State(particles=particles, log_weights=log_weights, weights=weights, log_likelihoods=log_likelihoods)

    def test_log_likelihoods_scalar_raises(self):
        particles = tf.zeros([BATCH, N, D], dtype=tf.float64)
        log_weights = tf.zeros([BATCH, N], dtype=tf.float64)
        weights = tf.zeros([BATCH, N], dtype=tf.float64) 
        log_likelihoods = tf.constant(1)  # scalar, needs 1D
        with pytest.raises(ValueError, match="1D"):
            State(particles=particles, log_weights=log_weights, weights=weights, log_likelihoods=log_likelihoods)

# ---------------------------------------------------------------------------
# State creation and derived fields
# ---------------------------------------------------------------------------

class TestStateCreation:
    def test_state_creation(self):
        state = _make_state()
        assert state.particles.shape == (BATCH, N, D)
        assert state.log_weights.shape == (BATCH, N)
        assert state.weights is not None
        assert state.log_likelihoods is not None
        assert state.ess is not None

    def test_state_properties(self):
        state = _make_state(batch=3, n=20, d=4)
        assert state.batch_size == 3
        assert state.n_particles == 20
        assert state.state_dim == 4

    def test_state_weights_sum_to_one(self):
        state = _make_state()
        sums = tf.reduce_sum(state.weights, axis=-1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_state_uniform_weights_ess(self):
        state = _make_state(uniform=True)
        # Uniform weights → ESS should equal N
        np.testing.assert_allclose(state.ess.numpy(), N, atol=0.1)

    def test_state_degenerate_weights_ess(self):
        state = _make_state(uniform=False)
        # All weight on one particle → ESS ≈ 1
        np.testing.assert_allclose(state.ess.numpy(), 1.0, atol=0.1)

    def test_state_log_likelihoods_init_zero(self):
        state = _make_state()
        np.testing.assert_allclose(state.log_likelihoods.numpy(), 0.0)

    def test_state_with_ancestor_indices(self):
        """State can be created with optional ancestor_indices."""
        state = _make_state()
        indices = tf.zeros([BATCH, N], dtype=tf.int32)
        new_state = attr.evolve(state, ancestor_indices=indices)
        assert new_state.ancestor_indices.shape == (BATCH, N)

    def test_state_with_time_step(self):
        """State can carry a time step field."""
        state = _make_state()
        new_state = attr.evolve(state, t=5)
        assert new_state.t == 5


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestStateImmutability:
    def test_state_frozen(self):
        state = _make_state()
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            state.particles = tf.zeros_like(state.particles)

    def test_state_evolve(self):
        state = _make_state()
        new_particles = tf.ones_like(state.particles)
        new_state = attr.evolve(state, particles=new_particles)
        # New state has updated particles
        np.testing.assert_allclose(new_state.particles.numpy(), 1.0)
        # Old state is unchanged
        assert not np.allclose(state.particles.numpy(), 1.0)

    def test_state_evolve_preserves_other_fields(self):
        """attr.evolve only changes the specified field; others are copied."""
        state = _make_state()
        new_state = attr.evolve(state, t=7)
        np.testing.assert_allclose(
            new_state.particles.numpy(), state.particles.numpy()
        )
        np.testing.assert_allclose(
            new_state.log_weights.numpy(), state.log_weights.numpy()
        )


# ---------------------------------------------------------------------------
# tf.function compatibility
# (cf. filterflow test_base.py: verify State tensors work in graph mode)
# ---------------------------------------------------------------------------

def test_state_fields_in_tf_function():
    """State tensor fields can be used inside @tf.function."""
    @tf.function
    def compute_weighted_mean(particles, weights):
        return tf.reduce_sum(
            particles * weights[:, :, tf.newaxis], axis=1
        )
    state = _make_state()
    result = compute_weighted_mean(state.particles, state.weights)
    assert result.shape == (BATCH, D)


# ---------------------------------------------------------------------------
# StateSeries
# ---------------------------------------------------------------------------

class TestStateSeries:
    def test_write_and_stack(self):
        T = 3
        series = StateSeries.create(T)
        for t in range(T):
            state = _make_state()
            series = series.write(t, state)
        result = series.stack()

        assert 'particles' in result
        assert 'log_weights' in result
        assert 'log_likelihoods' in result
        assert result['particles'].shape == (T, BATCH, N, D)
        assert result['log_weights'].shape == (T, BATCH, N)
        assert result['log_likelihoods'].shape == (T, BATCH)

    def test_stack_returns_all_keys(self):
        """stack() must return exactly the three expected keys."""
        T = 2
        series = StateSeries.create(T)
        for t in range(T):
            series = series.write(t, _make_state())
        result = series.stack()
        assert set(result.keys()) == {'particles', 'log_weights', 'log_likelihoods'}

    def test_different_values_at_different_times(self):
        """Each time step should store distinct particle/weight values."""
        T = 3
        series = StateSeries.create(T)
        # Create states with deterministic, distinguishable values
        states = []
        for t in range(T):
            particles = tf.fill([BATCH, N, D], tf.constant(float(t), dtype=tf.float64))
            log_weights = tf.fill([BATCH, N], tf.constant(-float(t + 1), dtype=tf.float64))
            s = State(particles=particles, log_weights=log_weights)
            states.append(s)
            series = series.write(t, s)

        result = series.stack()
        # Verify each time step has its own distinct values
        for t in range(T):
            np.testing.assert_allclose(
                result['particles'][t].numpy(), float(t)
            )
            np.testing.assert_allclose(
                result['log_weights'][t].numpy(), -float(t + 1)
            )
        # Verify time steps differ from each other
        assert not np.allclose(
            result['particles'][0].numpy(), result['particles'][1].numpy()
        )
        assert not np.allclose(
            result['particles'][1].numpy(), result['particles'][2].numpy()
        )

    def test_frozen(self):
        """StateSeries should be immutable like State."""
        series = StateSeries.create(2)
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            series._particles_ta = None
