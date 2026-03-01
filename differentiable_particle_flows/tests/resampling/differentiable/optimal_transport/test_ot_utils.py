"""Tests for dpf/resampling/differentiable/regularized_optimal_transport/ot_utils.py

Tests the utility functions used by the Sinkhorn algorithm:
  - squared_distances: pairwise ||x_i - y_j||^2
  - cost:             (1/2) ||x_i - y_j||^2
  - softmin:          T_epsilon operator (Corenflos et al. 2021, Eq. 11)
  - diameter:         particle-cloud scale factor (Section 3.2)
  - max_min:          particle-cloud range for epsilon-scaling

Reference: test structure adapted from filterflow-master/tests/resampling/
           differentiable/optimal_transport/test_utils.py
"""

import tensorflow as tf
import numpy as np

from dpf.resampling.differentiable.regularized_optimal_transport.ot_utils import (
    squared_distances,
    cost,
    softmin,
    diameter,
    max_min,
)

DTYPE = tf.float64
B = 5       # batch size
N = 100     # number of particles
D = 2       # state dimension


def _random_particles(batch=B, n_particles=N, dim=D, seed=42):
    """Generate uniform random particles in [0, 1]^D."""
    np.random.seed(seed)
    return tf.constant(np.random.uniform(0., 1., [batch, n_particles, dim]), dtype=DTYPE)


# ────────────────────────────────────────────────────────────
# squared_distances
# ────────────────────────────────────────────────────────────

class TestSquaredDistances:

    def test_output_shape(self):
        x = _random_particles()
        y = _random_particles(seed=99)
        d = squared_distances(x, y)
        assert d.shape == (B, N, N)

    def test_self_distance_diagonal_is_zero(self):
        """||x_i - x_i||^2 = 0 for all i."""
        x = _random_particles()
        d = squared_distances(x, x)
        diag = tf.linalg.diag_part(d).numpy()
        np.testing.assert_allclose(diag, 0., atol=1e-12)

    def test_non_negative(self):
        """All squared distances must be >= 0."""
        x = _random_particles()
        y = _random_particles(seed=99)
        d = squared_distances(x, y)
        assert tf.reduce_all(d >= 0.).numpy()

    def test_symmetry(self):
        """||x_i - y_j||^2 = ||y_j - x_i||^2 => D(x,y) = D(y,x)^T."""
        x = _random_particles()
        y = _random_particles(seed=99)
        dxy = squared_distances(x, y)
        dyx = squared_distances(y, x)
        np.testing.assert_allclose(dxy.numpy(), tf.transpose(dyx, [0, 2, 1]).numpy(), atol=1e-10)

    def test_expected_value_uniform(self):
        """For x, y ~ U[0,1]^D independently, E[||x-y||^2] = D/6 per dimension.

        For D=2: E[||x-y||^2] = 2 * (1/6) = 1/3.
        With N=100 and B=5, the mean should be close to 1/3.
        """
        x = _random_particles()
        y = _random_particles(seed=99)
        d = squared_distances(x, y)
        np.testing.assert_allclose(tf.reduce_mean(d).numpy(), 1. / 3., atol=5e-2)

    def test_known_pair(self):
        """Check against hand-computed distance for a known pair."""
        x = tf.constant([[[1., 0.], [0., 0.]]], dtype=DTYPE)     # [1, 2, 2]
        y = tf.constant([[[0., 1.], [1., 1.]]], dtype=DTYPE)     # [1, 2, 2]
        d = squared_distances(x, y).numpy()
        # ||[1,0] - [0,1]||^2 = 2, ||[1,0] - [1,1]||^2 = 1
        # ||[0,0] - [0,1]||^2 = 1, ||[0,0] - [1,1]||^2 = 2
        np.testing.assert_allclose(d[0], [[2., 1.], [1., 2.]], atol=1e-12)


# ────────────────────────────────────────────────────────────
# cost
# ────────────────────────────────────────────────────────────

class TestCost:

    def test_output_shape(self):
        x = _random_particles()
        y = _random_particles(seed=99)
        c = cost(x, y)
        assert c.shape == (B, N, N)

    def test_is_half_squared_distance(self):
        """cost(x, y) = squared_distances(x, y) / 2."""
        x = _random_particles()
        y = _random_particles(seed=99)
        c = cost(x, y)
        d = squared_distances(x, y)
        np.testing.assert_allclose(c.numpy(), d.numpy() / 2., atol=1e-12)

    def test_expected_value_uniform(self):
        """For U[0,1]^2: E[c] = E[||x-y||^2]/2 = (1/3)/2 = 1/6."""
        x = _random_particles()
        y = _random_particles(seed=99)
        c = cost(x, y)
        np.testing.assert_allclose(tf.reduce_mean(c).numpy(), 1. / 6., atol=3e-2)


# ────────────────────────────────────────────────────────────
# softmin  (T_epsilon, Eq. 11)
# ────────────────────────────────────────────────────────────

class TestSoftmin:
    
    def test_output_shape(self):
        x = _random_particles()
        c = cost(x, x)
        f = tf.zeros([B, N], dtype=DTYPE)
        eps = tf.constant([0.1] * B, dtype=DTYPE)
        result = softmin(eps, c, f)
        assert result.shape == (B, N)

    def test_very_small_epsilon_approaches_hard_min(self):
        """As epsilon -> 0, softmin -> hard min.

        With f=0, T_epsilon(., 0, C) -> -epsilon * log(sum exp(-c/epsilon))
        -> min_k c_{k,i} as epsilon -> 0.
        For self-distance, min_k c_{k,i} = 0 (diagonal), so the result -> 0.
        """
        x = _random_particles()
        c = cost(x, x)
        f = tf.zeros([B, N], dtype=DTYPE)
        eps = tf.constant([1e-6] * B, dtype=DTYPE)
        result = softmin(eps, c, f)
        np.testing.assert_allclose(result.numpy(), 0., atol=1e-5)

    def test_monotonicity_in_epsilon(self):
        """Softmin decreases as epsilon increases (for self-cost with f=0).

        Smaller epsilon => tighter approximation to min => closer to 0.
        Larger epsilon => softer => more negative.
        So: result(small_eps) > result(large_eps) element-wise.
        """
        x = _random_particles()
        c = cost(x, x)
        f = tf.zeros([B, N], dtype=DTYPE)
        small_eps = tf.constant([0.01] * B, dtype=DTYPE)
        large_eps = tf.constant([0.1] * B, dtype=DTYPE)
        r_small = softmin(small_eps, c, f)
        r_large = softmin(large_eps, c, f)
        assert tf.reduce_all(r_small >= r_large).numpy(), \
            "Softmin should decrease (become more negative) as epsilon increases"

    def test_scalar_epsilon(self):
        """Softmin should also work with a scalar (non-batched) epsilon."""
        x = _random_particles(batch=1)
        c = cost(x, x)
        f = tf.zeros([1, N], dtype=DTYPE)
        eps = tf.constant([0.5], dtype=DTYPE)
        result = softmin(eps, c, f)
        assert result.shape == (1, N)
        assert tf.reduce_all(tf.math.is_finite(result)).numpy()


# ────────────────────────────────────────────────────────────
# diameter (Section 3.2)
# ────────────────────────────────────────────────────────────

class TestDiameter:

    def test_positive(self):
        """Diameter should always be > 0 (returns 1.0 if all particles identical)."""
        x = _random_particles()
        d = diameter(x, x)
        assert tf.reduce_all(d > 0.).numpy()

    def test_identical_particles_returns_one(self):
        """If all particles are the same, diameter should return 1.0 (fallback)."""
        x = tf.ones([B, N, D], dtype=DTYPE) * 3.14
        d = diameter(x, x)
        np.testing.assert_allclose(d.numpy(), 1.0, atol=1e-12)

    def test_known_value(self):
        """For U[0,1]: std ~ 1/sqrt(12) ~ 0.289. With enough particles,
        diameter (max_dim std) should be close to this."""
        x = _random_particles(n_particles=1000)
        d = diameter(x, x)
        expected_std = 1. / np.sqrt(12.)
        np.testing.assert_allclose(d.numpy(), expected_std, atol=5e-2)


# ────────────────────────────────────────────────────────────
# max_min
# ────────────────────────────────────────────────────────────

class TestMaxMin:

    def test_positive(self):
        x = _random_particles()
        r = max_min(x, x)
        assert tf.reduce_all(r > 0.).numpy()

    def test_known_value(self):
        """For U[0,1]: range ~ 1.0 with enough particles."""
        x = _random_particles(n_particles=1000)
        r = max_min(x, x)
        np.testing.assert_allclose(r.numpy(), 1.0, atol=0.1)

    def test_identical_particles_returns_zero(self):
        """All particles identical => range = 0."""
        x = tf.ones([B, N, D], dtype=DTYPE) * 2.0
        r = max_min(x, x)
        np.testing.assert_allclose(r.numpy(), 0.0, atol=1e-12)
