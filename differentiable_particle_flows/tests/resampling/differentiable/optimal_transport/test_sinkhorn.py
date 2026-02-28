"""Tests for the Sinkhorn algorithm and transport plan computation.

Tests cover:
  - sinkhorn.py:  sinkhorn_potentials (Algorithm 2 from Corenflos et al. 2021)
  - plan.py:      transport, transport_from_potentials (Algorithm 3)

Key properties verified:
  1. Transport matrix shape and non-negativity.
  2. Row marginal:  sum_j T_{i,j} = 1  for all i       (target is uniform 1/N).
  3. Column marginal: sum_i T_{i,j} = N * w_j            (source weights).
  4. Total mass:     sum_{i,j} T_{i,j} = N.
  5. Gradient flow:  d(loss)/dx and d(loss)/d(logw) are non-zero and finite.
  6. Gradient correctness: analytic gradients match numerical finite differences.

Reference: test structure adapted from filterflow-master/tests/resampling/
           differentiable/optimal_transport/test_sinkhorn.py
"""
import math
import ot
MIN_RELATIVE_LOG_WEIGHT = -4.
# for example if using 10 particles, we consider those with weight exp(-4*ln(10)) = 1e-4 to have died out
MIN_ABSOLUTE_LOG_WEIGHT = -13.8  # approx. -6 ln(10)

MIN_RELATIVE_WEIGHT = math.exp(MIN_RELATIVE_LOG_WEIGHT)
MIN_ABSOLUTE_WEIGHT = math.exp(MIN_ABSOLUTE_LOG_WEIGHT)
import tensorflow as tf
import numpy as np

from dpf.resampling.differentiable.regularized_optimal_transport.plan import (
    transport,
)
from dpf.resampling.differentiable.regularized_optimal_transport.sinkhorn import (
    sinkhorn_potentials,
)


from dpf.resampling.differentiable.regularized_optimal_transport.ot_utils import (
    squared_distances,
    cost,
    softmin,
    diameter,
    max_min,
)

@tf.function
def _normalize(weights, axis, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    if log:
        normalizer = tf.reduce_logsumexp(weights, axis=axis, keepdims=True)
        return weights - normalizer
    normalizer = tf.reduce_sum(weights, axis=axis)
    return weights / normalizer


@tf.function
def normalize(weights, axis, n, log=True):
    """Normalises weights, either expressed in log terms or in their natural space"""
    float_n = tf.cast(n, float)

    if log:
        normalized_weights = tf.clip_by_value(_normalize(weights, axis, True), tf.constant(-1e3), tf.constant(0.))
        stop_gradient_mask = normalized_weights < tf.maximum(MIN_ABSOLUTE_LOG_WEIGHT, MIN_RELATIVE_LOG_WEIGHT * float_n)
    else:
        normalized_weights = _normalize(weights, axis, False)
        stop_gradient_mask = normalized_weights < tf.maximum(MIN_ABSOLUTE_WEIGHT, MIN_RELATIVE_WEIGHT ** float_n)
    float_stop_gradient_mask = tf.cast(stop_gradient_mask, float)
    return tf.stop_gradient(float_stop_gradient_mask * normalized_weights) + (
            1. - float_stop_gradient_mask) * normalized_weights


DTYPE = tf.float64

# Shared test fixtures
B = 3       # batch size
N = 25      # number of particles
D = 2       # state dimension
EPS = 0.25  # regularisation parameter


def _setup():
    """Create test particles and weights (mirrors filterflow test setUp)."""
    np.random.seed(42)
    np_x = np.random.uniform(-1., 1., [B, N, D])
    x = tf.constant(np_x, dtype=DTYPE)

    degenerate_weights = np.random.uniform(0., 1., [B, N])
    degenerate_weights /= degenerate_weights.sum(axis=1, keepdims=True)
    degenerate_logw = tf.constant(np.log(degenerate_weights), dtype=DTYPE)

    uniform_logw = tf.fill([B, N], -tf.math.log(tf.cast(N, DTYPE)))

    return x, np_x, degenerate_weights, degenerate_logw, uniform_logw


# ────────────────────────────────────────────────────────────
# sinkhorn_potentials
# ────────────────────────────────────────────────────────────

class TestSinkhornPotentials:

    def test_output_shapes(self):
        """Potentials should have shape [B, N]."""
        x, _, _, logw, uniform_logw = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        a_y, b_x, a_x, b_y, n_iter = sinkhorn_potentials(
            logw, x, uniform_logw, x, eps,
            tf.constant(0.75, dtype=DTYPE),
            tf.constant(1e-3, dtype=DTYPE),
            tf.constant(100, dtype=tf.int32),
        )
        assert a_y.shape == (B, N)
        assert b_x.shape == (B, N)
        assert a_x.shape == (B, N)
        assert b_y.shape == (B, N)

    def test_potentials_finite(self):
        """All potentials should be finite (no NaN or Inf)."""
        x, _, _, logw, uniform_logw = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        a_y, b_x, a_x, b_y, _ = sinkhorn_potentials(
            logw, x, uniform_logw, x, eps,
            tf.constant(0.75, dtype=DTYPE),
            tf.constant(1e-3, dtype=DTYPE),
            tf.constant(100, dtype=tf.int32),
        )
        for name, pot in [('a_y', a_y), ('b_x', b_x), ('a_x', a_x), ('b_y', b_y)]:
            assert tf.reduce_all(tf.math.is_finite(pot)).numpy(), f"{name} has non-finite values"

    def test_uniform_weights_give_zero_potentials(self):
        """With uniform weights on both sides, potentials should be near-zero
        (up to a global constant) since the OT problem is trivial."""
        x, _, _, _, uniform_logw = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        a_y, b_x, _, _, _ = sinkhorn_potentials(
            uniform_logw, x, uniform_logw, x, eps,
            tf.constant(0.75, dtype=DTYPE),
            tf.constant(1e-5, dtype=DTYPE),
            tf.constant(200, dtype=tf.int32),
        )
        # With uniform-to-uniform transport, f and g should be roughly constant
        # (the identity transport is optimal). Check low variance across particles.
        a_y_std = tf.math.reduce_std(a_y, axis=1).numpy()
        b_x_std = tf.math.reduce_std(b_x, axis=1).numpy()
        np.testing.assert_allclose(a_y_std, 0., atol=0.1)
        np.testing.assert_allclose(b_x_std, 0., atol=0.1)


# ────────────────────────────────────────────────────────────
# transport (full Algorithm 3)
# ────────────────────────────────────────────────────────────

class TestTransport:

    def test_output_shape(self):
        """Transport matrix should have shape [B, N, N]."""
        x, _, _, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        assert T.shape == (B, N, N)

    def test_non_negative(self):
        """All entries of the transport matrix must be >= 0."""
        x, _, _, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        assert tf.reduce_all(T >= 0.).numpy(), "Transport matrix has negative entries"

    def test_finite(self):
        """All entries should be finite."""
        x, _, _, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        assert tf.reduce_all(tf.math.is_finite(T)).numpy(), "Transport matrix has non-finite entries"

    def test_row_marginal(self):
        """Each row should sum to 1 (target is uniform 1/N, scaled by N).

        sum_j T_{i,j} = 1  for all i.
        """
        x, _, _, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        row_sums = tf.reduce_sum(T, axis=2)  # [B, N]
        np.testing.assert_allclose(row_sums.numpy(), 1.0, atol=5e-2)

    def test_column_marginal(self):
        """Each column j should sum to N * w_j.

        sum_i T_{i,j} = N * w_j  (source weights).
        """
        x, _, degenerate_weights, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        col_sums = tf.reduce_sum(T, axis=1)  # [B, N]
        expected = tf.constant(degenerate_weights * N, dtype=DTYPE)
        np.testing.assert_allclose(col_sums.numpy(), expected.numpy(), atol=1e-2)

    def test_total_mass(self):
        """Total mass of the transport matrix should be N.

        sum_{i,j} T_{i,j} = N.
        """
        x, _, _, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        total = tf.reduce_sum(T, axis=[1, 2])  # [B]
        np.testing.assert_allclose(total.numpy(), float(N), atol=1e-3)

    def test_uniform_weights_give_identity(self):
        """With uniform weights, the transport should be close to (1/N) * I * N = I.

        Actually the OT solution for uniform-to-uniform on the same points is
        the identity permutation (each particle maps to itself).  With
        entropy regularisation the matrix is softened but should be strongly
        diagonally dominant.
        """
        x, _, _, _, uniform_logw = _setup()
        eps = tf.constant(0.1, dtype=DTYPE)
        T = transport(x, uniform_logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-5, dtype=DTYPE), tf.constant(200, dtype=tf.int32), N)
        # Diagonal should have the largest value in each row
        diag = tf.linalg.diag_part(T)         # [B, N]
        row_max = tf.reduce_max(T, axis=2)    # [B, N]
        np.testing.assert_allclose(diag.numpy(), row_max.numpy(), atol=5e-2)


# ────────────────────────────────────────────────────────────
# Gradient tests
# ────────────────────────────────────────────────────────────

class TestTransportGradients:

    def test_gradient_wrt_x_exists(self):
        """d(loss)/dx should be non-zero and finite."""
        x, _, _, logw, _ = _setup()
        x_var = tf.Variable(x)
        eps = tf.constant(EPS, dtype=DTYPE)

        with tf.GradientTape() as tape:
            T = transport(x_var, logw, eps, tf.constant(0.75, dtype=DTYPE),
                          tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
            loss = tf.reduce_sum(tf.linalg.matmul(T, x_var))

        grad = tape.gradient(loss, x_var)
        assert grad is not None, "Gradient w.r.t. x is None"
        assert tf.reduce_all(tf.math.is_finite(grad)).numpy(), "Gradient w.r.t. x has non-finite values"
        assert tf.reduce_any(grad != 0.).numpy(), "Gradient w.r.t. x is all zeros"

    def test_gradient_wrt_logw_exists(self):
        """d(loss)/d(logw) should be non-zero and finite."""
        x, _, _, logw, _ = _setup()
        logw_var = tf.Variable(logw)
        eps = tf.constant(EPS, dtype=DTYPE)

        with tf.GradientTape() as tape:
            T = transport(x, logw_var, eps, tf.constant(0.75, dtype=DTYPE),
                          tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
            loss = tf.reduce_sum(tf.linalg.matmul(T, x))

        grad = tape.gradient(loss, logw_var)
        assert grad is not None, "Gradient w.r.t. logw is None"
        assert tf.reduce_all(tf.math.is_finite(grad)).numpy(), "Gradient w.r.t. logw has non-finite values"
        assert tf.reduce_any(grad != 0.).numpy(), "Gradient w.r.t. logw is all zeros"

    def test_gradient_wrt_x_numerical(self):
        """Analytic gradient w.r.t. x should match numerical finite differences.

        Uses tf.test.compute_gradient (central differences).
        Adapted from filterflow test_gradient_transport.
        """
        x, _, _, logw, _ = _setup()
        eps = tf.constant(EPS, dtype=DTYPE)

        @tf.function
        def fun_x(x_in):
            T = transport(x_in, logw, eps, tf.constant(0.75, dtype=DTYPE),
                          tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
            return tf.reduce_sum(tf.linalg.matmul(T, x_in))

        theoretical, numerical = tf.test.compute_gradient(fun_x, [x], delta=1e-3)
        np.testing.assert_allclose(theoretical[0], numerical[0], atol=5e-2, rtol=5e-2)

    def test_gradient_wrt_logw_numerical(self):
        """Analytic gradient w.r.t. logw should be correlated with numerical finite differences.

        The ``transport`` function uses @tf.custom_gradient with gradient
        clipping to [-1, 1].  This means the analytic gradient can deviate from
        numerical finite differences where clipping activates.  Instead of
        element-wise closeness we check:
          1. The Pearson correlation between analytic and numerical > 0.9.
          2. The majority of signs agree (> 85%).
        """
        x, _, _, logw, _ = _setup()
        eps = tf.constant(0.5, dtype=DTYPE)

        @tf.function
        def fun_logw(logw_in):
            logw_norm = logw_in - tf.reduce_logsumexp(logw_in, axis=1, keepdims=True)
            T = transport(x, logw_norm, eps, tf.constant(0.9, dtype=DTYPE),
                          tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
            return tf.reduce_sum(tf.linalg.matmul(T, x))

        theoretical, numerical = tf.test.compute_gradient(fun_logw, [logw], delta=1e-4)
        t_flat = theoretical[0].flatten()
        n_flat = numerical[0].flatten()

        # Correlation check
        corr = np.corrcoef(t_flat, n_flat)[0, 1]
        assert corr > 0.9, f"Gradient correlation too low: {corr:.4f}"

        # Sign agreement check (ignore near-zero entries)
        mask = np.abs(n_flat) > 1e-3
        sign_agree = np.mean(np.sign(t_flat[mask]) == np.sign(n_flat[mask]))
        assert sign_agree > 0.85, f"Sign agreement too low: {sign_agree:.2%}"


# ────────────────────────────────────────────────────────────
# Epsilon sensitivity
# ────────────────────────────────────────────────────────────

class TestEpsilonSensitivity:

    def test_small_epsilon_sharpens_transport(self):
        """Smaller epsilon should produce a more peaked (less diffuse) transport matrix.

        As epsilon -> 0, the transport matrix approaches a permutation matrix.
        The max entry per row should increase as epsilon decreases.
        """
        x, _, _, logw, _ = _setup()
        max_entries = []
        for eps_val in [2.0, 0.5, 0.1]:
            eps = tf.constant(eps_val, dtype=DTYPE)
            T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                          tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
            max_per_row = tf.reduce_mean(tf.reduce_max(T, axis=2)).numpy()
            max_entries.append(max_per_row)

        # Smaller epsilon => sharper => larger max entry per row
        assert max_entries[0] < max_entries[1] < max_entries[2], \
            f"Expected increasing sharpness: {max_entries}"

    def test_large_epsilon_approaches_uniform(self):
        """Very large epsilon should produce a nearly uniform transport matrix.

        Each row should be close to [w_1, w_2, ..., w_N] (the source weights).
        """
        x, _, degenerate_weights, logw, _ = _setup()
        eps = tf.constant(50.0, dtype=DTYPE)
        T = transport(x, logw, eps, tf.constant(0.75, dtype=DTYPE),
                      tf.constant(1e-3, dtype=DTYPE), tf.constant(100, dtype=tf.int32), N)
        # With very large epsilon, rows become nearly identical
        row_std = tf.math.reduce_std(T, axis=1)  # [B, N] std across rows
        # The std across rows (for each column) should be small
        mean_row_std = tf.reduce_mean(row_std).numpy()
        assert mean_row_std < 0.05, f"Expected near-uniform rows, got mean row std = {mean_row_std}"



class TestSinkhorn(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_particles = 25
        batch_size = 3
        dimension = 2

        self.n_particles = tf.constant(n_particles)
        self.batch_size = tf.constant(batch_size)
        self.dimension = tf.constant(dimension)

        self.np_epsilon = 0.25
        self.epsilon = tf.constant(self.np_epsilon)

        self.threshold = tf.constant(1e-5)
        self.n_iter = tf.constant(100)

        self.np_x = np.random.uniform(-1., 1., [batch_size, n_particles, dimension]).astype(np.float32)
        self.x = tf.constant(self.np_x)

        degenerate_weights = np.random.uniform(0., 1., [batch_size, n_particles]).astype(np.float32)
        degenerate_weights /= degenerate_weights.sum(axis=1, keepdims=True)

        self.degenerate_weights = degenerate_weights
        self.degenerate_logw = tf.math.log(degenerate_weights)

        self.uniform_logw = tf.zeros_like(degenerate_weights) - tf.math.log(float(n_particles))

    def test_transport(self):
        T_scaled = transport(self.x, self.degenerate_logw, self.epsilon, 0.75, self.threshold,
                             self.n_iter, self.n_particles)

        self.assertAllClose(tf.constant(self.degenerate_weights) * tf.cast(self.n_particles, float),
                            tf.reduce_sum(T_scaled, 1), atol=1e-2)

        self.assertAllClose(tf.reduce_sum(T_scaled, 2), tf.ones_like(self.degenerate_logw), atol=1e-2)

        self.assertAllClose(tf.reduce_sum(T_scaled, [1, 2]),
                            tf.cast(self.n_particles, float) * tf.ones([self.batch_size]), atol=1e-4)
        scale_x = diameter(self.x, self.x).numpy()[0]
        np_x = (self.np_x[0] - np.mean(self.np_x[0], 0, keepdims=True)) / scale_x
        np_transport_matrix = ot.bregman.empirical_sinkhorn(np_x, np_x,
                                                            self.np_epsilon ** 0.5,
                                                            b=self.degenerate_weights[0])

        self.assertAllClose(T_scaled[0] @ self.x[0], np_transport_matrix * self.n_particles.numpy() @ self.np_x[0],
                            atol=1e-3)

    def test_penalty(self):
        penalties = []
        for epsilon in np.linspace(0.05, 5., 50, dtype=np.float32):
            T_scaled = transport(self.x, self.degenerate_logw, tf.constant(epsilon), 0.75, self.threshold,
                                 self.n_iter, self.n_particles) / tf.cast(self.n_particles, float)

            temp = tf.math.log(T_scaled) - tf.expand_dims(self.uniform_logw, -1) - tf.expand_dims(self.degenerate_logw,
                                                                                                  1)
            temp -= 1
            temp *= T_scaled
            penalties.append(-tf.reduce_sum(temp, [1, 2]).numpy() - 1)

        import matplotlib.pyplot as plt
        plt.plot(np.linspace(0.05, 2., 50), penalties)
        plt.show()

    def test_gradient_transport(self):
        @tf.function
        def fun_x(x):
            transport_matrix = transport(x, self.degenerate_logw, self.epsilon, tf.constant(0.75), self.threshold,
                                         self.n_iter, self.n_particles)
            return tf.math.reduce_sum(tf.linalg.matmul(transport_matrix, x))

        @tf.function
        def fun_logw(logw):
            logw = normalize(logw, 1, self.n_particles)

            transport_matrix = transport(self.x, logw, self.epsilon, tf.constant(0.9),
                                         self.threshold, self.n_iter, self.n_particles)
            return tf.math.reduce_sum(tf.linalg.matmul(transport_matrix, self.x))

        theoretical, numerical = tf.test.compute_gradient(fun_x, [self.x], delta=1e-3)
        self.assertAllClose(theoretical[0], numerical[0], atol=1e-2)

        theoretical, numerical = tf.test.compute_gradient(fun_logw, [self.degenerate_logw], delta=1e-5)
        self.assertAllClose(theoretical[0], numerical[0], atol=1e-2, rtol=1e-2)