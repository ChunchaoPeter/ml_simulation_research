"""
Utility functions for entropy-regularised optimal transport.

Provides the core building blocks used by the Sinkhorn algorithm and
transport-plan recovery:

  - squared_distances / cost : pairwise cost matrix  c_{i,j}
  - softmin                  : the T_epsilon mapping (Corenflos et al. 2021, Eq. 11)
  - diameter / max_min       : particle-cloud scaling (Section 3.2)

Reference
---------
Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021).
Differentiable Particle Filtering via Entropy-Regularized Optimal Transport.
ICML 2021 (PMLR 139).
"""

import tensorflow as tf


@tf.function
def diameter(x, y):
    """Compute a scale factor for the particle cloud.

    Used in the particle-scaling step of Section 3.2 of Corenflos et al. (2021).
    The paper defines

        delta(X_t) = sqrt(d_x) * max_{k in [d_x]} std_i(X^t_{i,k})

    This function computes the  max_k std_i(.)  part; the sqrt(d_x) factor is
    applied externally in ``transport()`` (plan.py).

    Parameters
    ----------
    x : tf.Tensor, shape [B, N, D]
    y : tf.Tensor, shape [B, M, D]

    Returns
    -------
    tf.Tensor, shape [B]
        Per-batch maximum standard deviation across dimensions.
        Returns 1.0 for any batch element where the result would be zero
        (all particles identical) to avoid division by zero.
    """
    diameter_x = tf.reduce_max(tf.math.reduce_std(x, 1), -1)
    diameter_y = tf.reduce_max(tf.math.reduce_std(y, 1), -1)
    res = tf.maximum(diameter_x, diameter_y)
    return tf.where(res == 0., tf.ones_like(res), res)


@tf.function
def max_min(x, y):
    """Compute the range (max - min) of the particle cloud.

    Used as the initial large epsilon (epsilon_0 = range^2) for
    the epsilon-scaling schedule in the Sinkhorn loop (see sinkhorn.py).

    Parameters
    ----------
    x : tf.Tensor, shape [B, N, D]
    y : tf.Tensor, shape [B, M, D]

    Returns
    -------
    tf.Tensor, shape [B]
        Per-batch range: max(x, y) - min(x, y) across all particles
        and dimensions.
    """
    max_max = tf.maximum(tf.math.reduce_max(x, [1, 2]), tf.math.reduce_max(y, [1, 2]))
    min_min = tf.minimum(tf.math.reduce_min(x, [1, 2]), tf.math.reduce_min(y, [1, 2]))

    return max_max - min_min


def softmin(epsilon, cost_matrix, f):
    """The softmin operator T_epsilon (Corenflos et al. 2021, Eq. 11).

    This is the core building block of the Sinkhorn fixed-point iteration.
    The paper defines the mapping  T_epsilon : R^N x R^N x R^N -> R^N  as

        T_epsilon(a, f, C_{:,i}) = -epsilon * log sum_k { exp( log a_k + epsilon^{-1}(f_k - c_{k,i}) ) }

    In this implementation the log-measure weights  log a_k  are absorbed into
    ``f`` *before* the call (the caller passes  log_alpha + g / epsilon  as ``f``),
    so the function computes the reduced form:

        softmin(epsilon, C, f)_i = -epsilon * logsumexp_k( f_k - c_{k,i} / epsilon )

    The logsumexp trick is used for numerical stability.

    Parameters
    ----------
    epsilon : tf.Tensor, shape [B] or scalar
        Regularisation parameter (may vary per batch during epsilon-scaling).
    cost_matrix : tf.Tensor, shape [B, N, M]
        Pairwise cost matrix.
    f : tf.Tensor, shape [B, N]
        Dual potential (with log-measure weights already incorporated).

    Returns
    -------
    tf.Tensor, shape [B, N]
        The softmin values for each target index i.
    """
    n = cost_matrix.shape[1]
    b = cost_matrix.shape[0]

    f_ = tf.reshape(f, (b, 1, n))
    temp_val = f_ - cost_matrix / tf.reshape(epsilon, (-1, 1, 1))
    log_sum_exp = tf.reduce_logsumexp(temp_val, axis=2)
    res = -tf.reshape(epsilon, (-1, 1)) * log_sum_exp

    return res


@tf.function
def squared_distances(x, y):
    """Pairwise squared Euclidean distances.

    Computes  ||x_i - y_j||^2  for all pairs (i, j) using the identity

        ||x_i - y_j||^2 = ||x_i||^2 - 2 x_i . y_j + ||y_j||^2

    which is more efficient than explicit pairwise subtraction.

    Parameters
    ----------
    x : tf.Tensor, shape [B, N, D]
    y : tf.Tensor, shape [B, M, D]

    Returns
    -------
    tf.Tensor, shape [B, N, M]
        Entry (b, i, j) is  ||x^b_i - y^b_j||^2, clipped to [0, inf).
    """
    # x.shape = [B, N, D]
    xx = tf.reduce_sum(x * x, axis=2, keepdims=True)
    xy = tf.matmul(x, y, transpose_b=True)
    yy = tf.expand_dims(tf.reduce_sum(y * y, axis=-1), 1)
    result = xx - 2 * xy + yy
    return tf.clip_by_value(result, tf.cast(0., result.dtype), tf.cast(float('inf'), result.dtype))


@tf.function
def cost(x, y):
    """Pairwise squared-Euclidean cost matrix (halved).

    Computes  c_{i,j} = (1/2) ||x_i - y_j||^2 .

    This corresponds to the cost used in the 2-Wasserstein OT problem
    (Corenflos et al. 2021, Eq. 5) up to a factor of 2:

        W_2^2(alpha, beta) = min_{P in S(a,b)} sum_{i,j} c_{i,j} p_{i,j}

    The factor 1/2 that is rescale C

    Parameters
    ----------
    x : tf.Tensor, shape [B, N, D]
    y : tf.Tensor, shape [B, M, D]

    Returns
    -------
    tf.Tensor, shape [B, N, M]
    """
    return squared_distances(x, y) / 2.
