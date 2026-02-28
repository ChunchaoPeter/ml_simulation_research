"""
This is adapted from 
https://github.com/JTT94/filterflow/blob/master/filterflow/resampling/differentiable/regularized_transport/plan.py

Transport plan computation for entropy-regularised OT resampling.

Implements the core of Algorithm 3 (DET Resampling) from Corenflos et al. (2021):

  1. Centre and scale particles (Section 3.2).
  2. Solve for Sinkhorn dual potentials f*, g* (Algorithm 2).
  3. Recover the transport matrix P^OT_epsilon from the potentials (Eq. 12).

The main public function is ``transport()``, which returns the transport
matrix and supports a custom gradient for efficient back-propagation.

Reference
---------
Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021).
Differentiable Particle Filtering via Entropy-Regularized Optimal Transport.
ICML 2021 (PMLR 139).
"""

import tensorflow as tf

from dpf.resampling.differentiable.regularized_optimal_transport.sinkhorn import sinkhorn_potentials
from dpf.resampling.differentiable.regularized_optimal_transport.ot_utils import cost, diameter


@tf.function
def transport_from_potentials(x, f, g, eps, logw, n):
    """Recover the scaled transport matrix T from Sinkhorn potentials.

    This function builds the matrix T used in the resampling step (Eq. 13):

        X_tilde = T @ X

    where T = N * P^OT_epsilon and P^OT_epsilon is the entropy-regularised
    OT plan from Eq. 12 of Corenflos et al. (2021):

        P_{i,j} = (1/N) * w_j * exp( (f_i + g_j - c_{i,j}) / epsilon )

    so that T_{i,j} = w_j * exp( (f_i + g_j - c_{i,j}) / epsilon ),
    where the N and 1/N factors cancel.

    Parameters
    ----------
    x : tf.Tensor, shape [B, N, D]
        (Scaled) particle positions.
    f : tf.Tensor, shape [B, N]
        Converged dual potential f* (from sinkhorn_potentials).
    g : tf.Tensor, shape [B, N]
        Converged dual potential g* (from sinkhorn_potentials).
    eps : tf.Tensor
        Regularisation parameter epsilon.
    logw : tf.Tensor, shape [B, N]
        Log source weights log(w_j).
    n : tf.Tensor or float
        Number of particles N.

    Returns
    -------
    tf.Tensor, shape [B, N, N]
        The scaled transport matrix T = N * P^OT_epsilon, ready for
        X_tilde = T @ X.
    """
    float_n = tf.cast(n, x.dtype)
    log_n = tf.math.log(float_n)

    cost_matrix = cost(x, x)
    fg = tf.expand_dims(f, 2) + tf.expand_dims(g, 1)  # fg = f + g.T
    temp = fg - cost_matrix
    temp = temp / eps

    temp = temp - tf.reduce_logsumexp(temp, 1, keepdims=True) + log_n
    # Multiply by source weights to enforce marginal constraint
    temp = temp + tf.expand_dims(logw, 1)

    transport_matrix = tf.math.exp(temp)

    return transport_matrix


@tf.function
@tf.custom_gradient
def transport(x, logw, eps, scaling, threshold, max_iter, n):
    """Compute the entropy-regularised OT transport matrix (Algorithm 3).

    This is the main function implementing the Differentiable Ensemble
    Transform (DET) resampling from Corenflos et al. (2021, Algorithm 3):

        Algorithm 3  (DET Resampling)
        ─────────────────────────────
        Input : EnsembleTransform(X, w, N)
        1.  f, g <- Potentials(w, 1/N * 1, X, X)      [Sinkhorn, Alg. 2]
        2.  for i, j in [N]:
                p^OT_{eps,i,j} = (w_i / N) exp( (f_i + g_j - c_{i,j}) / eps )
        3.  Return  X_tilde = N * P^OT_eps * X         [Eq. 13]

    Before solving, particles are centred and rescaled following Section 3.2:

        X_bar = X - mean(X)
        X_hat = X_bar / ( delta(X) * sqrt(d_x) )

    where  delta(X) = max_k std_i(X_{i,k})  (see ``diameter`` in ot_utils.py).
    This makes epsilon approximately independent of the particle scale and
    dimension.

    A ``@tf.custom_gradient`` is used to implement efficient back-propagation:
    gradients are computed via automatic differentiation through the final
    transport-matrix expression only (not through the Sinkhorn loop), with
    gradient clipping for stability.

    Parameters
    ----------
    x : tf.Tensor, shape [B, N, D]
        Particle positions  X_t.
    logw : tf.Tensor, shape [B, N]
        Log importance weights  log(w_t^i).
    eps : tf.Tensor
        Entropy regularisation parameter  epsilon > 0.
    scaling : tf.Tensor
        Epsilon-scaling factor for multi-scale Sinkhorn  (e.g. 0.75).
    threshold : tf.Tensor
        Convergence threshold for the Sinkhorn potentials.
    max_iter : tf.Tensor
        Maximum number of Sinkhorn iterations.
    n : int
        Number of particles N.

    Returns
    -------
    transport_matrix : tf.Tensor, shape [B, N, N]
        The regularised OT transport matrix  P^OT_epsilon.
        Applied as  X_tilde = transport_matrix @ X  (Eq. 13, with the N
        factor absorbed into the normalisation).
    """
    float_n = tf.cast(n, x.dtype)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    dimension = tf.cast(x.shape[-1], x.dtype)

    # --- Step 0: Centre and scale particles (Section 3.2) ---
    # delta(X) = sqrt(d_x) * max_k std_i(X_{i,k})
    centered_x = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    diameter_value = diameter(x, x)
    scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(dimension)
    scaled_x = centered_x / tf.stop_gradient(scale)

    # --- Step 1: Solve for Sinkhorn potentials (Algorithm 2) ---
    # f, g <- Potentials(w, 1/N, X_hat, X_hat)
    alpha, beta, _, _, _ = sinkhorn_potentials(logw, scaled_x, uniform_log_weight, scaled_x, eps, scaling, threshold,
                                               max_iter)

    # --- Step 2: Recover transport matrix (Eq. 12) ---
    transport_matrix = transport_from_potentials(scaled_x, alpha, beta, eps, logw, float_n)

    def grad(d_transport):
        """Custom gradient: differentiate through the transport matrix expression only.

        Gradient clipping (to [-1, 1]) prevents exploding gradients that can
        occur when particles collapse or weights become extreme.
        """
        d_transport = tf.clip_by_value(d_transport, tf.cast(-1., d_transport.dtype), tf.cast(1., d_transport.dtype))
        dx, dlogw = tf.gradients(transport_matrix, [x, logw], d_transport)
        return dx, dlogw, None, None, None, None, None

    return transport_matrix, grad
