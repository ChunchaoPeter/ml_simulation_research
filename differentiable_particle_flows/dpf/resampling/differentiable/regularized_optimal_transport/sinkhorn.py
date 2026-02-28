"""
This is adapted from 
https://github.com/JTT94/filterflow/blob/master/filterflow/resampling/differentiable/regularized_transport/sinkhorn.py

Sinkhorn algorithm for computing dual potentials of entropy-regularised OT.

Implements Algorithm 2 of Corenflos et al. (2021) — the stabilised Sinkhorn


Algorithm 2  (Sinkhorn Algorithm)
---------------------------------
  Input : Potentials(a, b, u, v)   — log-measures a, b and point clouds u, v
  Local : f, g in R^N
  Init  : f = 0,  g = 0
  Set   : C <- uu^t + vv^t - 2uv^t              (squared-distance cost)
  while stopping criterion not met do
      for i in [N] do
          f_i <- (1/2)( f_i + T_epsilon(b, g, C_{:,i}) )
          g_i <- (1/2)( g_i + T_epsilon(a, f, C_{:,i}) )
      end for
  end while
  Return f, g

The averaging  f <- (f + T_epsilon(...))/2  is the stabilised update from

Additionally this implementation uses:

  * **Epsilon-scaling** (multi-scale strategy): Start from a large
    epsilon_0 ~ diameter^2 and geometrically decrease toward the target
    epsilon, multiplying by ``scaling^2`` each iteration.  This ensures
    convergence from a good initial point at each scale.

Reference
---------
Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021).
Differentiable Particle Filtering via Entropy-Regularized Optimal Transport.
ICML 2021 (PMLR 139).

"""

import tensorflow as tf

from dpf.resampling.differentiable.regularized_optimal_transport.ot_utils import cost, softmin, max_min


@tf.function
def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy,
                  epsilon, particles_diameter, scaling, threshold, max_iter):
    """Run the stabilised Sinkhorn loop with epsilon-scaling.

    This is the inner workhorse that iterates the fixed-point equations
    (Corenflos et al. 2021, Eq. 11):

        f_i* = T_epsilon(b, g*, C_{i,:})
        g_i* = T_epsilon(a, f*,  C_{:,i})

    using the stabilised (averaged) update from Algorithm 2:

        f_i <- (1/2)( f_i + T_epsilon(b, g, C_{:,i}) )
        g_i <- (1/2)( g_i + T_epsilon(a, f, C_{:,i}) )

    The epsilon-scaling schedule starts at  epsilon_0 = diameter^2  and
    geometrically reduces toward the target epsilon by multiplying by
    ``scaling^2`` each iteration, providing a coarse-to-fine resolution.

    After convergence, gradient stitching is applied: converged potentials
    are detached (stop_gradient) and one final differentiable softmin
    evaluation is performed at the target epsilon.

    Parameters
    ----------
    log_alpha : tf.Tensor, shape [B, N]
        Log source measure weights  (log w_i  in the resampling context).
    log_beta : tf.Tensor, shape [B, N]
        Log target measure weights  (log(1/N)  for uniform target).
    cost_xy, cost_yx, cost_xx, cost_yy : tf.Tensor, shape [B, N, N]
        Pairwise cost matrices.  In the self-transport setting (x = y)
        all four are identical, but they are kept separate for generality
        and for the debiased Sinkhorn divergence.
    epsilon : tf.Tensor, shape [B] or scalar
        Target regularisation parameter.
    particles_diameter : tf.Tensor, shape [B]
        Range of the particle cloud, used to set epsilon_0 = diameter^2.
    scaling : tf.Tensor, scalar
        Geometric scaling factor; epsilon decreases by ``scaling^2`` per step.
    threshold : tf.Tensor, scalar
        Convergence threshold on  max|f_new - f_old|.
    max_iter : tf.Tensor, scalar int
        Maximum number of Sinkhorn iterations.

    Returns
    -------
    final_a_y : tf.Tensor, shape [B, N]
        Converged dual potential f* (source potential evaluated on y).
    final_b_x : tf.Tensor, shape [B, N]
        Converged dual potential g* (target potential evaluated on x).
    final_a_x, final_b_y : tf.Tensor, shape [B, N]
        Debiasing potentials (self-interaction terms).
    total_iter : tf.Tensor, scalar int
        Total number of iterations performed (including the final step).
    """
    batch_size = log_alpha.shape[0]
    continue_flag = tf.ones([batch_size], dtype=bool)
    epsilon_0 = particles_diameter ** 2
    scaling_factor = scaling ** 2

    # --- Initialise potentials at coarse epsilon_0 ---
    a_y_init = softmin(epsilon_0, cost_yx, log_alpha)
    b_x_init = softmin(epsilon_0, cost_xy, log_beta)

    a_x_init = softmin(epsilon_0, cost_xx, log_alpha)
    b_y_init = softmin(epsilon_0, cost_yy, log_beta)

    # --- While-loop helpers ---
    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, continue_, _running_epsilon):
        n_iter_cond = i < max_iter - 1
        return tf.logical_and(n_iter_cond, tf.reduce_all(continue_))

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        """One stabilised Sinkhorn step (Algorithm 2, lines 6-9).

        For each potential pair, compute:
            f_i <- (1/2)( f_i + T_epsilon(b, g, C_{:,i}) )
            g_i <- (1/2)( g_i + T_epsilon(a, f, C_{:,i}) )

        The softmin arguments  ``log_alpha + b_x / epsilon``  combine the
        log-measure weight  log a_k  with the scaled potential  g_k / epsilon,
        matching the T_epsilon definition in Eq. 11.
        """
        running_epsilon_ = tf.reshape(running_epsilon, [-1, 1])
        continue_reshaped = tf.reshape(continue_, [-1, 1])

        # Cross-potentials  (Eq. 11:  f* = T_eps(b, g*, C_{:,i}) )
        at_y = tf.where(continue_reshaped, softmin(running_epsilon, cost_yx, log_alpha + b_x / running_epsilon_), a_y)
        bt_x = tf.where(continue_reshaped, softmin(running_epsilon, cost_xy, log_beta + a_y / running_epsilon_), b_x)

        at_x = tf.where(continue_reshaped, softmin(running_epsilon, cost_xx, log_alpha + a_x / running_epsilon_), a_x)
        bt_y = tf.where(continue_reshaped, softmin(running_epsilon, cost_yy, log_beta + b_y / running_epsilon_), b_y)

        # Stabilised averaging  (Algorithm 2, lines 7-8)
        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2

        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2

        # Convergence check:  max|delta f|, max|delta g|
        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y), 1)
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x), 1)

        local_continue = tf.logical_or(a_y_diff > threshold, b_x_diff > threshold)
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    def body(i, a_y, b_x, a_x, b_y, continue_, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, local_continue = apply_one(a_y, b_x, a_x, b_y, continue_,
                                                                       running_epsilon)
        # Epsilon-scaling: decrease running_epsilon toward target epsilon
        new_epsilon = tf.maximum(running_epsilon * scaling_factor, epsilon)
        # Continue if epsilon is still decreasing OR potentials have not converged
        global_continue = tf.logical_or(new_epsilon < running_epsilon, local_continue)

        return i + 1, new_a_y, new_b_x, new_a_x, new_b_y, global_continue, new_epsilon

    n_iter = tf.constant(0)

    # --- Main Sinkhorn loop ---
    total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, _, final_epsilon = tf.while_loop(
        stop_condition,
        body,
        loop_vars=[n_iter,
                   a_y_init,
                   b_x_init,
                   a_x_init,
                   b_y_init,
                   continue_flag,
                   epsilon_0])

    # --- Gradient stitching (Feydy et al. 2019) ---
    # Detach converged potentials so gradients do not flow through the loop,
    # then perform one final differentiable softmin step at the target epsilon.
    converged_a_y, converged_b_x, converged_a_x, converged_b_y, = tf.nest.map_structure(tf.stop_gradient,
                                                                                        (converged_a_y,
                                                                                         converged_b_x,
                                                                                         converged_a_x,
                                                                                         converged_b_y))
    epsilon_ = tf.reshape(epsilon, [-1, 1])
    final_a_y = softmin(epsilon, cost_yx, log_alpha + converged_b_x / epsilon_)
    final_b_x = softmin(epsilon, cost_xy, log_beta + converged_a_y / epsilon_)
    final_a_x = softmin(epsilon, cost_xx, log_alpha + converged_a_x / epsilon_)
    final_b_y = softmin(epsilon, cost_yy, log_beta + converged_b_y / epsilon_)

    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2


@tf.function
def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, scaling, threshold, max_iter):
    """Compute Sinkhorn dual potentials (f*, g*) for the OT problem.

    This is the entry point that:
      1. Builds the four cost matrices  C_xy, C_yx, C_xx, C_yy.
      2. Computes the particle-cloud range for epsilon-scaling.
      3. Calls ``sinkhorn_loop`` to solve for the potentials.

    In the resampling context (Algorithm 3, line 2), this is called as:

        f, g <- Potentials(w, 1/N * 1, X, X)

    i.e., source measure = weighted particles, target measure = uniform
    over the same particle locations.

    Parameters
    ----------
    log_alpha : tf.Tensor, shape [B, N]
        Log source weights  (log w_i).
    x : tf.Tensor, shape [B, N, D]
        Source point cloud.
    log_beta : tf.Tensor, shape [B, N]
        Log target weights  (log(1/N) for uniform).
    y : tf.Tensor, shape [B, N, D]
        Target point cloud  (= x in the self-transport case).
    epsilon : tf.Tensor
        Target regularisation parameter.
    scaling : tf.Tensor
        Epsilon-scaling factor.
    threshold : tf.Tensor
        Convergence threshold.
    max_iter : tf.Tensor
        Maximum iterations.

    Returns
    -------
    a_y : tf.Tensor, shape [B, N]
        Dual potential f* (source, evaluated on y points).
    b_x : tf.Tensor, shape [B, N]
        Dual potential g* (target, evaluated on x points).
    a_x, b_y : tf.Tensor, shape [B, N]
        Debiasing potentials.
    total_iter : tf.Tensor
        Number of iterations performed.
    """
    cost_xy = cost(x, tf.stop_gradient(y))
    cost_yx = cost(y, tf.stop_gradient(x))
    cost_xx = cost(x, tf.stop_gradient(x))
    cost_yy = cost(y, tf.stop_gradient(y))
    scale = tf.stop_gradient(max_min(x, y))
    a_y, b_x, a_x, b_y, total_iter = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon,
                                                   scale, scaling, threshold, max_iter)

    return a_y, b_x, a_x, b_y, total_iter
