"""
Differentiable Ensemble Transform (DET) resampling via entropy-regularised OT.

Implements the differentiable resampling scheme from:

    Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021).
    Differentiable Particle Filtering via Entropy-Regularized Optimal Transport.
    ICML 2021 (PMLR 139).

Standard resampling methods (multinomial, systematic, etc.) select particles
by sampling from a discrete distribution, which is not differentiable w.r.t.
the model parameters theta.  This blocks end-to-end gradient-based training
of state-space models.

The DET resampling scheme replaces discrete sampling with a smooth
**optimal transport** step.  It solves the entropy-regularised OT problem
(Cuturi, 2013):

    W^2_{2,epsilon}(alpha_N, beta_N)
        = min_{P in S(a,b)} sum_{i,j} p_{i,j} ( c_{i,j} + epsilon log(p_{i,j} / (a_i b_j)) )
                                                                              (Eq. 9)

where:
  - alpha_N = sum_i w_i delta_{X_i}    is the weighted particle measure,
  - beta_N  = (1/N) sum_i delta_{X_i}  is the uniform target,
  - c_{i,j} = (1/2)||X_i - X_j||^2    is the squared-Euclidean cost,
  - epsilon > 0                         is the regularisation strength.

The unique minimiser  P^OT_epsilon  is recovered from dual Sinkhorn
potentials (f*, g*) via (Eq. 12):

    p^OT_{epsilon,i,j} = a_i b_j exp( epsilon^{-1}(f*_i + g*_j - c_{i,j}) )

The transported (resampled) particles are then (Eq. 13):

    X_tilde_i = N * sum_k p^OT_{epsilon,i,k} X_k  :=  T_{N,epsilon}(X_i)

and the new weights are set to uniform  1/N.

Key properties compared to standard resampling:
  - DIFFERENTIABLE w.r.t. particles X and weights w (and hence theta).
  - No duplicate particles: each resampled particle is a smooth weighted
    combination of the original particles.
  - Computational cost  O(N^2)  per Sinkhorn iteration (vs. O(N log N) for
    multinomial), but the Sinkhorn algorithm converges quickly.

Module structure
----------------
  regularised_ot_resample.py      <- this file (public API)
  regularized_optimal_transport/
      ot_utils.py                 <- cost matrices, softmin, scaling
      sinkhorn.py                 <- Sinkhorn algorithm (Algorithm 2)
      plan.py                     <- transport matrix computation (Algorithm 3)
"""

import abc

import attr
import tensorflow as tf

from dpf.base import State
from dpf.resampling.base import ResamplerBase
from dpf.resampling.differentiable.regularized_optimal_transport.plan import transport


@tf.function
def resample(tensor, new_tensor, flags):
    """Conditionally select between original and resampled tensors.

    Used to apply resampling only to batch elements where the ESS criterion
    triggered (flags = True), leaving other batch elements unchanged.

    Parameters
    ----------
    tensor : tf.Tensor
        Original tensor (particles, weights, or log-weights).
    new_tensor : tf.Tensor
        Resampled tensor (same shape).
    flags : tf.Tensor, shape [B]
        Per-batch boolean flags.  True = use new_tensor, False = keep tensor.

    Returns
    -------
    tf.Tensor
        Element-wise selection:  flags ? new_tensor : tensor.
    """
    ndim = len(tensor.shape)
    shape = [-1] + [1] * (ndim - 1)
    return tf.where(tf.reshape(flags, shape), new_tensor, tensor)


def apply_transport_matrix(state, transport_matrix, flags):
    """Apply the OT transport matrix to produce resampled particles (Eq. 13).

    Computes the transported particles:

        X_tilde = P^OT_epsilon @ X          (matrix multiplication)

    and resets weights to uniform  1/N,  since the transport already
    redistributes mass according to the original weights.

    The ``flags`` tensor controls per-batch conditional application: only
    batch elements where the resampling criterion triggered are updated.

    Parameters
    ----------
    state : State
        Current particle filter state containing particles, weights,
        and log_weights.
    transport_matrix : tf.Tensor, shape [B, N, N]
        The regularised OT transport matrix  P^OT_epsilon.
    flags : tf.Tensor, shape [B]
        Per-batch boolean resampling flags from NeffCriterion.

    Returns
    -------
    State
        New state with transported particles and uniform weights (where
        flags = True), original state preserved (where flags = False).
    """
    float_n_particles = tf.cast(state.n_particles, state.particles.dtype)

    # X_tilde = P^OT_epsilon @ X   (Eq. 13, with N factor in P)
    transported_particles = tf.linalg.matmul(transport_matrix, state.particles)

    # After OT transport, weights become uniform: w_i = 1/N
    uniform_log_weights = -tf.math.log(float_n_particles) * tf.ones_like(state.log_weights)
    uniform_weights = tf.ones_like(state.weights) / float_n_particles

    # Conditionally apply based on resampling flags
    resampled_particles = resample(state.particles, transported_particles, flags)
    resampled_weights = resample(state.weights, uniform_weights, flags)
    resampled_log_weights = resample(state.log_weights, uniform_log_weights, flags)

    return attr.evolve(state, particles=resampled_particles, weights=resampled_weights,
                       log_weights=resampled_log_weights)


class RegularisedOTResampler(ResamplerBase, metaclass=abc.ABCMeta):
    """Differentiable Ensemble Transform (DET) resampler.

    Replaces standard (non-differentiable) multinomial resampling with
    entropy-regularised optimal transport solved via the Sinkhorn algorithm.

    This implements Algorithm 3 of Corenflos et al. (2021):

        Algorithm 3  (DET Resampling)
        ─────────────────────────────
        Input:  EnsembleTransform(X, w, N)
        1.  f, g  <-  Potentials(w, 1/N * 1, X, X)       [Sinkhorn, Alg. 2]
        2.  for i in [N], j in [N]:
                p^OT_{eps,i,j} = (w_i / N) exp( (f_i + g_j - c_{i,j}) / eps )
        3.  Return  X_tilde = N * P^OT_eps * X            [Eq. 13]

    The resulting resampling step is fully differentiable w.r.t. both the
    particles X and the importance weights w, enabling gradient-based
    optimisation of model parameters theta through the particle filter.

    Parameters
    ----------
    epsilon : float
        Entropy regularisation strength  (epsilon > 0).  Controls the
        trade-off between the exact OT solution (epsilon -> 0, sharper
        transport) and a smoother, more diffuse plan (large epsilon).
        The paper recommends  epsilon = 0.5  as a good default.
    scaling : float, default 0.75
        Epsilon-scaling factor for the multi-scale Sinkhorn schedule.
        The algorithm starts at a large  epsilon_0 ~ diameter^2  and
        geometrically decreases toward  epsilon  by multiplying by
        scaling^2  each iteration.
    max_iter : int, default 100
        Maximum number of Sinkhorn iterations.
    convergence_threshold : float, default 1e-3
        Stopping criterion: the Sinkhorn loop terminates when
        max|f_new - f_old| < threshold  for all batch elements.
    """

    DIFFERENTIABLE = True

    def __init__(self, epsilon, scaling=0.75, max_iter=100, convergence_threshold=1e-3,
                 name='RegularisedTransform', **_kwargs):
        self.convergence_threshold = tf.cast(convergence_threshold, tf.float64)
        self.max_iter = tf.cast(max_iter, tf.dtypes.int32)
        self.epsilon = tf.cast(epsilon, tf.float64)
        self.scaling = tf.cast(scaling, tf.float64)
        super(RegularisedOTResampler, self).__init__(name=name)

    def apply(self, state, flags, seed=None):
        """Apply DET resampling to the particle filter state.

        Computes the entropy-regularised OT transport matrix and applies
        it to produce resampled particles with uniform weights.

        Parameters
        ----------
        state : State
            Current particle filter state.
            - state.particles:   [B, N, D]  particle positions  X_t
            - state.log_weights: [B, N]     log importance weights  log(w_t)
        flags : tf.Tensor, shape [B]
            Per-batch boolean flags from NeffCriterion.
            True = resample this batch element, False = skip.
        seed : optional
            Unused (OT resampling is deterministic given particles and weights).

        Returns
        -------
        State
            Resampled state:
            - particles:   X_tilde = P^OT_epsilon @ X  (transported, Eq. 13)
            - weights:     1/N  (uniform)
            - log_weights: -log(N)  (uniform)
        """
        particles = state.particles

        # Algorithm 3: compute transport matrix via Sinkhorn
        transport_matrix = transport(particles, state.log_weights, self.epsilon, self.scaling,
                                     self.convergence_threshold, self.max_iter, state.n_particles)

        # Apply transport: X_tilde = P^OT_epsilon @ X, reset weights to 1/N
        return apply_transport_matrix(state, transport_matrix, flags)
