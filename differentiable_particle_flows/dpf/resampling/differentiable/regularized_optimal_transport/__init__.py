"""
Entropy-regularised optimal transport for differentiable particle resampling.

This package implements the Sinkhorn-based OT resampling scheme from:

    Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021).
    Differentiable Particle Filtering via Entropy-Regularized Optimal
    Transport. ICML 2021 (PMLR 139).

Modules
-------
ot_utils
    Utility functions: pairwise cost c_{i,j} = (1/2)||x_i - x_j||^2,
    the softmin operator T_epsilon (Eq. 11), and particle-cloud scaling
    delta(X) = sqrt(d_x) max_k std_i(X_{i,k}) (Section 3.2).

sinkhorn
    Stabilised Sinkhorn algorithm with epsilon-scaling and gradient
    stitching (Algorithm 2, adapted from Feydy et al. 2019).
    Solves for the dual potentials (f*, g*) of the regularised OT problem.

plan
    Transport matrix recovery from Sinkhorn potentials (Eq. 12) and the
    full DET resampling pipeline (Algorithm 3) with custom gradient support.
"""
