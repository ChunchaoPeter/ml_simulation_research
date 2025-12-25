import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from itertools import permutations

tfd = tfp.distributions

from acoustic_function import (
    state_transition,
    observation_model,
    compute_observation_jacobian,
    observation_model_general
)



def compute_lambda_steps(n_lambda, lambda_ratio):
    """
    Compute exponentially spaced lambda steps.
    
    Args:
        n_lambda: Number of lambda steps (typically 20)
        lambda_ratio: Ratio for exponential spacing (typically 1.2)
    
    Returns:
        lambda_steps: Step sizes ε_j, shape (n_lambda,)
        lambda_values: Cumulative lambda values λ_j, shape (n_lambda,)
    """
    q = lambda_ratio
    n = n_lambda
    
    # Initial step size: ε_1 = (1-q)/(1-q^n)
    epsilon_1 = (1 - q) / (1 - q**n)
    
    # Step sizes: ε_j = ε_1 * q^(j-1) for j=1,...,n
    step_sizes = [epsilon_1 * (q**j) for j in range(n)]
    lambda_steps = tf.constant(step_sizes, dtype=tf.float32)
    
    # Cumulative lambda values: λ_j = Σ_{i=1}^j ε_i
    lambda_values = tf.cumsum(lambda_steps)
    
    return lambda_steps, lambda_values


def initialize_particles(model_params, n_particle):
    """
    Initialize particles from Gaussian prior.

    Algorithm Lines 1-2:
      1  Draw {x_0^i}_{i=1}^N from the prior p_0(x)
      2  Set {w_0^i}_{i=1}^N = 1/Np

    Args:
        model_params: Dictionary from initialize_acoustic_model()
        n_particle: Number of particles

    Returns:
        particles: Initial particles, shape (state_dim, n_particle)
        weights: Initial weights, shape (n_particle,)
        m0: Random initial mean, shape (state_dim, 1)
        P0: Initial covariance, shape (state_dim, state_dim)
    """
    state_dim = model_params['state_dim']
    x0 = model_params['x0_initial_target_states']  # (state_dim, 1)
    n_targets = model_params['n_targets']
    sim_area_size = model_params['sim_area_size']

    # Initial uncertainty from paper
    sigma0_single = tf.constant(
        [0.1**0.5, 0.1**0.5, 0.0005**0.5, 0.0005**0.5],
        dtype=tf.float32
    )    
    sigma0 = tf.tile(sigma0_single, [n_targets])
    # Line 2: P_0 = diag(σ_0^2)
    P0 = tf.linalg.diag(tf.square(sigma0))

    # Sample random mean m0 and ensure it's within bounds
    out_of_bound = True
    while out_of_bound:
        noise = tf.random.normal((state_dim, 1), dtype=tf.float32)
        m0 = x0 + tf.expand_dims(sigma0, 1) * noise

        # Check if all target positions are within surveillance region
        x_positions = m0[0::4, 0]  # indices 0, 4, 8, 12, ... (x positions)
        y_positions = m0[1::4, 0]  # indices 1, 5, 9, 13, ... (y positions)

        # Check bounds: all positions should be in [0, sim_area_size]
        x_in_bounds = tf.reduce_all(x_positions >= 0.0) and tf.reduce_all(x_positions <= sim_area_size)
        y_in_bounds = tf.reduce_all(y_positions >= 0.0) and tf.reduce_all(y_positions <= sim_area_size)

        if x_in_bounds and y_in_bounds:
            out_of_bound = False

    # Sample particles around m0
    noise = tf.random.normal((state_dim, n_particle), dtype=tf.float32)
    particles = m0 + tf.expand_dims(sigma0, 1) * noise
    
    # Initialize uniform weights
    weights = tf.ones(n_particle, dtype=tf.float32) / n_particle

    return particles, weights, m0, P0


def propagate_particles(particles, model_params, no_noise=False):
    """
    Propagate particles through motion model.
    
    Algorithm Line 7 (LEDH) / Line 6 (EDH):
      Propagate particles η_0^i = g_k(x_{k-1}^i, v_k)
    
    Equation:
      x_k = Φ * x_{k-1} + w_k,  where w_k ~ N(0, Q)
    
    Args:
        particles: Current particles, shape (state_dim, n_particle)
        model_params: Dictionary with 'Phi' and 'Q'
    
    Returns:
        particles_pred: Predicted particles, shape (state_dim, n_particle)
    """
    state_dim = model_params['state_dim']
    Phi = model_params['Phi']  # State transition matrix
    Q = model_params['Q']      # Process noise covariance
    n_particle = tf.shape(particles)[1]
    
    # Linear propagation: x_k = Φ * x_{k-1}
    particles_pred = tf.matmul(Phi, particles)
    
    # no nosie
    if no_noise:
        return particles_pred

    # Add process noise: w_k ~ N(0, Q)
    Q_chol = tf.linalg.cholesky(Q)
    noise = tf.matmul(Q_chol, tf.random.normal((state_dim, n_particle), dtype=tf.float32))
    
    return particles_pred + noise

def estimate_covariance(particles):
    """
    Estimate covariance from particles.
    
    Algorithm Line 5 (LEDH) / Line 4 (EDH): P prediction
    
    Equation:
      P = (1/(N-1)) * Σ (x_i - x̄)(x_i - x̄)^T
    
    Args:
        particles: Particles, shape (state_dim, n_particle)
    
    Returns:
        P: Covariance matrix, shape (state_dim, state_dim)
    """
    # Compute mean
    mean = tf.reduce_mean(particles, axis=1, keepdims=True)
    
    # Center particles
    centered = particles - mean
    
    # Covariance: P = (1/(N-1)) * centered @ centered^T
    n_particles = tf.cast(tf.shape(particles)[1], tf.float32)
    P = tf.matmul(centered, tf.transpose(centered)) / (n_particles - 1.0)
    
    # Add small regularization for numerical stability
    P = P + 1e-6 * tf.eye(tf.shape(P)[0], dtype=tf.float32)
    
    return P


def particle_estimate(log_weights, particles):
    """
    Form estimate based on weighted set of particles
    
    Args:
        log_weights: logarithmic weights [N x 1] or [N,] tensor
        particles: the state values of the particles [dim x N] 
                   (state dimension x number of particles)
    
    Returns:
        estimate: weighted estimate of the state [dim x 1] or [dim,] tensor
        ml_weights: normalized weights [N x 1] or [N,] tensor
    """
    
    # Ensure log_weights is a 1D tensor
    log_weights = tf.reshape(log_weights, [-1])
    
    # Normalize log_weights by subtracting the maximum (for numerical stability)
    log_weights = log_weights - tf.reduce_max(log_weights)
    
    # Compute weights
    ml_weights = tf.exp(log_weights)
    
    # Normalize weights to sum to 1
    ml_weights = ml_weights / tf.reduce_sum(ml_weights)
    
    # Reshape ml_weights for matrix multiplication [N,] -> [N, 1]
    ml_weights_col = tf.reshape(ml_weights, [-1, 1])
    
    # Compute weighted estimate: particles @ ml_weights
    # particles shape: [dim, N], ml_weights_col shape: [N, 1]
    # result shape: [dim, 1]
    estimate = tf.matmul(particles, ml_weights_col)
    
    # Squeeze to remove the last dimension
    estimate = tf.squeeze(estimate, axis=-1)
    
    return estimate, ml_weights


# line 4 - 8
def propagateAndEstimatePriorCovariance(x_est_prev, vg, model_params):

    """
    This is kind of initialization for the propagate and estimate prior covraince.
    We obtain different values that used for next steps, for example, we use them calcualte weights
    
    Returns:
        vg: a list of parameters
    """

    particles = vg['particles']
    
    # Algorithm Line 4: Global covariance prediction
    # (Could use EKF prediction here; for simplicity, estimate from particles)
    vg['PP'] = estimate_covariance(particles) 

    # Algorithm Lines 5-8: Propagate particles
    xp_prop_deterministic= propagate_particles(particles, model_params, no_noise=True)
    xp_prop= propagate_particles(particles, model_params)


    # Algorithm Line 9: Calculate global mean trajectory
    eta_bar_mu_0 = state_transition(x_est_prev, model_params, no_noise=True) ## This is ita_bar
    
    
    # save the results in a dictionary
    vg['mu_0'] = eta_bar_mu_0
    vg['xp_prop_deterministic'] = xp_prop_deterministic
    vg['xp_prop'] = xp_prop
    vg['xp_auxiliary_individual'] = eta_bar_mu_0


    vg['particles_pre'] = particles
    vg['particles'] = xp_prop

    xp_m, ml_weights = particle_estimate(vg['logW'], vg['particles'])
    vg['particles_m'] = xp_m
    vg['ml_weights'] = ml_weights

    return vg


# compute A and b
def compute_flow_parameters(linearization_point, x_bar_mu_0, P, measurement, lam, model_params):
    """
    Compute flow parameters A and b for particle x.
    
    Algorithm Line 10:
      10  Calculate A and b from (8) and (9) using P_{k|k-1}, x̄ and H_x
    
    Equations:
      A(λ) = -1/2 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
      b(λ) = (I + 2λA) * [(I + λA) * P*H^T*R^{-1}*(z-e) + A*x̄]
    
    Args:
        linearization_point: Particle position (state_dim, 1) - used for linearization
        x_bar_mu_0: Mean trajectory (state_dim, 1) - used in b computation
        P: Covariance (state_dim, state_dim) - should be P_{k|k-1}
        measurement: Current measurement z, shape (n_sensor, 1)
        lam: Current lambda value (scalar)
        model_params: Dictionary with observation model
    
    Returns:
        A: Flow matrix, shape (state_dim, state_dim)
        b: Flow vector, shape (state_dim,)
    """
    state_dim = tf.shape(P)[0]
    R = model_params['R']
    
    # Algorithm Line 9: Calculate H_x by linearizing at x̄_k (or x_i for local)
    H = compute_observation_jacobian(linearization_point, model_params)
    
    # Compute h(x̄) and linearization residual: e = h(x̄) - H*x̄
    h_x_bar = observation_model(linearization_point, model_params, no_noise=True)
    h_x_bar = tf.squeeze(h_x_bar, axis=1)  # (n_sensor,)
    e = h_x_bar - tf.linalg.matvec(H, tf.squeeze(linearization_point, axis=1))
    
    # Compute H*P*H^T
    HPHt = tf.matmul(tf.matmul(H, P), tf.transpose(H))
    
    # Innovation covariance: S = λ*H*P*H^T + R
    # Add regularization for numerical stability
    n_sensor = tf.shape(R)[0]
    regularization = 1e-6 * tf.eye(n_sensor, dtype=tf.float32)
    S = lam * HPHt + R + regularization
    
    # Use Cholesky decomposition instead of direct inversion (more stable)
    S_chol = tf.linalg.cholesky(S)
    
    # Compute P*H^T
    PHt = tf.matmul(P, tf.transpose(H))
    
    # Equation (8): Flow matrix A = -0.5 * P*H^T * S^{-1} * H
    # Using Cholesky solve: S^{-1} * H = solve(S, H)
    S_inv_H = tf.linalg.cholesky_solve(S_chol, H)
    A = -0.5 * tf.matmul(PHt, S_inv_H)
    
    # Innovation: z - e
    innovation = tf.squeeze(measurement, axis=1) - e
    
    # Compute R^{-1}*(z - e) using Cholesky decomposition
    R_regularized = R + regularization
    R_chol = tf.linalg.cholesky(R_regularized)
    R_inv_innov = tf.linalg.cholesky_solve(R_chol, tf.expand_dims(innovation, 1))
    R_inv_innov = tf.squeeze(R_inv_innov, axis=1)
    
    # Identity matrix
    I = tf.eye(state_dim, dtype=tf.float32)
    
    # (I + λA) and (I + 2λA)
    I_plus_lam_A = I + lam * A
    I_plus_2lam_A = I + 2 * lam * A
    
    # Equation (9): Flow vector b
    # First term: (I + λA) * P*H^T * R^{-1}*(z - e)
    term1 = tf.linalg.matvec(tf.matmul(I_plus_lam_A, PHt), R_inv_innov)
    # Second term: A * x̄
    term2 = tf.linalg.matvec(A, tf.squeeze(x_bar_mu_0, axis=1))
    # Combined: b = (I + 2λA) * [term1 + term2]
    b = tf.linalg.matvec(I_plus_2lam_A, term1 + term2)
    
    return A, b


# line 8 - 18
def particle_flow_edh(vg, model_params, measurement, lambda_steps, lambda_values):
    """
    Migrate particles from prior to posterior using EDH flow (global linearization).

    Algorithm Lines 10-18 (EDH):
      10  Set λ = 0
      11  for j = 1, ..., N_λ do
      12    Set λ = λ + ε_j
      13    Calculate A_j(λ) and b_j(λ) with linearization at η̄
      14    Migrate η̄
      15    for i = 1, ..., Np do
      16      Migrate particles
      17    endfor
      18  endfor

    Args:
        vg: include the relevant parameters
        measurement: Current measurement, shape (n_sensor, 1)
        P_pred: Prior covariance P, shape (state_dim, state_dim)
        lambda_steps: Step sizes ε_j, shape (n_lambda,)
        lambda_values: Cumulative lambda values λ_j, shape (n_lambda,)
        eta_bar_mu_0: Mean trajectory η̄_0, shape (state_dim, 1)
        model_params: Dictionary with observation model

    Returns:
        particles_flowed: Updated particles, shape (state_dim, n_particle)
    """

    particles = vg['particles']
    particles_mean = vg['particles_m']
    Pvariance_pred = vg['PP']
    n_lambda = len(lambda_steps)
    log_weights = vg['logW']
    # Initialize auxiliary trajectory for global linearization
    eta_bar_mu_0 = vg['mu_0']
    eta_bar = vg['xp_auxiliary_individual']


    # Algorithm Line 10: Set λ = 0
    # Algorithm Line 11: for j = 1, ..., N_λ do
    for j in range(n_lambda):
        # Algorithm Line 12: Set λ = λ + ε_j
        epsilon_j = lambda_steps[j]   # Step size
        lambda_j = lambda_values[j]   # Current lambda value
        
        # Algorithm Line 13: Compute A, b ONCE at global mean η̄
        A, b = compute_flow_parameters(eta_bar, eta_bar_mu_0, Pvariance_pred, measurement,
                                        lambda_j, model_params)
        
        
        # Algorithm Line 14: Migrate η̄
        slope_bar = tf.linalg.matvec(A, tf.squeeze(particles_mean)) + b # It could be wrong even if it match the orginal matlab code
        eta_bar = eta_bar + epsilon_j * tf.expand_dims(slope_bar, 1)

        # Algorithm Lines 15-17: Migrate all particles using the same A, b
        slopes = tf.matmul(A, particles) + tf.expand_dims(b, 1)
        particles = particles + epsilon_j * slopes

        particles_mean, _ = particle_estimate(log_weights, particles)
    
    return particles

############ calcuate log density proposal and prir

def log_proposal_density(xp_prop, xp_prop_deterministic, Q, log_jacobian_det_sum):
    """
    Calculate the log proposal density after particle flow.
    
    MATLAB Reference: log_proposal_density.m lines 23, 26
    
    This function exactly mirrors the MATLAB implementation:
    -------------------------------------------------------
    MATLAB code:
        log_proposal = loggausspdf(vg.xp_prop, vg.xp_prop_deterministic, ps.propparams.Q);
        log_proposal = log_proposal - log_jacobian_det_sum;
    -------------------------------------------------------
    
    The proposal density with change of variables:
        q(x_k|x_{k-1}, z_k) = p(η_0|x_{k-1}) / |det(∂T/∂η_0)|
    
    In log space:
        log q = log p(η_0|x_{k-1}) - log|det(∂T/∂η_0)|
    
    Args:
        xp_prop: Propagated particles WITH noise η_0, shape (state_dim, n_particle)
        xp_prop_deterministic: Deterministic propagation Φ·x_{k-1}, shape (state_dim, n_particle)
        Q: Process noise covariance, shape (state_dim, state_dim)
        log_jacobian_det_sum: Sum of log Jacobian determinants, shape (n_particle,)
    
    Returns:
        log_proposal: Log proposal density for each particle, shape (n_particle,)
    """
    # Line 23: Calculate base proposal density
    
    mean = tf.transpose(xp_prop_deterministic)   # (n_particle, state_dim)
    xp   = tf.transpose(xp_prop)     # (n_particle, state_dim)
    Q = Q  

    scale_tril = tf.linalg.cholesky(Q)
    dist = tfd.MultivariateNormalTriL(
        loc=mean,
        scale_tril=scale_tril
    )
    log_proposal = dist.log_prob(xp)  # (n_particle,) 
    
    # Line 26: Subtract Jacobian determinant
    log_proposal = log_proposal - log_jacobian_det_sum


    return log_proposal


def log_process_density(xp, xp_prop_deterministic, Q):
    """
    Calculate the log process density p(x_k|x_{k-1}).

    
    For the acoustic example:
        p(x_k|x_{k-1}) = N(x_k; Φ·x_{k-1}, Q)
    
    Args:
        xp: Current particles (after flow), shape (state_dim, n_particle)
        xp_prop_deterministic: Deterministic propagation Φ·x_{k-1}, shape (state_dim, n_particle)
        Q: Process noise covariance, shape (state_dim, state_dim)
    
    Returns:
        log_prior: Log process density for each particle, shape (n_particle,)
    """
    
    mean = tf.transpose(xp_prop_deterministic)   # (n_particle, state_dim)
    xp   = tf.transpose(xp)     # (n_particle, state_dim)
    Q = Q 

    scale_tril = tf.linalg.cholesky(Q)

    dist = tfd.MultivariateNormalTriL(
        loc=mean,                 # shape (n_particle, di)
        scale_tril=scale_tril       # shape (state_dim, state_dim), broadcasted
    )
    log_prior = dist.log_prob(xp)  # (n_particle,) 
    
    return log_prior

def log_likehood_density(particles_flowed, measurement, model_params):
    """
    Compute likelihood p(z_k|x_k) for Gaussian measurement model.
    
    Equation:
      p(z_k|x_k) = N(z_k; h(x_k), R)
    
    Args:
        x: State, shape (state_dim, 1)
        measurement: Measurement z_k, shape (n_sensor, 1)
        model_params: Dictionary with observation model and R
    
    Returns:
        likelihood: Scalar likelihood value
    """
    R = model_params['R']
    z_pre = observation_model_general(particles_flowed, model_params, no_noise=True)
    residual = measurement - z_pre
    residual_t = tf.transpose(residual)     # (n_particle, state_dim)
    mean_zeros = tf.zeros(residual_t.shape, dtype=tf.float32)   # (n_particle, state_dim)

    scale_tril = tf.linalg.cholesky(R)

    dist = tfd.MultivariateNormalTriL(
        loc=mean_zeros,                 # shape (n_particle, di)
        scale_tril=scale_tril       # shape (state_dim, state_dim), broadcasted
    )
    log_likelihood = dist.log_prob(residual_t)  # (n_particle,) 
    return log_likelihood

def multinomial_resample(weights):
    """
    Multinomial resampling.

    Draws ancestry vector A^{1:N} where A^i ~ Cat(w^1,...,w^N)

    Args:
        weights: Normalized particle weights (N,)

    Returns:
        indices: Resampled particle indices (N,)
    """
    # Sample from categorical distribution
    # This implements: A^i ~ Cat(w^1,...,w^N)
    # Use log probabilities for numerical stability
    logits = tf.math.log(weights + 1e-10)

    # tf.random.categorical expects shape [batch_size, num_classes]
    # and returns [batch_size, num_samples]
    # We want to sample num_particles times from one distribution
    logits_2d = tf.reshape(logits, [1, -1])
    indices = tf.random.categorical(logits_2d, weights.shape[0], dtype=tf.int32)

    # Reshape from [1, num_particles] to [num_particles]
    indices = tf.reshape(indices, [-1])

    # This method is much faster than using tfd.Categorical(probs=weights) 
    return indices


def force_resample(particles, weights):
    """
    Force resampling of all particles, ignoring the effective sample size.

    Args:
        particles: Current particles (state_dim, num_particles)
        weights: Current weights (num_particles,)

    Returns:
        particles: Resampled particles (state_dim, num_particles)
        weights: Uniform weights after resampling
        eff: Compute effective sample size N_eff
    """

    # Always resample
    indices = multinomial_resample(weights)

    eff = 1.0 / tf.reduce_sum(weights ** 2)

    # Resample particles
    particles = tf.gather(particles, indices, axis=1)

    # Reset weights to uniform
    weights = tf.ones(particles.shape[1], dtype=tf.float32) / particles.shape[1]
    

    return particles, weights, eff


def correctoinAndCalculateWeights(particles_flowed, 
                                  vg,
                                  measurement, 
                                  log_jacobian_det_sum, 
                                  model_params):
    """
    Weight update formula:
        w_k ∝ [p(x_k|x_{k-1}) * p(z_k|x_k) / q(x_k|x_{k-1},z_k)] * w_{k-1}
    
    In log space:
        log w_k = log p(x_k|x_{k-1}) + log p(z_k|x_k) - log q(...) + log w_{k-1}
    
    Args:
        particles_flowed: Particles after flow (x_k), shape (state_dim, n_particle)
        xp_prop: Propagated particles WITH noise (η_0), shape (state_dim, n_particle)
        xp_prop_deterministic: Deterministic propagation (Φ·x_{k-1}), shape (state_dim, n_particle)
        weights_prev: Previous weights, shape (n_particle,)
        measurement: Current measurement, shape (n_sensor, 1)
        log_jacobian_det_sum: Sum of log Jacobian determinants, shape (n_particle,)
        model_params: Dictionary with model parameters
    
    Returns:
        weights_updated: Updated normalized weights, shape (n_particle,)
    """


    Q = model_params['Q']
    xp_prop_deterministic = vg['xp_prop_deterministic'] 
    xp_prop = vg['xp_prop']
    log_weights = vg['logW']

    # Line 16: Calculate log proposal density
    # log_proposal = log p(η_0|x_{k-1}) - log_jacobian_det_sum (n_particle, )
    log_proposal = log_proposal_density(xp_prop, xp_prop_deterministic, Q, log_jacobian_det_sum)
    
    # Line 18: Calculate log process/prior density
    # log_prior = log p(x_k|x_{k-1})
    log_prior = log_process_density(particles_flowed, xp_prop_deterministic, Q)
    
    # Line 20: Calculate log likelihood
    # llh = log p(z_k|x_k)
    llh = log_likehood_density(particles_flowed, measurement, model_params)
    
    # Line 22: Weight update
    # log w_k = log_prior + llh - log_proposal + log w_{k-1}
    log_weights = log_prior + llh - log_proposal + log_weights
        
    # Line 23: Normalize by subtracting max
    log_weights = log_weights - tf.reduce_max(log_weights)
    
    # Convert back from log space
    partilcles_mean, weights = particle_estimate(log_weights, particles_flowed)

    return weights, partilcles_mean



def plot_true_vs_estimated_trajectories(
    model_params,
    ground_truth,
    estimates,
    title="EDH Filter: True vs Estimated Trajectories",
    estimate_label="EDH",
    figsize=(12, 10),
    colors=None
):
    """
    Plot ground-truth and estimated target trajectories along with sensor positions.

    Args:
        model_params (dict):
            Must contain:
                - 'sensor_positions': Tensor or ndarray, shape (n_sensors, 2)
                - 'n_targets': int
                - 'sim_area_size': float
        ground_truth (Tensor or ndarray):
            Shape (state_dim, T)
        estimates (Tensor or ndarray):
            Shape (state_dim, T)
        title (str):
            Plot title
        estimate_label (str):
            Label for estimated trajectories (e.g., 'EDH', 'PFPF')
        figsize (tuple):
            Figure size
        colors (list or None):
            List of colors for each target
    """

    # Convert to NumPy if needed
    sensors = model_params['sensor_positions']
    if hasattr(sensors, "numpy"):
        sensors = sensors.numpy()

    gt = ground_truth.numpy() if hasattr(ground_truth, "numpy") else ground_truth
    est = estimates.numpy() if hasattr(estimates, "numpy") else estimates

    n_targets = model_params['n_targets']
    sim_area_size = model_params['sim_area_size']

    if colors is None:
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'cyan']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot sensors
    ax.scatter(
        sensors[:, 0], sensors[:, 1],
        c='blue', marker='o', s=100,
        label='Sensors', alpha=0.6, zorder=5
    )

    for i in range(n_targets):
        x_idx = i * 4
        y_idx = i * 4 + 1
        color = colors[i % len(colors)]

        # Ground truth
        x_true = gt[x_idx, :]
        y_true = gt[y_idx, :]
        ax.plot(
            x_true, y_true,
            '-', color=color, linewidth=2.5,
            label=f'Target {i+1} (True)', alpha=0.7
        )

        # Estimates
        x_est = est[x_idx, :]
        y_est = est[y_idx, :]
        ax.plot(
            x_est, y_est,
            '--', color=color, linewidth=2.5,
            label=f'Target {i+1} ({estimate_label})',
            alpha=0.9
        )

        # Mark start position
        ax.plot(
            x_true[0], y_true[0],
            'o', color=color, markersize=12,
            markeredgecolor='black', markeredgewidth=2
        )

    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    ax.set_xlim([0, sim_area_size])
    ax.set_ylim([0, sim_area_size])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

################################## ------ Evaluation matrix -------##################################################

def compute_omat(true_states, estimated_states, num_targets, n_target, p=1.0):
    """
    Compute OMAT metric using pure TensorFlow.

    Args:
        true_states: TensorFlow tensor, shape (state_dim,) or (state_dim, T)
        estimated_states: TensorFlow tensor, same shape
        num_targets: number of target 
        n_target: Dimension of each target's state
        p: Exponent (default: 1.0)

    Returns:
        OMAT error (scalar tensor)
    """
    # Handle single time step
    if len(true_states.shape) == 1:
        true_states = tf.expand_dims(true_states, 1)
    if len(estimated_states.shape) == 1:
        estimated_states = tf.expand_dims(estimated_states, 1)

    # Build cost matrix: cost[i,j] = distance from true target i to estimated target j
    cost_matrix = []
    for i in range(num_targets):
        row = []
        for j in range(num_targets):
            # Extract all n_target dimensions for target i and j
            # This handles any dimension: [x], [x,y], [x,y,z], [x,y,vx,vy], etc.
            # Extract all n_target dimensions
            true_pos = true_states[i*n_target:(i+1)*n_target]
            est_pos = estimated_states[j*n_target:(j+1)*n_target]

            # Euclidean distance across all dimensions
            diff = true_pos - est_pos
            dist = tf.sqrt(tf.reduce_sum(diff**2, axis=0))  # Sum over all n_target dimensions
            avg_dist = tf.reduce_mean(dist) ## aim to reduce the dimension 
            row.append(avg_dist ** p) ## calculate true pos for all different target of estimate values 
        cost_matrix.append(row) ## [i, j] i means the true target, j mean the estimate target

    cost_matrix = tf.stack([tf.stack(row) for row in cost_matrix])

    # ============================================================
    # MINIMIZATION: Find optimal assignment
    # ============================================================
    # This implements: min_{π∈Π} Σ d(x_c, x̂_{π(c)})^p
    #
    # We try ALL permutations and find the one with MINIMUM total distance.
    #
    # Example: perm = (1, 0, 3, 2) means:
    #   True target 0 → Estimated target 1
    #   True target 1 → Estimated target 0
    #   True target 2 → Estimated target 3
    #   True target 3 → Estimated target 2
    # ============================================================

    min_cost = tf.constant(float('inf'), dtype=tf.float32)

    for perm in permutations(range(num_targets)):
        # Calculate total cost for this assignment
        cost = tf.reduce_sum([cost_matrix[i, perm[i]] for i in range(num_targets)])

        # *** MINIMIZATION HAPPENS HERE ***
        # Keep the smallest cost across all permutations
        min_cost = tf.minimum(min_cost, cost)

    # Apply OMAT formula: (min_cost / num_targets)^(1/p)
    return (min_cost / num_targets) ** (1.0 / p)


def compute_omat_per_timestep(true_states, estimated_states, num_targets, n_target, p=1.0):
    """Compute OMAT at each time step using pure TensorFlow."""
    T = tf.shape(true_states)[1]
    errors = tf.TensorArray(dtype=tf.float32, size=T)

    for t in tf.range(T):
        error = compute_omat(true_states[:, t], estimated_states[:, t],num_targets, n_target, p)
        errors = errors.write(t, error)

    return errors.stack()