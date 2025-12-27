"""
PFPF (Particle Filtering with Particle Flow) Implementation

This module implements the PFPF algorithm based on:
J. Li and M. J. Coates, "Particle filtering with invertible particle flow,"
IEEE Trans. Signal Process., vol. 65, no. 15, pp. 4102-4116, Aug. 2017.

The PFPF extends the EDH filter by adding:
- Importance weight updates accounting for the flow transformation
- Resampling based on effective sample size
- Proper proposal density calculation
"""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, Dict, Callable, Optional
from edh import EDHFilter


tfd = tfp.distributions


class PFPF_EDH(EDHFilter):
    """
    Particle Filtering with Particle Flow (PFPF)

    Extends EDHFilter with importance weighting and resampling.
    Uses particle flow as a proposal distribution with proper weight updates.

    Additional Attributes:
        weights: Particle weights, shape (n_particle,)
        log_weights: Log particle weights, shape (n_particle,)
        particles_pred: Propagated particles WITH noise
        particles_pred_deterministic: Propagated particles WITHOUT noise
        particles_mean: Mean of particle distribution
        mu_0: Mean trajectory for linearization
        auxiliary_trajectory: Auxiliary trajectory point
        P_pred: Prior covariance estimate
    """

    def __init__(
        self,
        observation_jacobian: Callable,
        observation_model: Callable,
        observation_model_general: Callable,
        state_transition: Callable,
        n_particle: int = 100,
        n_lambda: int = 20,
        lambda_ratio: float = 1.2,
        use_local: bool = False,
        use_ekf: bool = False,
        ekf_filter: Optional['ExtendedKalmanFilter'] = None,
        verbose: bool = True
    ):
        """
        Initialize PFPF filter.

        Args:
            observation_jacobian: Callable that computes the observation Jacobian
            observation_model: Callable that maps state → observation
            observation_model_general: Callable that maps all state → all observation
            n_particle: Number of particles (default: 100)
            n_lambda: Number of lambda steps (default: 20)
            lambda_ratio: Exponential spacing ratio (default: 1.2)
            use_local: Use local linearization (default: False)
            use_ekf: Use EKF for covariance tracking (default: False)
            ekf_filter: Pre-configured EKF filter instance
            verbose: Print progress information (default: True)
        """
        super().__init__(
            observation_jacobian, observation_model,
            n_particle, n_lambda, lambda_ratio,
            use_local, use_ekf, ekf_filter, verbose
        )


        self.state_transition = state_transition
        self.observation_model_general = observation_model_general

        # Additional PFPF state
        self.weights = None
        self.log_weights = None
        # Intermediate computation states 
        self.particles_pred = None  # Propagated particles WITH noise
        self.particles_pred_deterministic = None  # Propagated WITHOUT noise, that is used for log prior and proposed
        self.particles_previous = None # Previous step for particles
        self.particles_mean = None  # Mean of particles
        self.mu_0 = None  # Mean trajectory η̄_μ0, this is strong connection with particles_mean
        self.auxiliary_trajectory = None  # Auxiliary trajectory for linearization. It is used for calculate A, b
        self.P_pred = None  # Prior covariance P_{k|k-1}, this is also can be called prediction covariance % Kalman variables: preditive varianace
        self.M = None # the original mean, this one is used to calcuate the mu_0. % Kalman variables: mean

        ## There are many parameters that can be used interchangeably.
        ## I have not reduced the duplicated variables, as keeping them
        ## makes the code easier to follow. self.P is the posterior variance. % Kalman variables: updated varianace

    def initialize(self, model_params: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        In PFPF_EDH algorithm: corresponds to lines 1 to line 2.
        Initialize particles with uniform weights.

        Overrides EDHFilter.initialize to add weight initialization.
        Initializes all vg-like state variables.

        Args:
            model_params: Dictionary with model parameters

        Returns:
            particles: Initial particles, shape (state_dim, n_particle)
            m0: Random initial mean, shape (state_dim, 1)
        """
        # Call parent initialization
        particles, m0 = super().initialize(model_params)

        # Initialize uniform weights
        self.weights = tf.ones(self.n_particle, dtype=tf.float32) / self.n_particle
        self.log_weights = tf.math.log(self.weights)

        # Initialize intermediate states
        self.particles_mean = tf.squeeze(m0)  # Convert (state_dim, 1) to (state_dim,)
        self.M = m0
        self.mu_0 = m0  # Mean trajectory
        self.P_pred = self.P  # Initial covariance

        if self.verbose:
            print("  Initialized uniform particle weights")
            print("  Initialized PFPF state variables")

        return particles, m0

    def _propagate_particles_deterministic(
        self,
        particles: tf.Tensor,
        model_params: Dict
    ) -> tf.Tensor:
        """
        Propagate particles through motion model WITHOUT noise.

        Used for computing proposal density.

        Args:
            particles: Current particles, shape (state_dim, n_particle)
            model_params: Dictionary with model parameters

        Returns:
            particles_pred: Predicted particles (deterministic), shape (state_dim, n_particle)
        """
        Phi = model_params['Phi']
        return tf.matmul(Phi, particles)

    def _update_weights(
        self,
        particles_flowed: tf.Tensor,
        particles_pred: tf.Tensor,
        particles_pred_deterministic: tf.Tensor,
        measurement: tf.Tensor,
        log_jacobian_det_sum: float,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Update particle weights using PFPF weight formula.

        Weight update formula:
            w_k ∝ [p(x_k|x_{k-1}) * p(z_k|x_k) / q(x_k|x_{k-1},z_k)] * w_{k-1}

        In log space:
            log w_k = log p(x_k|x_{k-1}) + log p(z_k|x_k) - log q(...) + log w_{k-1}

        Args:
            particles_flowed: Particles after flow (x_k)
            particles_pred: Propagated particles WITH noise (η_0)
            particles_pred_deterministic: Deterministic propagation (Φ·x_{k-1})
            measurement: Current measurement
            log_jacobian_det_sum: Sum of log Jacobian determinants
            model_params: Dictionary with model parameters

        Returns:
            weights: Updated normalized weights
            log_weights: Updated log weights
        """
        Q = model_params['Q']

        # Calculate log proposal density using prove_function.py
        log_proposal = self._log_proposal_density(
            particles_pred, particles_pred_deterministic, Q, log_jacobian_det_sum
        )

        # Calculate log process/prior density using prove_function.py
        log_prior = self._log_process_density(
            particles_flowed, particles_pred_deterministic, Q
        )

        # Calculate log likelihood using prove_function.py
        llh = self._log_likehood_density(particles_flowed, measurement, model_params)

        # Weight update
        log_weights_updated = log_prior + llh - log_proposal + self.log_weights

        # Normalize by subtracting max for numerical stability
        log_weights_updated = log_weights_updated - tf.reduce_max(log_weights_updated)

        # Convert to linear weights using particle_estimate
        # This normalizes the weights properly
        _, weights = self._particle_estimate(log_weights_updated, particles_flowed)

        return weights, log_weights_updated

    @tf.function
    def _compute_effective_sample_size(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Compute effective sample size N_eff = 1 / Σ(w_i^2).

        Args:
            weights: Normalized particle weights

        Returns:
            N_eff: Effective sample size
        """
        return 1.0 / tf.reduce_sum(weights ** 2)

    def _resample(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, float]:
        """
        Resample particles based on effective sample size.

        Args:
            particles: Current particles
            weights: Current weights

        Returns:
            particles: Resampled particles (or original if no resampling)
            weights: Updated weights (uniform if resampled)
            log_weights: Updated log weights
            N_eff: Effective sample size
        """
        N_eff = self._compute_effective_sample_size(weights)

        # Resample
        indices = self._multinomial_resample(weights)
        particles = tf.gather(particles, indices, axis=1)

        # Reset to uniform weights
        weights = tf.ones(self.n_particle, dtype=tf.float32) / self.n_particle
        log_weights = tf.math.log(weights)

        if self.verbose:
            print(f"    Resampled (N_eff={N_eff.numpy():.1f})")


        return particles, weights, log_weights, N_eff, indices


    def _particle_flow_edh(self, model_params, measurement):
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

        P_pred = self.P_pred

        log_weights = self.log_weights
        # Initialize auxiliary trajectory for global linearization
        eta_bar = tf.expand_dims(self.auxiliary_trajectory, 1) if len(self.auxiliary_trajectory.shape) == 1 else self.auxiliary_trajectory
        eta_bar_mu_0 = tf.expand_dims(self.mu_0, 1) if len(self.mu_0.shape) == 1 else self.mu_0

        # Algorithm Line 10: Set λ = 0
        # Algorithm Line 11: for j = 1, ..., N_λ do
        for j in range(self.n_lambda):
            # Algorithm Line 12: Set λ = λ + ε_j
            epsilon_j = self.lambda_steps[j]   # Step size
            lambda_j = self.lambda_values[j]   # Current lambda value
            
            # Algorithm Line 13: Compute A, b ONCE at global mean η̄
            A, b = self._compute_flow_parameters(eta_bar, eta_bar_mu_0, P_pred, measurement,
                                            lambda_j, model_params)
            
            #############################################################################################################################################################
            ############# Attention here we use self.particles_mean and updated self.particles_mean later that is different with the algorithm ##########################
            #############################################################################################################################################################
            # Algorithm Line 14: Migrate η̄
            slope_bar = tf.linalg.matvec(A, self.particles_mean) + b ############# It could be wrong even if it match the orginal matlab code
            eta_bar = eta_bar + epsilon_j * tf.expand_dims(slope_bar, 1)

            # Algorithm Lines 15-17: Migrate all particles using the same A, b
            slopes = tf.matmul(A, self.particles) + tf.expand_dims(b, 1)
            self.particles = self.particles + epsilon_j * slopes

            particles_mean, _ = self._particle_estimate(log_weights, self.particles)
            self.particles_mean = particles_mean


    def _particle_estimate(self, log_weights, particles):
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



    #################################### calcuate log density proposal ####################################

    def _log_proposal_density(self, xp_prop, xp_prop_deterministic, Q, log_jacobian_det_sum):
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


    def _log_process_density(self, xp, xp_prop_deterministic, Q):
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

    def _log_likehood_density(self, particles_flowed, measurement, model_params):
        """
        Compute likelihood p(z_k|x_k) for Gaussian measurement model.
        observation_model_general: it is called function that we can calculate z_pre for all partical
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
        z_pre = self.observation_model_general(particles_flowed, model_params, no_noise=True)
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

    def _multinomial_resample(self, weights):
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



    def step(
        self,
        measurement: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform one PFPF filter step.

        Follows the workflow from PFPF_function_clean_EDH.ipynb using self.* state.
        Extends EDH step with weight updates and resampling.

        Args:
            measurement: Current measurement, shape (n_sensor, 1) or (n_sensor,)
            model_params: Dictionary with model parameters

        Returns:
            particles_updated: Updated particles
            mean_estimate: Weighted state estimate
            P_updated: Posterior covariance
        """
        # Ensure measurement is (n_sensor, 1)
        if len(measurement.shape) == 1:
            measurement = tf.expand_dims(measurement, 1)

        # Store previous estimate;  use x_est_prev to represent self.M 
        x_est_prev = tf.expand_dims(self.M, 1) if len(self.M.shape) == 1 else self.M

        # Step 1: Propagate particles (with and without noise) 
        # In PFPF_EDH algorithm: corresponds to lines 5-8
        self.particles_pred = self._propagate_particles(self.particles, model_params)
        self.particles_pred_deterministic = self._propagate_particles_deterministic(
            self.particles, model_params
        )

        # Step 2: Estimate prior covariance
        if self.use_ekf:
            x_ekf_pred, P_pred = self._ekf_predict(self.x_ekf, self.P) #In PFPF_EDH algorithm: corresponds to lines 4

            eigenvalues = tf.linalg.eigvalsh(P_pred)
            min_eigenvalue = tf.reduce_min(eigenvalues)
            if min_eigenvalue <= 0:
                P_pred = self._cov_regularize(P_pred)
            self.P_pred = P_pred

        else:
            self.P_pred = self._estimate_covariance(self.particles_pred, model_params)

        # Step 3: Update mean trajectory mu_0 #In PFPF_EDH algorithm: corresponds to lines 9
        # This is the deterministic propagation of the previous estimate
        self.mu_0 = self.state_transition(x_est_prev, model_params, no_noise=True)

        self.particles_previous = self.particles
        self.particles = self.particles_pred

        # Step 4: Update auxiliary trajectory for linearization 
        # For EDH, this is the same as mu_0 (global linearization point)
        mean_estimate, _ = self._particle_estimate(self.log_weights, self.particles)
        self.auxiliary_trajectory = mean_estimate
        self.particles_mean = mean_estimate

        # Step 5: Particle flow
        # In PFPF_EDH algorithm: corresponds to lines 10 - 18
        self._particle_flow_edh(model_params, measurement)

        # Step 6: PFPF weight update
        # In PFPF_EDH algorithm: corresponds to lines 19 - 25
        log_jacobian_det_sum = 0.0  # Can be computed for local linearization
        weights, log_weights = self._update_weights(
            self.particles,
            self.particles_pred,
            self.particles_pred_deterministic,
            measurement,
            log_jacobian_det_sum,
            model_params
        )

        # Step 7: Weighted estimate using particle_estimate from prove_function.py
        # In PFPF_EDH algorithm: corresponds to lines 27
        mean_estimate, _ = self._particle_estimate(log_weights, self.particles)
        self.particles_mean = mean_estimate

        # Step 8: Covariance update
        # In PFPF_EDH algorithm: corresponds to lines 26
        if self.use_ekf:
            x_ekf_updated, P_updated = self._ekf_update(x_ekf_pred, self.P_pred, measurement)
            self.x_ekf = x_ekf_updated

            eigenvalues = tf.linalg.eigvalsh(P_updated)
            min_eigenvalue = tf.reduce_min(eigenvalues)
            if min_eigenvalue <= 0:
                P_updated = self._cov_regularize(P_updated)

        else:
            P_updated = self._estimate_covariance(self.particles, model_params)

        # Step 9: Resample if needed
        particles_resampled, weights_resampled, log_weights_resampled, N_eff, _ = self._resample(
            self.particles, weights
        )

        # Step 10: Update all internal state 
        self.particles = particles_resampled
        self.weights = weights_resampled
        self.log_weights = log_weights_resampled
        self.particles_mean = mean_estimate
        self.P = P_updated
        self.M = mean_estimate

        return particles_resampled, mean_estimate, P_updated, N_eff

    def run(
        self,
        measurements: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run PFPF filter on measurement sequence.

        Overrides EDHFilter.run with PFPF-specific processing.

        Args:
            measurements: Measurements, shape (n_sensor, T)
            model_params: Dictionary with model parameters

        Returns:
            estimates: State estimates, shape (state_dim, T)
            particles_all: All particles, shape (state_dim, n_particle, T)
            covariances_all: All covariances, shape (state_dim, state_dim, T)
        """
        T = tf.shape(measurements)[1].numpy()

        if self.verbose:
            print(f"\nRunning PFPF Filter:")
            print(f"  Particles: {self.n_particle}")
            print(f"  Lambda steps: {self.n_lambda}")
            print(f"  Lambda ratio: {self.lambda_ratio}")
            print(f"  Linearization: {'Local' if self.use_local else 'Global (EDH)'}")
            print(f"  EKF Covariance: {'Enabled' if self.use_ekf else 'Disabled'}")
            print(f"  Time steps: {T}")

        
        # Initialize
        self.initialize(model_params) #In PFPF_EDH algorithm: corresponds to lines 1-2.

        # Storage
        estimates_list = []
        particles_list = []
        covariances_list = []
        Neff_list = []
        # Main loop
        if self.verbose:
            print("\nProcessing time steps...")

        for t in range(T):  #In PFPF_EDH algorithm: corresponds to lines 3
            if self.verbose and (t + 1) % 10 == 0:
                print(f"  Step {t+1}/{T}")
            print('t', t)

            z_t = tf.expand_dims(measurements[:, t], 1)

            # Filter step
            particles, mean_estimate, P, Neff = self.step(z_t, model_params) #In PFPF_EDH algorithm: corresponds to lines 4-28

            # Store results
            estimates_list.append(tf.expand_dims(mean_estimate, 1))
            particles_list.append(tf.expand_dims(particles, 2))
            covariances_list.append(tf.expand_dims(P, 2))
            Neff_list.append(Neff)
        # Concatenate results
        estimates = tf.concat(estimates_list, axis=1)
        particles_all = tf.concat(particles_list, axis=2)
        covariances_all = tf.concat(covariances_list, axis=2)
        combined = tf.stack(Neff_list, axis=0)
        Neff_all = tf.reshape(combined, (-1, 1))

        if self.verbose:
            print("\nPFPF filter completed successfully!")
            print(f"  Estimates shape: {estimates.shape}")
            print(f"  Particles shape: {particles_all.shape}")
            print(f"  Covariances shape: {covariances_all.shape}")
            print(f"  Effective sample size: {Neff_all.shape}")

        return estimates, particles_all, covariances_all, Neff_all #In PFPF_EDH algorithm: corresponds to lines 29


