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

# Import utility functions from prove_function.py
from prove_function import (
    particle_estimate,
    log_proposal_density,
    log_process_density,
    log_likehood_density,
    multinomial_resample
)

tfd = tfp.distributions


class PFPF(EDHFilter):
    """
    Particle Filtering with Particle Flow (PFPF)

    Extends EDHFilter with importance weighting and resampling.
    Uses particle flow as a proposal distribution with proper weight updates.

    This class replaces the vg dictionary from PFPF_function_clean_EDH.ipynb
    with proper instance variables (self.*).

    Mapping from notebook's vg dictionary to class attributes:
        vg['particles']                -> self.particles
        vg['logW']                     -> self.log_weights
        vg['ml_weights']               -> self.weights
        vg['particles_m']              -> self.particles_mean
        vg['M']                        -> self.m0 (initial mean)
        vg['PU']                       -> self.P (posterior covariance)
        vg['PP']                       -> self.P_pred (prior covariance)
        vg['mu_0']                     -> self.mu_0 (mean trajectory)
        vg['xp_prop']                  -> self.particles_pred
        vg['xp_prop_deterministic']    -> self.particles_pred_deterministic
        vg['xp_auxiliary_individual']  -> self.auxiliary_trajectory

    Additional Attributes:
        weights: Particle weights, shape (n_particle,)
        log_weights: Log particle weights, shape (n_particle,)
        resample_threshold: Effective sample size threshold for resampling
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
        state_transition: Callable,
        n_particle: int = 100,
        n_lambda: int = 20,
        lambda_ratio: float = 1.2,
        use_local: bool = False,
        use_ekf: bool = False,
        ekf_filter: Optional['ExtendedKalmanFilter'] = None,
        resample_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize PFPF filter.

        Args:
            observation_jacobian: Callable that computes the observation Jacobian
            observation_model: Callable that maps state → observation
            n_particle: Number of particles (default: 100)
            n_lambda: Number of lambda steps (default: 20)
            lambda_ratio: Exponential spacing ratio (default: 1.2)
            use_local: Use local linearization (default: False)
            use_ekf: Use EKF for covariance tracking (default: False)
            ekf_filter: Pre-configured EKF filter instance
            resample_threshold: Resample when N_eff/N < threshold (default: 0.5)
            verbose: Print progress information (default: True)
        """
        super().__init__(
            observation_jacobian, observation_model,
            n_particle, n_lambda, lambda_ratio,
            use_local, use_ekf, ekf_filter, verbose
        )

        self.resample_threshold = resample_threshold

        self.state_transition = state_transition

        # Additional PFPF state
        self.weights = None
        self.log_weights = None

        # Intermediate computation states 
        self.particles_pred = None  # Propagated particles WITH noise
        self.particles_pred_deterministic = None  # Propagated WITHOUT noise
        self.particles_previous = None # Previous step for particles
        self.particles_mean = None  # Mean of particles
        self.mu_0 = None  # Mean trajectory η̄_μ0
        self.auxiliary_trajectory = None  # Auxiliary trajectory for linearization
        self.P_pred = None  # Prior covariance P_{k|k-1}
        self.M = None # the original mean 
        
    def initialize(self, model_params: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """
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
        log_proposal = log_proposal_density(
            particles_pred, particles_pred_deterministic, Q, log_jacobian_det_sum
        )

        # Calculate log process/prior density using prove_function.py
        log_prior = log_process_density(
            particles_flowed, particles_pred_deterministic, Q
        )

        # Calculate log likelihood using prove_function.py
        llh = log_likehood_density(particles_flowed, measurement, model_params)

        # Weight update
        log_weights_updated = log_prior + llh - log_proposal + self.log_weights

        # Normalize by subtracting max for numerical stability
        log_weights_updated = log_weights_updated - tf.reduce_max(log_weights_updated)

        # Convert to linear weights using particle_estimate
        # This normalizes the weights properly
        _, weights = particle_estimate(log_weights_updated, particles_flowed)

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
        threshold = self.resample_threshold * self.n_particle

        if N_eff < threshold:
            # Resample
            indices = multinomial_resample(weights)
            particles = tf.gather(particles, indices, axis=1)

            # Reset to uniform weights
            weights = tf.ones(self.n_particle, dtype=tf.float32) / self.n_particle
            log_weights = tf.math.log(weights)

            if self.verbose:
                print(f"    Resampled (N_eff={N_eff.numpy():.1f})")
        else:
            log_weights = tf.math.log(weights)

        return particles, weights, log_weights, N_eff.numpy()


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
            
            # Algorithm Line 14: Migrate η̄
            slope_bar = tf.linalg.matvec(A, self.particles_mean) + b # It could be wrong even if it match the orginal matlab code
            eta_bar = eta_bar + epsilon_j * tf.expand_dims(slope_bar, 1)

            # Algorithm Lines 15-17: Migrate all particles using the same A, b
            slopes = tf.matmul(A, self.particles) + tf.expand_dims(b, 1)
            self.particles = self.particles + epsilon_j * slopes

            particles_mean, _ = particle_estimate(log_weights, self.particles)
            self.particles_mean = particles_mean

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

        # Store previous estimate (like x_est_prev in notebook)
        x_est_prev = tf.expand_dims(self.M, 1) if len(self.M.shape) == 1 else self.M

        # Step 1: Propagate particles (with and without noise)
        # This corresponds to propagateAndEstimatePriorCovariance in notebook
        self.particles_pred = self._propagate_particles(self.particles, model_params)
        self.particles_pred_deterministic = self._propagate_particles_deterministic(
            self.particles, model_params
        )

        # Step 2: Estimate prior covariance (vg['PP'] in notebook)
        if self.use_ekf:
            x_ekf_pred, self.P_pred = self._ekf_predict(self.x_ekf, self.P)
        else:
            self.P_pred = self._estimate_covariance(self.particles_pred, model_params)

        # Step 3: Update mean trajectory mu_0 (This is line 5 - 6)
        # This is the deterministic propagation of the previous estimate
        self.mu_0 = self.state_transition(x_est_prev, model_params, no_noise=True)

        self.particles_previous = self.particles
        self.particles = self.particles_pred

        # Step 4: Update auxiliary trajectory for linearization (vg['xp_auxiliary_individual'])
        # For EDH, this is the same as mu_0 (global linearization point)
        mean_estimate, _ = particle_estimate(self.log_weights, self.particles)
        self.auxiliary_trajectory = mean_estimate
        self.particles_mean = mean_estimate

        # Step 5: Particle flow
        self._particle_flow_edh(model_params, measurement)

        # Step 6: PFPF weight update
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
        mean_estimate, _ = particle_estimate(log_weights, self.particles)
        self.particles_mean = mean_estimate

        # Step 8: Covariance update (vg['PU'] in notebook)
        if self.use_ekf:
            x_ekf_updated, P_updated = self._ekf_update(x_ekf_pred, self.P_pred, measurement)
            self.x_ekf = x_ekf_updated
        else:
            P_updated = self._estimate_covariance(self.particles, model_params)

        # Step 9: Resample if needed
        particles_resampled, weights_resampled, log_weights_resampled, N_eff = self._resample(
            self.particles, weights
        )

        # Step 10: Update all internal state (replaces vg dictionary updates)
        self.particles = particles_resampled
        self.weights = weights_resampled
        self.log_weights = log_weights_resampled
        self.particles_mean = mean_estimate
        self.P = P_updated
        self.M = mean_estimate

        return particles_resampled, mean_estimate, P_updated

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
            print(f"  Resample threshold: {self.resample_threshold}")
            print(f"  Time steps: {T}")

        # Initialize
        self.initialize(model_params)

        # Storage
        estimates_list = []
        particles_list = []
        covariances_list = []

        # Main loop
        if self.verbose:
            print("\nProcessing time steps...")

        for t in range(T):
            if self.verbose and (t + 1) % 10 == 0:
                print(f"  Step {t+1}/{T}")
            print('t', t)

            z_t = tf.expand_dims(measurements[:, t], 1)

            # Filter step
            particles, mean_estimate, P = self.step(z_t, model_params)

            # Store results
            estimates_list.append(tf.expand_dims(mean_estimate, 1))
            particles_list.append(tf.expand_dims(particles, 2))
            covariances_list.append(tf.expand_dims(P, 2))

        # Concatenate results
        estimates = tf.concat(estimates_list, axis=1)
        particles_all = tf.concat(particles_list, axis=2)
        covariances_all = tf.concat(covariances_list, axis=2)

        if self.verbose:
            print("\nPFPF filter completed successfully!")
            print(f"  Estimates shape: {estimates.shape}")
            print(f"  Particles shape: {particles_all.shape}")
            print(f"  Covariances shape: {covariances_all.shape}")

        return estimates, particles_all, covariances_all


