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
from pfpf_edh import PFPF_EDH


tfd = tfp.distributions


class PFPF_LEDH(PFPF_EDH):
    """
    Particle Filtering with Particle Flow using Localized EDH (PFPF_LEDH)

    Extends PFPF_EDH with local linearization at each particle.
    Implements Algorithm 1 from Li & Coates (2017).

    Key differences from PFPF_EDH:
        - Linearization performed at each particle's position (local)
        - Each particle has its own flow parameters A^i(λ) and b^i(λ)
        - Jacobian determinants computed for weight updates
        - Higher computational cost but better accuracy for nonlinear models

    Additional Attributes:
        log_jacobian_det_sum: Jacobian determinant products for each particle, shape (n_particle,)
    """

    def __init__(
        self,
        observation_jacobian: Callable,
        observation_model: Callable,
        state_transition: Callable,
        observation_model_general: Callable,
        n_particle: int = 100,
        n_lambda: int = 20,
        lambda_ratio: float = 1.2,
        use_ekf: bool = False,
        ekf_filter: Optional['ExtendedKalmanFilter'] = None,
        verbose: bool = True,
        redraw: bool = True
    ):
        """
        Initialize PFPF_LEDH filter.

        Args:
            observation_jacobian: Callable that computes the observation Jacobian
            observation_model: Callable that maps state → observation
            state_transition: Callable for state propagation
            observation_model_general: Callable that maps all state → all observation
            n_particle: Number of particles (default: 100)
            n_lambda: Number of lambda steps (default: 20)
            lambda_ratio: Exponential spacing ratio (default: 1.2)
            use_ekf: Use EKF for covariance tracking (default: False)
            ekf_filter: Pre-configured EKF filter instance
            verbose: Print progress information (default: True)
            redraw: Redraw particles from Multivariate Normal Distribution (default: True)
        """
        # Initialize parent PFPF_EDH
        super().__init__(
            observation_jacobian, observation_model, observation_model_general, state_transition,
            n_particle, n_lambda, lambda_ratio,
            use_local=True,  # LEDH uses local linearization
            use_ekf=use_ekf, ekf_filter=ekf_filter,
            verbose=verbose,
            redraw=redraw
        )

        # LEDH-specific state
        self.log_jacobian_det_sum = None  # Jacobian determinant products for each particle
        self.P_all = None  # P^{i}k_1 for each partical (Kalman variables: updated variance)
        self.P_pred_all = None # It has the same dimension as P_all (Kalman variables: updated variance)
        self.mu_0_all = None # It has the same dimension as P_all (Kalman variables: mean)
        self.M_prior_all = None # The mean of the Kalman variables

    def initialize(self, model_params: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initialize particles with uniform weights.

        Extends parent initialization to include log_jacobian_det_sum initialization and add P_all.

        Args:
            model_params: Dictionary with model parameters

        Returns:
            particles: Initial particles, shape (state_dim, n_particle)
            m0: Random initial mean, shape (state_dim, 1)
        """
        particles, m0 = super().initialize(model_params)

        # Initialize log_jacobian_det_sum for Jacobian determinants (Algorithm 1, Line 8)
        self.log_jacobian_det_sum = tf.zeros(self.n_particle, dtype=tf.float32)

        # Initialize the P^{i}k_1 and P_pred_all for each partical as we calculate Local
        self.P_all = tf.repeat(self.P[None, :, :], repeats=self.n_particle, axis=0)
        self.P_pred_all = tf.repeat(self.P_pred[None, :, :], repeats=self.n_particle, axis=0)

        if self.verbose:
            print("  Initialized PFPF_LEDH (local linearization)")

        return particles, m0

    def _particle_flow_ledh(
        self,
        model_params: Dict,
        measurement: tf.Tensor
    ) -> tf.Tensor:
        """
        Migrate particles using LEDH flow (local linearization at each particle).

        Implements Algorithm 1, Lines 11-21 from Li & Coates (2017).

        Algorithm Lines 11-21 (LEDH):
        11  Set λ = 0
        12  for j = 1, ..., N_λ do
        13    Set λ = λ + ε_j
        14    for i = 1, ..., Np do
        15      Set η̄₀ = η̄ᵢ and P = Pᵢ
        16      Calculate Aⁱⱼ(λ) and bⁱⱼ(λ) with linearization at η̄ᵢ
        17      Migrate η̄ᵢ: η̄ᵢ = η̄ᵢ + εⱼ(Aⁱⱼ(λ)η̄ᵢ + bⁱⱼ(λ))
        18      Migrate particles: ηᵢ₁ = ηᵢ₁ + εⱼ(Aⁱⱼ(λ)ηᵢ₁ + bⁱⱼ(λ))
        19      Calculate θⁱ = θⁱ / |det(I + εⱼAⁱⱼ(λ))|
        20    endfor
        21  endfor

        Args:
            model_params: Dictionary with observation model
            measurement: Current measurement, shape (n_sensor, 1)

        Returns:
            log_jacobian_det_sum: log theta +  Log Jacobian determinant products for each particle
        """

        # Each particle gets its own auxiliary trajectory η̄ᵢ
        # Initialize with predicted particles
        eta_bar = self.auxiliary_individual
        eta_bar_mu_0 = tf.expand_dims(self.mu_0, 1) if len(self.mu_0.shape) == 1 else self.mu_0
        log_jacobian_det_sum = tf.zeros(self.n_particle)
        log_weights = self.log_weights

        # Algorithm Line 11: Set λ = 0
        # Algorithm Line 12: for j = 1, ..., N_λ do
        for j in range(self.n_lambda):
            # Algorithm Line 13: Set λ = λ + ε_j
            epsilon_j = self.lambda_steps[j]   # Step size
            lambda_j = self.lambda_values[j]   # Current lambda value

            # Algorithm Line 14: for i = 1, ..., Np do
            # We'll process all particles in parallel for efficiency

            # Lists to store per-particle flow parameters
            slope_bar_list = []
            slopes_list = []
            log_jacobian_det_list = []

            for i in range(self.n_particle):
                # Algorithm Line 15: Set η̄₀ = η̄ᵢ and P = Pᵢ
                eta_bar_i_current = tf.expand_dims(eta_bar[:, i], 1)  # (state_dim, 1)
                P_pred_all_i_current = self.P_pred_all[i]
                particles_i = tf.expand_dims(self.particles[:,i], 1)

                # Algorithm Line 16: Calculate Aⁱⱼ(λ) and bⁱⱼ(λ)
                # with linearization at η̄ᵢ (local linearization)
                A_i, b_i = self._compute_flow_parameters(
                    eta_bar_i_current,  # Linearize at this particle
                    eta_bar_mu_0,
                    P_pred_all_i_current,
                    measurement,
                    lambda_j,
                    model_params
                )
                
                # Algorithm Line 17-18: Calculate part of them
                slope_bar_i = tf.matmul(A_i, eta_bar_i_current) + tf.expand_dims(b_i, 1)
                slopes_i = tf.matmul(A_i, particles_i) + tf.expand_dims(b_i, 1)

                # Algorithm Line 19: Calculate part of θⁱ = θⁱ|det(I + εⱼAⁱⱼ(λ))|
                dim = tf.shape(A_i)[0]  # or tf.constant(dim, dtype=tf.int32)
                I = tf.eye(dim, dtype=A_i.dtype)
                J = I + epsilon_j * A_i
                log_jacobian_det_i = tf.math.log(
                    tf.math.abs(
                        tf.linalg.det(J)
                    )
                )

                slope_bar_list.append(slope_bar_i)
                slopes_list.append(slopes_i)
                log_jacobian_det_list.append(log_jacobian_det_i)



            # Stack flow parameters for vectorized computation
            slope_bar_matrix = tf.concat(slope_bar_list, axis=1) # (state_dim, n_particle)
            slopes_matrix = tf.concat(slopes_list, axis=1)  # (state_dim, n_particle)
            log_jacobian_det_vector = tf.stack(log_jacobian_det_list)  # (n_particle, )



            # Algorithm Line 17: Migrate η̄ᵢ
            # Algorithm Line 18: Migrate particles
            # Migrate auxiliary trajectory
            self.auxiliary_individual = self.auxiliary_individual + epsilon_j * slope_bar_matrix
            # Migrate particle
            self.particles = self.particles + epsilon_j * slopes_matrix
            
            # Algorithm Line 19: Calculate part of θⁱ = θⁱ|det(I + εⱼAⁱⱼ(λ))|
            log_jacobian_det_sum = log_jacobian_det_sum + log_jacobian_det_vector
            log_jacobian_det_sum = log_jacobian_det_sum - tf.reduce_max(log_jacobian_det_sum)

            particles_mean, _ = self._particle_estimate(log_weights, self.particles)
            self.particles_mean = particles_mean

        return log_jacobian_det_sum

    def step(
        self,
        measurement: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform one PFPF_LEDH filter step.

        Implements Algorithm 1 from Li & Coates (2017).
        Uses local linearization for particle flow.

        Args:
            measurement: Current measurement, shape (n_sensor, 1) or (n_sensor,)
            model_params: Dictionary with model parameters

        Returns:
            particles_updated: Updated particles
            mean_estimate: Weighted state estimate
            P_updated: Posterior covariance
            N_eff: Effective sample size
        """
        # Ensure measurement is (n_sensor, 1)
        if len(measurement.shape) == 1:
            measurement = tf.expand_dims(measurement, 1)

        # Store previous estimate
        x_est_prev = tf.expand_dims(self.M, 1) if len(self.M.shape) == 1 else self.M

        # Step 1: Update mean trajectory mu_0, that is for global case in for calculating A_i and b_i
        self.mu_0 = self.state_transition(x_est_prev, model_params, no_noise=True)
        
        # Step 2: Estimate prior covariance (Algorithm 1, Line 4 - 5)
        P_pred_list = []
        M_prior_list = []
        for i in range(self.n_particle):
            particles_i = tf.expand_dims(self.particles[:,i], 1)
            
            if self.use_ekf:
                x_ekf_pred, P_pred = self._ekf_predict(
                    particles_i, 
                    self.P_all[i]
                )

                eigenvalues = tf.linalg.eigvalsh(P_pred)
                min_eigenvalue = tf.reduce_min(eigenvalues)
                if min_eigenvalue <= 0:
                    P_pred = self._cov_regularize(P_pred)

                P_pred_list.append(P_pred)
                M_prior_list.append(x_ekf_pred)
            else:
                raise RuntimeError("EKF must be enabled")
        self.P_pred_all = tf.stack(P_pred_list, axis=0)
        self.M_prior_all = tf.concat(M_prior_list, axis=1)

        # Step 3: Propagate particles (Algorithm 1, Lines 6-10)
        self.particles_pred = self._propagate_particles(self.particles, model_params)
        self.particles_pred_deterministic = self._propagate_particles_deterministic(
            self.particles, model_params
        )

        self.mu_0_all = self._propagate_particles_deterministic(
            self.particles, model_params
        )
        self.auxiliary_individual = self._propagate_particles_deterministic(
            self.particles, model_params
        )

        self.particles_previous = self.particles
        self.particles = self.particles_pred

        # Step 4: Update auxiliary trajectory for linearization
        mean_estimate, _ = self._particle_estimate(self.log_weights, self.particles)
        self.particles_mean = mean_estimate

        # self.auxiliary_trajectory = mean_estimate

        # Step 5: LEDH Particle flow (Algorithm 1, Lines 11-21)
        log_jacobian_det_sum = self._particle_flow_ledh(model_params, measurement)
        self.log_jacobian_det_sum = log_jacobian_det_sum


        # Step 6: PFPF weight update with Jacobian determinants (Algorithm 1, Line 22-27)
        weights, log_weights = self._update_weights(
            self.particles,
            self.particles_pred,
            self.particles_pred_deterministic,
            measurement,
            log_jacobian_det_sum,  # Use Log Jacobian determinants
            model_params
        )

        # Step 7: Weighted estimate (Algorithm 1, Line 30)
        mean_estimate, _ = self._particle_estimate(log_weights, self.particles)
        self.particles_mean = mean_estimate

        # Step 8: Covariance update (Algorithm 1, Line 26-29)
        P_update_list = []
        for i in range(self.n_particle):
            particles_i = tf.expand_dims(self.M_prior_all[:,i], 1)
            if self.use_ekf:
                x_ekf_updated, P_updated = self._ekf_update(
                    particles_i,
                    self.P_pred_all[i],
                    measurement
                )
                eigenvalues = tf.linalg.eigvalsh(P_updated)
                min_eigenvalue = tf.reduce_min(eigenvalues)
                if min_eigenvalue <= 0:
                    P_updated = self._cov_regularize(P_updated)

                P_update_list.append(P_updated)
            else:
                raise RuntimeError("EKF must be enabled")
        self.P_all = tf.stack(P_update_list, axis=0)


        # Step 9: Resample if needed
        particles_resampled, weights_resampled, log_weights_resampled, N_eff, indices = self._resample(
            self.particles, weights
        )
        P_all_resampled = tf.gather(self.P_all, indices, axis=0)


        # Step 10: Update all internal state
        self.P_all = P_all_resampled
        self.particles = particles_resampled
        self.weights = weights_resampled
        self.log_weights = log_weights_resampled
        self.particles_mean = mean_estimate
        self.M = mean_estimate

        return particles_resampled, mean_estimate, P_updated, N_eff
    

    def run(
        self,
        measurements: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run PFPF_LEDH filter on measurement sequence.

        Args:
            measurements: Measurements, shape (n_sensor, T)
            model_params: Dictionary with model parameters

        Returns:
            estimates: State estimates, shape (state_dim, T)
            particles_all: All particles, shape (state_dim, n_particle, T)
            covariances_all: All covariances, shape (state_dim, state_dim, T)
            Neff_all: Effective sample sizes, shape (T, 1)
        """
        T = tf.shape(measurements)[1].numpy()

        if self.verbose:
            print(f"\nRunning PFPF_LEDH Filter (Local Linearization):")
            print(f"  Particles: {self.n_particle}")
            print(f"  Lambda steps: {self.n_lambda}")
            print(f"  Lambda ratio: {self.lambda_ratio}")
            print(f"  Linearization: Local (LEDH)")
            print(f"  EKF Covariance: {'Enabled' if self.use_ekf else 'Disabled'}")
            print(f"  Time steps: {T}")

        # Initialize
        self.initialize(model_params) #In PFPF_LEDH algorithm: corresponds to lines 1-2.

        # Storage
        estimates_list = []
        particles_list = []
        covariances_list = []
        Neff_list = []

        # Main loop
        if self.verbose:
            print("\nProcessing time steps...")

        for t in range(T): #In PFPF_EDH algorithm: corresponds to lines 3
            if self.verbose and (t + 1) % 10 == 0:
                print(f"  Step {t+1}/{T}")

            z_t = tf.expand_dims(measurements[:, t], 1)

            # Filter step
            particles, mean_estimate, P, Neff = self.step(z_t, model_params)

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
            print("\nPFPF_LEDH filter completed successfully!")
            print(f"  Estimates shape: {estimates.shape}")
            print(f"  Particles shape: {particles_all.shape}")
            print(f"  Covariances shape: {covariances_all.shape}")
            print(f"  Effective sample size: {Neff_all.shape}")

        return estimates, particles_all, covariances_all, Neff_all #In PFPF_EDH algorithm: corresponds to lines 32
