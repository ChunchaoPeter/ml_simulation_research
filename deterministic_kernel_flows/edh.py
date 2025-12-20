"""
Exact Daum-Huang (EDH) Filter Implementation

This module implements the Exact Daum-Huang particle filter based on:
T. Ding and M. J. Coates, "Implementation of the Daum-Huang exact-flow
particle filter," in Proc. IEEE Statistical Signal Processing Workshop
(SSP), Ann Arbor, MI, Aug. 2012, pp. 257-260.

"""

import tensorflow as tf
from typing import Tuple, Dict, Optional
from acoustic_function import compute_observation_jacobian, observation_model


class EDHFilter:
    """
    Exact Daum-Huang (EDH) Particle Filter

    Implements Algorithm 1 from Ding & Coates (2012):
    - Particle propagation through motion model
    - Exact flow-based particle migration
    - EKF/UKF covariance tracking (optional)

    Attributes:
        n_particle (int): Number of particles
        n_lambda (int): Number of lambda steps for particle flow
        lambda_ratio (float): Ratio for exponential lambda spacing
        use_local (bool): If True, use local linearization (Algorithm 2)
        use_ekf (bool): If True, use EKF for covariance tracking
        ekf_filter (ExtendedKalmanFilter): pre-configured EKF instance
        verbose (bool): If True, print progress information
    """

    def __init__(
        self,
        n_particle: int = 100,
        n_lambda: int = 20,
        lambda_ratio: float = 1.2,
        use_local: bool = False,
        use_ekf: bool = False,
        ekf_filter: Optional['ExtendedKalmanFilter'] = None,
        verbose: bool = True
    ):
        """
        Initialize EDH filter.

        Args:
            n_particle: Number of particles (default: 100)
            n_lambda: Number of lambda steps (default: 20)
            lambda_ratio: Exponential spacing ratio (default: 1.2)
            use_local: Use local linearization (default: False)
            use_ekf: Use EKF for covariance tracking (default: False)
            ekf_filter: Pre-configured EKF filter instance
                       Required if use_ekf=True
            verbose: Print progress information (default: True)

        Raises:
            ValueError: If use_ekf=True but ekf_filter is None
        """
        self.n_particle = n_particle
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.use_local = use_local
        self.use_ekf = use_ekf
        self.verbose = verbose

        # Validate EKF configuration
        if use_ekf and ekf_filter is None:
            raise ValueError(
                "ekf_filter must be provided when use_ekf=True. "
                "Use create_ekf_for_acoustic_model() to create an EKF filter."
            )

        self.ekf_filter = ekf_filter  # Store user-provided filter

        # Pre-compute lambda steps
        self.lambda_steps, self.lambda_values = self._compute_lambda_steps()

        # Storage for filter state
        self.particles = None
        self.P = None  # Covariance matrix
        self.m0 = None  # Initial random mean
        self.ekf = None  # EKF instance (initialized when use_ekf=True)
        self.x_ekf = None  # EKF state estimate (used only when use_ekf=True)

    def _compute_lambda_steps(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute exponentially spaced lambda steps.

        Algorithm: Pre-processing

        Equations:
            ε_j = ε_1 * q^(j-1) for j=1,...,n
            ε_1 = (1-q)/(1-q^n)
            λ_j = Σ_{i=1}^j ε_i

        Returns:
            lambda_steps: Step sizes ε_j, shape (n_lambda,)
            lambda_values: Cumulative lambda values λ_j, shape (n_lambda,)
        """
        q = self.lambda_ratio
        n = self.n_lambda

        # Initial step size: ε_1 = (1-q)/(1-q^n)
        epsilon_1 = (1 - q) / (1 - q**n)

        # Step sizes: ε_j = ε_1 * q^(j-1) for j=1,...,n
        step_sizes = [epsilon_1 * (q**j) for j in range(n)]
        lambda_steps = tf.constant(step_sizes, dtype=tf.float32)

        # Cumulative lambda values: λ_j = Σ_{i=1}^j ε_i
        lambda_values = tf.cumsum(lambda_steps)

        return lambda_steps, lambda_values

    def initialize(self, model_params: Dict) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Initialize particles from Gaussian prior.

        Algorithm Lines 1-2:
            1  Draw {x_0^i}_{i=1}^N from the prior p(x_0)
            2  Set x̂_0 and m_0 as the mean; P_0 as the covariance matrix

        MATLAB Reference: AcousticGaussInit.m
            - Samples random mean m0 ~ N(x0, diag(sigma0^2))
            - Ensures m0 is within surveillance region bounds
            - Samples particles around m0: xp ~ N(m0, diag(sigma0^2))

        Args:
            model_params: Dictionary with model parameters

        Returns:
            particles: Initial particles, shape (state_dim, n_particle)
            m0: Random initial mean, shape (state_dim, 1)
        """
        state_dim = model_params['state_dim']
        x0 = model_params['x0_initial_target_states']
        P0 = model_params['P0']
        L = tf.linalg.cholesky(P0)
        sim_area_size = model_params['sim_area_size']

        # Sample random mean m0 within bounds
        out_of_bound = True
        while out_of_bound:
            noise = tf.random.normal((state_dim, 1), dtype=tf.float32)
            m0 = x0 + tf.matmul(L, noise)

            # Extract x and y positions for all targets
            x_positions = m0[0::4, 0]
            y_positions = m0[1::4, 0]

            # Check bounds: all positions in [0, sim_area_size]
            x_in_bounds = tf.reduce_all(x_positions >= 0.0) and tf.reduce_all(x_positions <= sim_area_size)
            y_in_bounds = tf.reduce_all(y_positions >= 0.0) and tf.reduce_all(y_positions <= sim_area_size)

            if x_in_bounds and y_in_bounds:
                out_of_bound = False

        # Sample particles around m0
        noise = tf.random.normal((state_dim, self.n_particle), dtype=tf.float32)
        particles = m0 + tf.matmul(L, noise)

        # Store initial state
        self.particles = particles
        self.P = P0
        self.m0 = m0

        # Initialize EKF if needed
        if self.use_ekf:
            self.ekf = self.ekf_filter
            self.x_ekf = m0
            if self.verbose:
                print("  Using provided EKF filter for covariance tracking")

        return particles, m0

    @tf.function
    def _propagate_particles_tf(
        self,
        particles: tf.Tensor,
        Phi: tf.Tensor,
        Q: tf.Tensor,
        state_dim: tf.Tensor
    ) -> tf.Tensor:
        """
        Propagate particles through motion model (TF graph-compiled version).

        Algorithm Line 4:
            4  Propagate particles x_{k-1}^i = f_k(x_{k-1}^i) + v_k

        Equation:
            x_k = Φ * x_{k-1} + w_k,  where w_k ~ N(0, Q)

        Args:
            particles: Current particles, shape (state_dim, n_particle)
            Phi: State transition matrix
            Q: Process noise covariance
            state_dim: State dimension (as tensor)

        Returns:
            particles_pred: Predicted particles, shape (state_dim, n_particle)
        """
        n_particle = tf.shape(particles)[1]

        # Linear propagation: x_k = Φ * x_{k-1}
        particles_pred = tf.matmul(Phi, particles)

        # Add process noise: w_k ~ N(0, Q)
        Q_chol = tf.linalg.cholesky(Q)
        noise = tf.matmul(Q_chol, tf.random.normal([state_dim, n_particle], dtype=tf.float32))

        return particles_pred + noise

    def _propagate_particles(self, particles: tf.Tensor, model_params: Dict) -> tf.Tensor:
        """Wrapper that extracts parameters and calls TF-compiled version."""
        state_dim = tf.constant(model_params['state_dim'], dtype=tf.int32)
        return self._propagate_particles_tf(
            particles,
            model_params['Phi'],
            model_params['Q'],
            state_dim
        )

    def _ekf_predict(self, x_prev: tf.Tensor, P_prev: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        EKF prediction step using ExtendedKalmanFilter from ekf.py.

        Note: NOT decorated with @tf.function because self.ekf.predict()
        is already decorated in the EKF class. Double-decoration causes issues.

        Algorithm Line 6:
            6  Apply UKF/EKF prediction: (m_{k-1|k-1}, P_{k-1|k-1}) → (m_{k|k-1}, P_{k|k-1})

        Equation:
            P_{k|k-1} = Φ * P_{k-1|k-1} * Φ^T + Q

        Args:
            x_prev: Previous state estimate, shape (state_dim,) or (state_dim, 1)
            P_prev: Previous posterior covariance P_{k-1|k-1}

        Returns:
            x_pred: Predicted state x_{k|k-1}
            P_pred: Predicted covariance P_{k|k-1}
        """
        # Ensure x_prev is (state_dim, 1)
        if len(x_prev.shape) == 1:
            x_prev = tf.expand_dims(x_prev, 1)

        # Use EKF predict method (already @tf.function decorated)
        x_pred, P_pred = self.ekf.predict(x_prev, P_prev, u=None)

        return x_pred, P_pred

    def _ekf_update(
        self,
        x_pred: tf.Tensor,
        P_pred: tf.Tensor,
        measurement: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        EKF update step using ExtendedKalmanFilter from ekf.py.

        Note: NOT decorated with @tf.function because self.ekf.update()
        is already decorated in the EKF class. Double-decoration causes issues.

        Algorithm Line 17:
            17  Apply UKF/EKF update: (m_{k|k-1}, P_{k|k-1}) → (m_{k|k}, P_{k|k})

        Equations:
            H = ∂h/∂x|_{x̄}
            K = P_{k|k-1} * H^T * (H * P_{k|k-1} * H^T + R)^{-1}
            P_{k|k} = (I - K*H) * P_{k|k-1}

        Args:
            x_pred: Predicted state x_{k|k-1}
            P_pred: Predicted covariance P_{k|k-1}
            measurement: Current measurement

        Returns:
            x_updated: Updated state x_{k|k}
            P_updated: Updated covariance P_{k|k}
        """
        # Ensure measurement is (n_sensor, 1)
        if len(measurement.shape) == 1:
            measurement = tf.expand_dims(measurement, 1)

        # Ensure x_pred is (state_dim, 1)
        if len(x_pred.shape) == 1:
            x_pred = tf.expand_dims(x_pred, 1)

        # Use EKF update method (already @tf.function decorated)
        x_updated, P_updated = self.ekf.update(measurement, x_pred, P_pred)

        return x_updated, P_updated

    @tf.function
    def _estimate_covariance_tf(self, particles: tf.Tensor, state_dim: tf.Tensor) -> tf.Tensor:
        """
        Estimate covariance from particles (TF graph-compiled version).

        Equation:
            P = (1/(N-1)) * Σ (x_i - x̄)(x_i - x̄)^T

        Args:
            particles: Particles, shape (state_dim, n_particle)
            state_dim: State dimension (as tensor)

        Returns:
            P: Covariance matrix, shape (state_dim, state_dim)
        """
        mean = tf.reduce_mean(particles, axis=1, keepdims=True)
        centered = particles - mean
        n_particles = tf.cast(tf.shape(particles)[1], tf.float32)
        P = tf.matmul(centered, tf.transpose(centered)) / (n_particles - 1.0)
        P = P + 1e-6 * tf.eye(state_dim, dtype=tf.float32)
        return P

    def _estimate_covariance(self, particles: tf.Tensor, model_params: Dict) -> tf.Tensor:
        """Wrapper that extracts parameters and calls TF-compiled version."""
        state_dim = tf.constant(model_params['state_dim'], dtype=tf.int32)
        return self._estimate_covariance_tf(particles, state_dim)

    @tf.function
    def _compute_flow_parameters_tf(
        self,
        x: tf.Tensor,
        x_bar: tf.Tensor,
        P: tf.Tensor,
        measurement: tf.Tensor,
        lam: tf.Tensor,
        R: tf.Tensor,
        H: tf.Tensor,
        h_x_bar: tf.Tensor,
        state_dim: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute flow parameters A and b (TF graph-compiled version).

        Algorithm Line 10:
            10  Calculate A and b from (8) and (9) using P_{k|k-1}, x̄ and H_x

        Equations:
            A(λ) = -1/2 * P * H^T * (λ*H*P*H^T + R)^{-1} * H
            b(λ) = (I + 2λA) * [(I + λA) * P*H^T*R^{-1}*(z-e) + A*x̄]

        Args:
            x: Particle position - used for linearization
            x_bar: Mean trajectory - used in b computation
            P: Covariance P_{k|k-1}
            measurement: Current measurement z
            lam: Current lambda value (as tensor)
            R: Measurement noise covariance
            H: Observation Jacobian (precomputed)
            h_x_bar: Observation at x_bar (precomputed, no noise)
            state_dim: State dimension (as tensor)

        Returns:
            A: Flow matrix, shape (state_dim, state_dim)
            b: Flow vector, shape (state_dim,)
        """
        # Compute linearization residual: e = h(x̄) - H*x̄
        h_x_bar_squeezed = tf.squeeze(h_x_bar, axis=1)
        e = h_x_bar_squeezed - tf.linalg.matvec(H, tf.squeeze(x_bar, axis=1))

        # Compute H*P*H^T
        HPHt = tf.matmul(tf.matmul(H, P), tf.transpose(H))

        # Innovation covariance: S = λ*H*P*H^T + R
        S = lam * HPHt + R
        S_inv = tf.linalg.inv(S)

        # Compute P*H^T
        PHt = tf.matmul(P, tf.transpose(H))

        # Flow matrix A = -0.5 * P*H^T * S^{-1} * H
        A = -0.5 * tf.matmul(tf.matmul(PHt, S_inv), H)

        # Innovation: z - e
        innovation = tf.squeeze(measurement, axis=1) - e

        # Compute R^{-1}*(z - e)
        R_inv = tf.linalg.inv(R)
        R_inv_innov = tf.linalg.matvec(R_inv, innovation)

        # Identity matrix
        I = tf.eye(state_dim, dtype=tf.float32)

        # (I + λA) and (I + 2λA)
        I_plus_lam_A = I + lam * A
        I_plus_2lam_A = I + 2 * lam * A

        # Flow vector b
        # First term: (I + λA) * P*H^T * R^{-1}*(z - e)
        term1 = tf.linalg.matvec(tf.matmul(I_plus_lam_A, PHt), R_inv_innov)
        # Second term: A * x̄
        term2 = tf.linalg.matvec(A, tf.squeeze(x_bar, axis=1))
        # Combined: b = (I + 2λA) * [term1 + term2]
        b = tf.linalg.matvec(I_plus_2lam_A, term1 + term2)

        return A, b

    def _compute_flow_parameters(
        self,
        x: tf.Tensor,
        x_bar: tf.Tensor,
        P: tf.Tensor,
        measurement: tf.Tensor,
        lam: float,
        model_params: Dict,
        use_local: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wrapper that computes H and h, then calls TF-compiled version.

        Args:
            x: Particle position - used for linearization
            x_bar: Mean trajectory - used in b computation
            P: Covariance P_{k|k-1}
            measurement: Current measurement z
            lam: Current lambda value
            model_params: Dictionary with observation model
            use_local: If True, linearize at x; if False, at x_bar

        Returns:
            A: Flow matrix, shape (state_dim, state_dim)
            b: Flow vector, shape (state_dim,)
        """
        # Calculate H_x by linearizing at x̄_k (or x_i for local)
        linearization_point = x if use_local else x_bar
        H = compute_observation_jacobian(linearization_point, model_params)

        # Compute h(x̄) and linearization residual: e = h(x̄) - H*x̄
        h_x_bar = observation_model(x_bar, model_params, no_noise=True)

        # Extract parameters
        R = model_params['R']
        state_dim = tf.constant(model_params['state_dim'], dtype=tf.int32)
        lam_tensor = tf.constant(lam, dtype=tf.float32)

        return self._compute_flow_parameters_tf(
            x, x_bar, P, measurement, lam_tensor, R, H, h_x_bar, state_dim
        )

    def _particle_flow(
        self,
        particles: tf.Tensor,
        measurement: tf.Tensor,
        P_pred: tf.Tensor,
        model_params: Dict
    ) -> tf.Tensor:
        """
        Migrate particles from prior to posterior using exact flow.

        Algorithm Lines 7-16:
            7  for j = 1, ..., N_λ do
            8    Set λ = j∆λ
            9    Calculate H_x by linearizing γ_k() at x̄_k
           10    Calculate A and b from (8) and (9)
           11    for i = 1, ..., N do
           12      Evaluate dx_k^i/dλ for each particle from (7)
           13      Migrate particles: x_k^i = x_k^i + ∆λ · dx_k^i/dλ
           14    endfor
           15    Re-evaluate x̄_k using the updated particles x_k^i
           16  endfor

        Flow Equation (7):
            dx/dλ = A(λ) * x + b(λ)

        Args:
            particles: Predicted particles
            measurement: Current measurement
            P_pred: Prior covariance P_{k|k-1}
            model_params: Dictionary with observation model

        Returns:
            particles_flowed: Updated particles
        """
        eta = tf.identity(particles)
        n_particle = tf.shape(particles)[1]

        for j in range(self.n_lambda):
            epsilon_j = self.lambda_steps[j]
            lambda_j = self.lambda_values[j]

            # Calculate/re-evaluate mean x̄_k
            eta_bar = tf.reduce_mean(eta, axis=1, keepdims=True)

            if self.use_local:
                # Local linearization at each particle
                # Use tf.map_fn for vectorized computation
                def compute_local_slope(i):
                    x_i = tf.expand_dims(eta[:, i], 1)
                    A_i, b_i = self._compute_flow_parameters(
                        x_i, eta_bar, P_pred, measurement,
                        lambda_j, model_params, use_local=True
                    )
                    slope_i = tf.linalg.matvec(A_i, tf.squeeze(x_i)) + b_i
                    return slope_i

                # Vectorize over particles
                slopes = tf.map_fn(
                    compute_local_slope,
                    tf.range(n_particle),
                    fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float32)
                )
                slopes = tf.transpose(slopes)  # (state_dim, n_particle)
            else:
                # Global linearization at mean
                A, b = self._compute_flow_parameters(
                    eta_bar, eta_bar, P_pred, measurement,
                    lambda_j, model_params, use_local=False
                )
                slopes = tf.matmul(A, eta) + tf.expand_dims(b, 1)

            # Migrate particles
            eta = eta + epsilon_j * slopes

        return eta

    def step(
        self,
        measurement: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform one EDH filter step.

        Algorithm Lines 4-18:
            4  Propagate particles
            5  Calculate mean
            6  Apply UKF/EKF prediction → P_{k|k-1}
            7-16  Particle flow (using P_{k|k-1})
           17  Apply UKF/EKF update → P_{k|k}
           18  Estimate x̂_k

        Args:
            measurement: Current measurement, shape (n_sensor, 1) or (n_sensor,)
            model_params: Dictionary with model parameters

        Returns:
            particles_updated: Updated particles
            mean_estimate: State estimate
            P_updated: Posterior covariance P_{k|k}
        """
        # Ensure measurement is (n_sensor, 1)
        if len(measurement.shape) == 1:
            measurement = tf.expand_dims(measurement, 1)

        # Line 4: Propagate particles
        particles_pred = self._propagate_particles(self.particles, model_params)

        # Line 6: Covariance prediction
        if self.use_ekf:
            x_ekf_pred, P_pred = self._ekf_predict(self.x_ekf, self.P)
        else:
            P_pred = self._estimate_covariance(particles_pred, model_params)

        # Lines 7-16: Particle flow
        particles_flowed = self._particle_flow(
            particles_pred, measurement, P_pred, model_params
        )

        # Line 18: Estimate state (from particles)
        mean_estimate = tf.reduce_mean(particles_flowed, axis=1)

        # Line 17: Covariance update
        if self.use_ekf:
            x_ekf_updated, P_updated = self._ekf_update(x_ekf_pred, P_pred, measurement)
            self.x_ekf = x_ekf_updated
        else:
            P_updated = self._estimate_covariance(particles_flowed, model_params)

        # Update internal state
        self.particles = particles_flowed
        self.P = P_updated

        return particles_flowed, mean_estimate, P_updated

    def run(
        self,
        measurements: tf.Tensor,
        model_params: Dict
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Run EDH filter on measurement sequence.

        Algorithm Main Loop:
            1-2  Initialize particles and covariance
            3    for k = 1 to T do
                   [Lines 4-19: filter step]
                 endfor

        Args:
            measurements: Measurements, shape (n_sensor, T)
            model_params: Dictionary with model parameters

        Returns:
            estimates: State estimates, shape (state_dim, T)
            particles_all: All particles, shape (state_dim, n_particle, T)
            covariances_all: All covariances P_{k|k}, shape (state_dim, state_dim, T)
        """
        T = tf.shape(measurements)[1].numpy()

        if self.verbose:
            print(f"\nRunning EDH Filter:")
            print(f"  Particles: {self.n_particle}")
            print(f"  Lambda steps: {self.n_lambda}")
            print(f"  Lambda ratio: {self.lambda_ratio}")
            print(f"  Linearization: {'Local (Algorithm 2)' if self.use_local else 'Global (Algorithm 1)'}")
            print(f"  EKF Covariance: {'Enabled' if self.use_ekf else 'Disabled (particle-based)'}")
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
            print("\nEDH filter completed successfully!")
            print(f"  Estimates shape: {estimates.shape}")
            print(f"  Particles shape: {particles_all.shape}")
            print(f"  Covariances shape: {covariances_all.shape}")

        return estimates, particles_all, covariances_all
