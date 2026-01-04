import tensorflow as tf
from typing import Tuple, Callable, Optional


class ParticleFlowFilter:
    """
    Particle Flow Filter (PFF) for Data Assimilation.
    
    This class implements the Particle Flow Filter algorithm for sequential
    data assimilation in dynamical systems. 
    """
    
    def __init__(
        self,
        dim: int,
        np_particles: int,
        nt: int,
        obs_interval: int,
        dim_interval: int,
        total_obs: int,
        nx: int,
        R: tf.Tensor,
        alpha: float = None,
        max_pseudo_step: int = 150,
        eps_init: float = 5e-2,
        stop_cri_percentage: float = 0.05,
        min_learning_rate: float = 1e-5,
        inflation_fac: float = 1.25,
        r_influ: int = 4,
        generate_Hx_si: Optional[Callable] = None,
        H_linear_adjoint: Optional[Callable] = None,
        dtype: tf.DType = tf.float32,
        learning_rate_factor: float = 1.5,
        kernel_type: str = 'matrix',
    ):
        """
        Initialize the Particle Flow Filter.
        
        Parameters:
        -----------
        dim : int
            State space dimension
        np_particles : int
            Number of ensemble particles
        nt : int
            Total number of time steps
        obs_interval : int
            Observation frequency (observe every obs_interval timesteps)
        dim_interval : int
            Spatial observation density (observe every dim_interval-th variable)
        nx : int
            Full model dimension
        R : tf.Tensor
            Observation error covariance matrix (ny_obs, ny_obs)
        alpha : float, optional
            Kernel bandwidth parameter (default: 1/np_particles)
        max_pseudo_step : int, optional
            Maximum pseudo-time iterations per cycle (default: 150)
        eps_init : float, optional
            Initial learning rate (default: 5e-2)
        stop_cri_percentage : float, optional
            Stopping criterion as percentage of initial gradient norm (default: 0.05)
        min_learning_rate : float, optional
            Minimum allowed learning rate (default: 1e-5)
        inflation_fac : float, optional
            Prior covariance inflation factor (default: 1.25)
        r_influ : int, optional
            Localization radius (default: 4)
        generate_Hx_si : callable, optional
            Observation operator function
        H_linear_adjoint : callable, optional
            Adjoint of linearized observation operator
        dtype : tf.DType, optional
            TensorFlow data type (default: tf.float32)
        learning_rate_factor : float, optional
            Learning rate adjustment factor (default: 1.5)
        kernel_type : str, optional
            Type of kernel to use: 'scalar' or 'matrix' (default: 'matrix')
            - 'scalar': Uses scalar Gaussian kernel (Eq. 16-19 in Hu & Van Leeuwen 2021)
            - 'matrix': Uses matrix-valued Gaussian kernel (Eq. 20-23 in Hu & Van Leeuwen 2021)
        """
        # Core dimensions
        self.dim = dim
        self.np_particles = np_particles
        self.nt = nt
        self.obs_interval = obs_interval
        self.total_obs = total_obs
        self.nx = nx

        # Observation configuration
        self.dim_interval = dim_interval
        # The dimension start for 4 for the L96_RK4 dataset
        self.dim_indices = tf.range(3, nx, dim_interval, dtype=tf.int32)
        self.ny_obs = len(self.dim_indices)
        self.R = R
        self.dtype = dtype

        # PFF algorithm parameters
        self.alpha = alpha if alpha is not None else 1.0 / np_particles
        self.max_pseudo_step = max_pseudo_step
        self.eps_init = eps_init
        self.stop_cri_percentage = stop_cri_percentage
        self.min_learning_rate = min_learning_rate

        # Kernel configuration
        if kernel_type not in ['scalar', 'matrix']:
            raise ValueError(f"kernel_type must be 'scalar' or 'matrix', got '{kernel_type}'")
        self.kernel_type = kernel_type
        
        # Prior configuration
        self.inflation_fac = inflation_fac
        self.r_influ = r_influ
        
        # Observation operators
        self.generate_Hx_si = generate_Hx_si
        self.H_linear_adjoint = H_linear_adjoint
        
        # Initialize gradient norm tracking
        self.norm_grad_KL = tf.Variable(
            tf.zeros((self.total_obs, max_pseudo_step), dtype=self.dtype),
            trainable=False
        )

        self.learning_rate_factor = learning_rate_factor

    def compute_prior_covariance(
        self, 
        particles: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute prior covariance from ensemble of particles.
        
        B = (inflation_fac / (N_p - 1)) * X X^T 
        
        Parameters:
        -----------
        particles : tf.Tensor, shape (dim, np_particles)
            Ensemble of particles
            
        Returns:
        --------
        B : tf.Tensor, shape (dim, dim)
            Prior covariance matrix
        mean : tf.Tensor, shape (dim, 1)
            Ensemble mean
        """
        # Ensemble mean
        mean = tf.reduce_mean(particles, axis=1)
        
        # Anomalies
        anomalies = particles - tf.expand_dims(mean, axis=1)
        
        # Sample covariance with inflation
        B = (self.inflation_fac / (self.np_particles - 1)) * \
            tf.matmul(anomalies, anomalies, transpose_b=True)
        
        return B, tf.expand_dims(mean, axis=1)
    
    def build_localization_mask(self) -> tf.Tensor:
        """
        Construct a periodic Gaussian localization (correlation) mask.

        This function builds a symmetric, banded localization matrix used for
        covariance localization in data assimilation algorithms (e.g. EnKF,
        particle filters, particle flow filters).

        The mask entries decay with the periodic distance between state indices
        according to a Gaussian function,

            rho_ij = exp( - (d(i, j) / r_influ)^2 ),

        where d(i, j) = min(|i - j|, dim - |i - j|) is the periodic distance.
        Correlations are truncated to zero for distances larger than 3 * r_influ,
        resulting in a compact-support, numerically stable localization.

        The diagonal entries are equal to 1, ensuring each state variable is
        fully correlated with itself.

        The returned mask is intended to be applied via element-wise (Hadamard)
        multiplication with a covariance matrix:

            P_localized = P * mask_tf

        Parameters
        ----------
        dim : int
            Dimension of the state vector.
        r_influ : int
            Localization influence radius controlling the decay of correlations.
        dtype : tf.DType, optional
            TensorFlow floating-point data type (default: tf.float32).

        Returns
        -------
        mask_tf : tf.Tensor
            Periodic Gaussian localization mask of shape (dim, dim).
        """
        mask = tf.eye(self.dim, dtype=self.dtype)
        
        for i in range(1, 3 * self.r_influ + 1):
            diag_val = tf.exp(tf.constant(-i**2 / self.r_influ**2, dtype=self.dtype))
            
            # Main diagonals
            upper_diag = tf.linalg.diag(tf.ones(self.dim - i, dtype=self.dtype), k=i)
            lower_diag = tf.linalg.diag(tf.ones(self.dim - i, dtype=self.dtype), k=-i)
            
            # Periodic wrap-around
            wrap_upper = tf.linalg.diag(tf.ones(i, dtype=self.dtype), k=-(self.dim - i))
            wrap_lower = tf.linalg.diag(tf.ones(i, dtype=self.dtype), k=(self.dim - i))
            
            mask += diag_val * (upper_diag + lower_diag + wrap_upper + wrap_lower)
        
        return mask
    
    def compute_grad_log_posterior(
        self,
        X_tmp: tf.Tensor,
        y_obs: tf.Tensor,
        obs_time: int,
        B_inv: tf.Tensor,
        X_mean: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute gradient of log-posterior for all particles.
        
        ∇ log p(x | y) = ∇ log p(y | x) + ∇ log p(x)
        
        Parameters:
        -----------
        X_tmp : tf.Tensor, shape (dim, np_particles)
            Current particle states
        y_obs : tf.Tensor, shape (ny_obs, total_obs)
            All observations
        obs_time : int
            Current observation time index
        B_inv : tf.Tensor, shape (dim, dim)
            Inverse prior covariance
        X_mean : tf.Tensor, shape (dim, 1)
            Prior mean
            
        Returns:
        --------
        grad_log_posterior : tf.Tensor, shape (dim, np_particles)
            Gradient for each particle
        """
        pseudo_Xs = tf.identity(X_tmp)
        obs_current = tf.expand_dims(y_obs[:, obs_time], axis=1)
        
        # Forward observation model
        y_i = self.generate_Hx_si(pseudo_Xs, self.dim_interval, self.nx)
        
        # Build dH/dx tensor
        dHdx = tf.zeros((self.ny_obs, self.dim, self.np_particles), self.dtype)
        for i in range(self.ny_obs):
            inner_ind = self.dim_indices[i]
            tmp_dHdx = self.H_linear_adjoint(tf.expand_dims(pseudo_Xs[i, :], 1))
            dHdx = tf.tensor_scatter_nd_update(
                dHdx,
                indices=[[i, inner_ind]],
                updates=tf.reshape(tmp_dHdx, (1, self.np_particles))
            )
        
        grad_log_posterior = []
        for i in range(self.np_particles):
            # Prior gradient
            grad_log_prior_i = -tf.matmul(B_inv, pseudo_Xs[:, i][:, None] - X_mean)
            
            # Likelihood gradient
            innovation = obs_current - y_i[:, i][:, None]
            dHi_T = tf.transpose(dHdx[:, :, i])
            R_inv_innov = tf.linalg.solve(self.R, innovation)
            grad_log_likelihood_i = tf.matmul(dHi_T, R_inv_innov)
            
            grad_log_posterior_i = grad_log_likelihood_i + grad_log_prior_i
            grad_log_posterior.append(grad_log_posterior_i)
        
        return tf.concat(grad_log_posterior, 1)
    
    def compute_matrix_kernel_and_gradient(
        self,
        pseudo_X: tf.Tensor,
        d: int,
        B: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute matrix kernel and its gradient for dimension d.

        Parameters:
        -----------
        pseudo_X : tf.Tensor, shape (dim, np_particles)
            Current particle states
        d : int
            Dimension index
        B : tf.Tensor, shape (dim, dim)
            Prior covariance matrix

        Returns:
        --------
        K : tf.Tensor, shape (np_particles, np_particles)
            Kernel matrix
        grad_K : tf.Tensor, shape (np_particles, np_particles)
            Gradient of kernel matrix
        """
        # Extract d-th dimension
        pseudo_X_d = pseudo_X[d:d+1, :]

        # Pairwise differences
        diff_matrix = pseudo_X_d - tf.transpose(pseudo_X_d)

        # RBF kernel
        K = tf.exp(-0.5 * diff_matrix**2 / (B[d, d] * self.alpha))

        # Kernel gradient
        grad_K = -K / (B[d, d] * self.alpha) * (-diff_matrix)

        return K, grad_K

    def compute_scalar_kernel_and_divergence(
        self,
        pseudo_X: tf.Tensor,
        B: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute scalar Gaussian kernel and its divergence.

        Based on Equations 16-19 from Hu & Van Leeuwen (2021):
        K(x, z) = K(x, z) * I_nx
        where K(x, z) = exp(-1/2 * (x - z)^T * A * (x - z))
        and A = (α * B)^(-1)

        Divergence: ∇_x · K(x, z) = -A^T(x - z)K(x, z)

        Parameters:
        -----------
        pseudo_X : tf.Tensor, shape (dim, np_particles)
            Current particle states
        B : tf.Tensor, shape (dim, dim)
            Prior covariance matrix

        Returns:
        --------
        K : tf.Tensor, shape (np_particles, np_particles)
            Scalar kernel values (same for all dimensions)
        div_K : tf.Tensor, shape (dim, np_particles, np_particles)
            Divergence of kernel for each dimension
        """
        # Compute A = (α * B)^(-1)
        A = tf.linalg.inv(self.alpha * B)

        # Compute pairwise kernel values
        # K[i, j] = exp(-0.5 * (x_i - x_j)^T * A * (x_i - x_j))
        K_matrix = tf.zeros((self.np_particles, self.np_particles), dtype=self.dtype)

        for i in range(self.np_particles):
            for j in range(self.np_particles):
                diff = tf.reshape(pseudo_X[:, i] - pseudo_X[:, j], (-1, 1))
                quad_form = tf.matmul(tf.matmul(tf.transpose(diff), A), diff)
                K_val = tf.exp(-0.5 * quad_form[0, 0])
                K_matrix = tf.tensor_scatter_nd_update(
                    K_matrix,
                    indices=[[i, j]],
                    updates=[K_val]
                )

        # Compute divergence: ∇_x · K(x, z) = -A^T(x - z)K(x, z)
        # For each dimension d and particle pair (i, j):
        # div_K[d, i, j] = -sum_k A^T[d, k] * (x_i[k] - x_j[k]) * K[i, j]
        div_K = tf.zeros((self.dim, self.np_particles, self.np_particles), dtype=self.dtype)

        A_T = tf.transpose(A)
        for i in range(self.np_particles):
            for j in range(self.np_particles):
                diff_vec = pseudo_X[:, i] - pseudo_X[:, j]  # (dim,)
                # A^T @ diff_vec gives (dim,) vector
                grad_component = -tf.linalg.matvec(A_T, diff_vec) * K_matrix[i, j]

                for d in range(self.dim):
                    div_K = tf.tensor_scatter_nd_update(
                        div_K,
                        indices=[[d, i, j]],
                        updates=[grad_component[d]]
                    )

        return K_matrix, div_K
    
    def adaptive_pseudo_step(
        self,
        pseudo_X: tf.Tensor,
        eps: tf.Tensor,
        grad_KL: tf.Tensor,
        qn: tf.Tensor,
        obs_time: int,
        s: int,
        ct: int,
        stop_cri: Optional[float],
    ) -> Tuple[tf.Tensor, tf.Tensor, int, int, bool, Optional[float]]:
        """
        Perform adaptive pseudo-time update with automatic learning rate adjustment.
        
        This function implements an adaptive gradient descent scheme that:
        - Reduces learning rate when gradient norm increases (backtracking)
        - Increases learning rate after consecutive successful steps
        - Stops when learning rate becomes too small

        Notes
        -----
        The function uses the following adaptive strategy:
        - If gradient norm increases by >2%, reduce learning rate by 1.*x and backtrack
        - If gradient norm is stable for 7+ steps, increase learning rate by 1.*x
        - Learning rate adjustments propagate to all future steps

        Parameters:
        -----------
        pseudo_X : tf.Tensor
            Current pseudo state to be updated
        eps : tf.Tensor, shape (max_pseudo_step,)
            Learning rate schedule for all pseudo steps
        grad_KL : tf.Tensor
            KL divergence gradient at current step
        qn : tf.Tensor
            Preconditioner matrix
        obs_time : int
            Current observation time index
        s : int
            Current pseudo-time step
        ct : int
            Consecutive success counter
        stop_cri : float or None
            Stopping criterion value
            
        Returns:
        --------
        pseudo_X : tf.Tensor
            Updated state
        eps : tf.Tensor
            Updated learning rate
        s : int
            Updated step index
        ct : int
            Updated counter
        stop : bool
            Whether to stop iteration
        stop_cri : float or None
            Updated stopping criterion
        """
        stop = False
        
        # Initialization
        if s == 0:
            stop_cri = self.stop_cri_percentage * self.norm_grad_KL[obs_time, 0]
            pseudo_X = pseudo_X + eps[s] * tf.matmul(qn, grad_KL)
            s += 1
            ct += 1
        
        # Learning rate too small
        elif eps[s] < self.min_learning_rate:
            print("    [Note] Learning rate too small, stopping iteration")
            stop = True
        
        # Gradient increased → reduce eps, backtrack
        elif s >= 1 and self.norm_grad_KL[obs_time, s] > 1.02 * self.norm_grad_KL[obs_time, s - 1]:
            new_eps = eps[s] / self.learning_rate_factor
            indices = tf.range(s - 1, self.max_pseudo_step)
            updates = tf.fill([self.max_pseudo_step - s + 1], new_eps)
            eps = tf.tensor_scatter_nd_update(
                eps,
                tf.reshape(indices, [-1, 1]),
                updates
            )
            s -= 1
            ct = 0
            print(f"    [Note] eps changed to {eps[s].numpy():.2e}, redo iteration")
        
        # Stable gradient → increase eps
        elif ct >= 7 and self.norm_grad_KL[obs_time, s] <= 1.02 * self.norm_grad_KL[obs_time, s - 1]:
            new_eps = eps[s] * self.learning_rate_factor
            indices = tf.range(s, self.max_pseudo_step)
            updates = tf.fill([self.max_pseudo_step - s], new_eps)
            eps = tf.tensor_scatter_nd_update(
                eps,
                tf.reshape(indices, [-1, 1]),
                updates
            )
            pseudo_X = pseudo_X + eps[s] * tf.matmul(qn, grad_KL)
            s += 1
            ct = 0
        
        # Normal update
        else:
            pseudo_X = pseudo_X + eps[s] * tf.matmul(qn, grad_KL)
            s += 1
            ct += 1
        
        return pseudo_X, eps, s, ct, stop, stop_cri
    
    def assimilate(
        self,
        X_list: list,
        y_obs: tf.Tensor,
        t: int,
        obs_time: int,
        verbose: bool = True
    ) -> Tuple[tf.Tensor, int]:
        """
        Perform PFF assimilation at a single observation time.
        
        Parameters:
        -----------
        X_list : list of tf.Tensor
            List of ensemble states at each timestep
        y_obs : tf.Tensor, shape (ny_obs, total_obs)
            All observations
        t : int
            Current model timestep
        obs_time : int
            Current observation index
        verbose : bool, optional
            Print iteration details (default: True)
            
        Returns:
        --------
        X_updated : tf.Tensor, shape (dim, np_particles)
            Updated ensemble after assimilation
        s_end : int
            Number of iterations performed
        """
        if verbose:
            print(f"\n  Time t={t+1}, Observation {obs_time+1}/{self.total_obs}")
        
        # Get current particles
        X_tmp = X_list[t+1]
        particles = X_tmp
        
        # Compute prior covariance
        B, X_mean = self.compute_prior_covariance(particles)
        C = B / tf.constant(self.np_particles, dtype=self.dtype)
        
        # Apply localization
        mask = self.build_localization_mask()
        C = C * mask
        
        # Inverse covariances
        C_inv = tf.linalg.inv(C)
        B = C * self.np_particles
        B_inv = C_inv / self.np_particles
        
        # Preconditioner
        qn = B / self.inflation_fac
        
        # Initialize iteration
        s = 0
        ct = 0
        
        # Set initial gradient norm
        self.norm_grad_KL.assign(
            tf.tensor_scatter_nd_update(
                self.norm_grad_KL,
                indices=tf.constant([[obs_time, 0]], dtype=tf.int32),
                updates=tf.constant([1e8], dtype=self.dtype)
            )
        )
        
        eps = self.eps_init * tf.ones(self.max_pseudo_step, dtype=self.dtype)
        pseudo_X = tf.identity(X_tmp)
        stop_cri = None
        
        # Main iteration loop
        while s < self.max_pseudo_step - 1:
            if s > 0 and self.norm_grad_KL[obs_time, s-1] <= stop_cri:
                if verbose:
                    print(f"    Convergence reached at iteration {s}")
                break
            
            # Compute gradient
            grad_log_post = self.compute_grad_log_posterior(
                pseudo_X, y_obs, obs_time, B_inv, X_mean
            )

            # Compute particle flow (Stein gradient)
            if self.kernel_type == 'matrix':
                # Matrix-valued kernel (Eq. 20-23)
                grad_KL_list = []
                for d in range(self.dim):
                    K_d, grad_K_d = self.compute_matrix_kernel_and_gradient(pseudo_X, d, B)
                    grad_KL_d = (
                        tf.reduce_sum(K_d * grad_log_post[d:d+1, :], axis=1, keepdims=True) +
                        tf.reduce_sum(grad_K_d, axis=1, keepdims=True)
                    ) / self.np_particles
                    grad_KL_list.append(tf.transpose(grad_KL_d))
                grad_KL = tf.concat(grad_KL_list, axis=0)

            elif self.kernel_type == 'scalar':
                # Scalar kernel (Eq. 16-19)
                K_scalar, div_K_scalar = self.compute_scalar_kernel_and_divergence(pseudo_X, B)

                # For each particle i, compute the flow
                # f_s(x_i) = (1/N_p) * sum_j [K(x_j, x_i) * grad_log_post(x_j) + div_K(x_j, x_i)]
                grad_KL_list = []
                for i in range(self.np_particles):
                    # Kernel term: sum over j of K[j, i] * grad_log_post[:, j]
                    kernel_term = tf.reduce_sum(
                        K_scalar[:, i:i+1] * grad_log_post,  # (np_particles, 1) * (dim, np_particles)
                        axis=1,
                        keepdims=True
                    )  # (dim, 1)

                    # Divergence term: sum over j of div_K[:, j, i]
                    div_term = tf.reduce_sum(div_K_scalar[:, :, i], axis=1, keepdims=True)  # (dim, 1)

                    grad_KL_i = (kernel_term + div_term) / self.np_particles
                    grad_KL_list.append(grad_KL_i)

                grad_KL = tf.concat(grad_KL_list, axis=1)  # (dim, np_particles)
            
            # Record gradient norm
            norm_value = tf.sqrt(tf.reduce_sum(grad_KL**2) / (self.dim * self.np_particles))
            self.norm_grad_KL.assign(
                tf.tensor_scatter_nd_update(
                    self.norm_grad_KL,
                    indices=tf.constant([[obs_time, s]], dtype=tf.int32),
                    updates=tf.reshape(norm_value, (1,))
                )
            )
            
            if verbose and s % 10 == 0:
                pct = self.norm_grad_KL[obs_time, s] / self.norm_grad_KL[obs_time, 0] * 100
                print(f"    Iteration s={s}, norm={pct:.2f}%, eps={eps[s]:.2e}")
            
            # Adaptive step
            pseudo_X, eps, s, ct, stop, stop_cri = self.adaptive_pseudo_step(
                pseudo_X, eps, grad_KL, qn, obs_time, s, ct, stop_cri
            )
            
            if stop:
                break
        
        if verbose:
            print(f"  PFF completed in {s} iterations")
        
        return pseudo_X, s
    
    def run(
        self,
        X_list: list,
        y_obs: tf.Tensor,
        model_step: Callable,
        t_start: int = 0,
        verbose: bool = True
    ) -> list:
        """
        Run full PFF data assimilation cycle.
        
        Parameters:
        -----------
        X_list : list of tf.Tensor
            Initial list of ensemble states
        y_obs : tf.Tensor, shape (ny_obs, total_obs)
            All observations
        model_step : callable
            Function to advance model one timestep: X_next = model_step(X)
        t_start : int, optional
            Starting timestep (default: 0)
        verbose : bool, optional
            Print progress information (default: True)
            
        Returns:
        --------
        X_list : list of tf.Tensor
            Updated ensemble states after assimilation
        """
        if verbose:
            print("\n" + "="*60)
            print("Starting Particle Flow Filter Data Assimilation")
            print("="*60)
        
        t = t_start
        
        while t < self.nt - 1:
            # Check if observation available
            io_obs = ((t + 1) % self.obs_interval == 0)
            
            # Step 1: Forecast
            X_next = model_step(X_list[t])
            X_list[t+1] = X_next
            
            # Step 2: Assimilation
            if io_obs:
                obs_time = (t + 1) // self.obs_interval - 1
                X_list[t+1], _ = self.assimilate(
                    X_list, y_obs, t, obs_time, verbose=verbose
                )
            
            t += 1
        
        if verbose:
            print("\n" + "="*60)
            print("PFF Data Assimilation Complete!")
            print("="*60)
        
        return X_list
    
