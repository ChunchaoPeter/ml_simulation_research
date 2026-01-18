import tensorflow as tf
import numpy as np
from typing import Callable
from matplotlib.patches import Ellipse

def L96_RK4(X_in: tf.Tensor, dt: float, F: float) -> tf.Tensor:
    """
    Fourth-order Runge-Kutta (RK4) integration for the Lorenz 96 model.
    
    LORENZ 96 EQUATION
    ------------------
    The governing equation for each grid point i is:
    
        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    
    where:
        - x_i is the state variable at grid point i
        - Indices wrap around periodically (circular boundary conditions)
        - F is the forcing parameter controlling chaotic behavior
    
    This represents advection, dissipation, and external forcing in a 
    simplified atmospheric model.
    
    RK4 ALGORITHM
    -------------
    The RK4 method evaluates the tendency (derivative) at four points:
    
        k1 = f(X_in)                           [slope at beginning]
        k2 = f(X_in + 0.5 * dt * k1)          [slope at midpoint using k1]
        k3 = f(X_in + 0.5 * dt * k2)          [slope at midpoint using k2]
        k4 = f(X_in + dt * k3)                 [slope at endpoint using k3]
        
        X_out = X_in + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    The midpoint slopes (k2, k3) receive double weight because they provide
    more accurate estimates of the average slope over the interval.
    
    PARAMETERS
    ----------
    X_in : tf.Tensor
        Input state tensor of shape (dim, np); np is the number of particles
        
    dt : float
        Time step size for numerical integration.
    F : float
        Forcing parameter for the Lorenz 96 model.
        Typical value: 8.0
    
    RETURNS
    -------
    X_out : tf.Tensor
        Output state tensor of shape (dim, np)
        
        Represents the system state at time t + dt
            
        Relationship to input:
            X_out ≈ X_in + ∫[t to t+dt] (dx/dt) dt
            
            The RK4 method approximates this integral with high accuracy.
    """

    
    def compute_derivative(X: tf.Tensor) -> tf.Tensor:
        """
        Compute the Lorenz 96 tendency function (time derivative).
        Implementation Details
        ----------------------
        Uses tf.roll for efficient circular shifts:
            - shift=-1: moves elements UP (forward in index)
            - shift=1: moves elements DOWN (backward in index)
            - axis=0: operates on spatial dimension (rows)
            
        Example for 5-point system [x1, x2, x3, x4, x5]:
            X_p1[i] = x_{i+1} → [x2, x3, x4, x5, x1]
            X_n1[i] = x_{i-1} → [x5, x1, x2, x3, x4]
            X_n2[i] = x_{i-2} → [x4, x5, x1, x2, x3]
            
        The tendency at x3 (i=2, index 2) would be:
            dX[2] = (x4 - x1) * x2 - x3 + F
        """
        # Apply circular shifts to access neighboring grid points
        # These operations are vectorized across all ensemble members
        
        X_p1 = tf.roll(X, shift=-1, axis=0)  # X[i+1]: one grid point ahead
        X_n1 = tf.roll(X, shift=1, axis=0)   # X[i-1]: one grid point behind
        X_n2 = tf.roll(X, shift=2, axis=0)   # X[i-2]: two grid points behind
        X_00 = X                              # X[i]: current grid point

        dX = (X_p1 - X_n2) * X_n1 - X_00 + F
        
        return dX
    

    
    k1 = compute_derivative(X_in)
    k2 = compute_derivative(X_in + 0.5 * k1 * dt)
    k3 = compute_derivative(X_in + 0.5 * k2 * dt)
    k4 = compute_derivative(X_in + k3 * dt)

    X_out = X_in + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return X_out


def generate_observations(Xt, nt, warm_nt, obs_interval, dim_interval, R, nx=40):
    """
    Generate synthetic observations from true state with observation noise.
    
    Parameters:
    -----------
    Xt : tf.Tensor
        True state trajectory, shape [nx, warm_nt + nt]
    nt : int
        Number of time steps (after warmup)
    warm_nt : int
        Number of warmup time steps
    obs_interval : int
        Observation interval (observe every obs_interval time steps)
    dim_interval : int
        Dimension interval (observe every dim_interval-th variable)
    R : float
        Observation error
    nx : int, default=40
        State dimension
        
    Returns:
    --------
    y_obs : tf.Tensor
        Observations with noise, shape [ny_obs, total_obs]
    obs_indices : tf.Tensor
        Time indices of observations (0-based)
    dim_indices : tf.Tensor
        Variable indices observed (0-based)
    """
    # Calculate number of observations
    total_obs = nt // obs_interval
    
    # Determine which variables are observed (every dim_interval-th variable starting from index 3)
    dim_indices = tf.range(3, nx, dim_interval, dtype=tf.int32)
    ny_obs = len(dim_indices)
        
    # Generate observation noise: obs_rnd ~ N(0, R)
    z = tf.random.normal(shape=[ny_obs, total_obs], mean=0.0, stddev=1.0, dtype=tf.float32)
    L = tf.linalg.cholesky(R)
    obs_rnd = tf.matmul(L, z)  # Shape: [ny_obs, total_obs]
    
    # Determine which time steps are observed
    # Python (0-based): warm_nt-1+obs_interval : obs_interval : warm_nt+nt
    obs_indices = tf.range(warm_nt - 1 + obs_interval, warm_nt + nt, obs_interval, dtype=tf.int32)
    
    # Extract observed variables (spatial sampling)
    Xt_dim = tf.gather(Xt, dim_indices, axis=0)  # Shape: [ny_obs, warm_nt + nt]
    
    # Extract observed time steps (temporal sampling)
    Xt_final = tf.gather(Xt_dim, obs_indices, axis=1)  # Shape: [ny_obs, total_obs]
    
    # Apply observation operator and add noise
    y_obs = H_linear(Xt_final) + obs_rnd  # Shape: [ny_obs, total_obs]
    
    return y_obs, obs_indices, dim_indices


def regularized_inverse(A: tf.Tensor, cond_num: float) -> tf.Tensor:
    """
    Compute a numerically stable inverse of a matrix using SVD with
    condition-number-based truncation.

    This is a regularized (truncated) inverse, NOT a plain matrix inverse.

    Parameters
    ----------
    A : tf.Tensor
        Input matrix of shape (m, n)
    cond_num : float
        Condition number exponent.
        Singular values smaller than:
            10**cond_num * max(singular_value)
        are set to zero.

    Returns
    -------
    A_inv : tf.Tensor
        Regularized inverse of A
    """

    # SVD: A = U diag(S) V^T
    s, u, v = tf.linalg.svd(A, full_matrices=False)

    # Threshold for truncation
    cond_num = tf.cast(cond_num, s.dtype)

    ten = tf.constant(10.0, dtype=s.dtype)
    
    s_max = tf.reduce_max(s)
    threshold = tf.pow(ten, cond_num) * s_max
    # Invert singular values safely
    s_inv = tf.where(
        s < threshold,
        tf.zeros_like(s),
        tf.math.reciprocal(s)
    )

    # Reconstruct inverse: V diag(S^-1) U^T
    A_inv = tf.matmul(
        v,
        tf.matmul(tf.linalg.diag(s_inv), u, transpose_b=True)
    )

    return A_inv


def H_linear(X):
    """
    Linear observation operator (identity mapping).
    
    Parameters:
    -----------
    X : tf.Tensor or np.ndarray
        Ensemble in state space (size: [# state variables in inner domain, # ensemble members])
    
    Returns:
    --------
    Hx : tf.Tensor or np.ndarray
        Ensemble in observation space (same size as X)
    """
    return X


def H_linear_adjoint(X):
    """
    Adjoint of linear observation operator. This is dHdx
    
    Parameters:
    -----------
    X : tf.Tensor or np.ndarray
        Ensemble in state space (size: [# state variables in inner domain, # ensemble members])
    
    Returns:
    --------
    dHdx : np.ndarray
        Adjoint (size: [# state variables in inner domain, # ensemble members])
    """
    dim_inner, np_ens = X.shape
    return tf.ones((dim_inner, np_ens), dtype=X.dtype)


def generate_L96_trajectory(dim, warm_nt, nt, dt, F, L96_RK4):
    """Generate Lorenz 96 trajectory using TensorFlow."""
    
    total_steps = warm_nt + nt
    
    # Initialize as list to store states
    Xt_list = []
    
    # Set initial condition
    initial_state = tf.ones(dim, dtype=tf.float32) * F
    
    # Add perturbations at every dim//5 positions (adjusted for 0-indexing)
    perturb_indices = tf.range(dim // 5 - 1, dim, dim // 5, dtype=tf.int32)
    updates = tf.ones(len(perturb_indices), dtype=tf.float32)
    initial_state = tf.tensor_scatter_nd_add(
        initial_state,
        tf.expand_dims(perturb_indices, 1),
        updates
    )
    
    Xt_list.append(tf.reshape(initial_state, (-1, 1)))
    
    # Integrate
    print("Integrating L96 model...")
    for t in range(total_steps - 1):
        X_current = Xt_list[t]
        X_next = L96_RK4(X_current, dt, F)
        Xt_list.append(X_next)
        
        if (t + 1) % 200 == 0:
            print(f"  Step {t+1}/{total_steps}")
        
    # Stack all states into a single tensor
    Xt = tf.concat(Xt_list, axis=1)
    
    return Xt


def generate_initial_ensemble(Xt, warm_nt, dim, np_particles, nt, Q):
    """Generate initial ensemble for particle filter using TensorFlow."""
    
    # Get control mean: truth at end of warm-up + noise
    ctlmean = Xt[:, warm_nt] + tf.random.normal((dim,), dtype=tf.float32)
    
    # Initialize ensemble array
    X_list = []
    
    # Generate initial ensemble from N(ctlmean, Q)
    Q = tf.cast(Q, tf.float32)
    L = tf.linalg.cholesky(Q)
    standard_normal = tf.random.normal((dim, np_particles), dtype=tf.float32)
    initial_ensemble = tf.expand_dims(ctlmean, 1) + tf.matmul(L, standard_normal)
    
    X_list.append(tf.expand_dims(initial_ensemble, 2))
    
    # Add empty timesteps (will be filled later)
    for t in range(1, nt):
        X_list.append(tf.zeros((dim, np_particles, 1), dtype=tf.float32))
    
    X = tf.concat(X_list, axis=2)
    
    print(f"Initial ensemble generated with {np_particles} particles")
    
    return X

def run_ensemble_no_DA(X, nt, dt, F, L96_RK4):
    """Run ensemble without data assimilation (free run) using TensorFlow."""
    
    print("Running ensemble without DA (free run)...")
    
    # Initialize list to store ensemble states
    XnoDA_list = []
    
    # Copy initial ensemble
    XnoDA_list.append(X[:, :, 0])
    
    # Run ensemble forward without DA
    for t in range(nt - 1):
        X_current = XnoDA_list[t]
        X_next = L96_RK4(X_current, dt, F)
        XnoDA_list.append(X_next)
        
        if (t + 1) % 50 == 0:
            print(f"  Timestep {t+1}/{nt}")
    
    print("Free run complete!")
    
    # Stack all states into a single tensor
    XnoDA = tf.stack(XnoDA_list, axis=2)
    
    # Save initial state
    np.save('XnoDA.npy', XnoDA.numpy())
    
    return XnoDA


    """
    Compute prior covariance from ensemble of particles.
    
    B = (inflation_fac / (N_p - 1)) * X X^T
    
    where X is the anomaly matrix.
    
    Parameters:
    -----------
    particles : tf.Tensor (dim, n_particles)
        Ensemble of particles
    inflation_fac : float
        Covariance inflation factor
        
    Returns:
    --------
    B : tf.Tensor (dim, dim)
        Prior covariance matrix
    mean : tf.Tensor (dim,)
        Ensemble mean
    """
    n_particles = particles.shape[1]
    
    # Ensemble mean
    mean = tf.reduce_mean(particles, axis=1)
    
    # Anomalies
    anomalies = particles - tf.expand_dims(mean, axis=1)
    
    # Sample covariance with inflation
    B = (inflation_fac / (n_particles - 1)) * tf.matmul(anomalies, anomalies, transpose_b=True)/n_particles
    
    return B, tf.expand_dims(mean, axis=1)

def generate_Hx_si(pseudo_Xs, dim_interval, nx=40):
    """
    Generate synthetic observations from true state with observation noise.
    
    Parameters:
    -----------
    pseudo_Xs : tf.Tensor
        True state trajectory, shape [nx, np_particles]
 
    Returns:
    --------
    Hx_si : tf.Tensor
        Model Observations, shape [ny_obs, np_particles]
    """
    # Calculate number of observations
    
    # Determine which variables are observed (every dim_interval-th variable starting from index 3)
    dim_indices = tf.range(3, nx, dim_interval, dtype=tf.int32)

    # Extract observed variables (spatial sampling)
    Xt_dim = tf.gather(pseudo_Xs, dim_indices, axis=0)
    
    
    # Apply observation operator 
    Hx_si = H_linear(Xt_dim) 
    
    return Hx_si

def scalar_kernel(x: tf.Tensor, z: tf.Tensor, A: tf.Tensor) -> tf.Tensor:
    """
    Compute the scalar kernel K(x, z) as defined in Equation (17).
    
    K(x, z) = exp(-1/2 * (x - z)^T * A * (x - z))
    
    Parameters:
    -----------
    x : tf.Tensor
        First input vector of shape (n_x, 1)
    z : tf.Tensor
        Second input vector of shape (n_x, 1)
    A : tf.Tensor
        Matrix that defines the distance between particles in space, shape (n_x, n_x)
    
    Returns:
    --------
    tf.Tensor
        The scalar kernel value K(x, z)
    """
    diff = x - z
    exponent = -0.5 * tf.matmul(tf.matmul(tf.transpose(diff), A), diff)[0, 0]
    return tf.exp(exponent)

def scalar_kernel_gradient(x: tf.Tensor, z: tf.Tensor, A: tf.Tensor) -> tf.Tensor:
    """
    Compute the gradient of the scalar kernel with respect to x as in Equation (19).
    
    ∇_x · K(x, z) = -A^T(x - z)K(x, z)
    
    Parameters:
    -----------
    x : tf.Tensor
        First input vector of shape (n_x, 1)
    z : tf.Tensor
        Second input vector of shape (n_x, 1)
    A : tf.Tensor
        Matrix that defines the distance, shape (n_x, n_x)
    
    Returns:
    --------
    tf.Tensor
        Gradient vector of shape (n_x, 1)
    """
    K_val = scalar_kernel(x, z, A)
    diff = x - z
    gradient = -tf.matmul(tf.transpose(A), diff) * K_val
    return gradient

def matrix_valued_kernel(x: tf.Tensor, z: tf.Tensor, 
                         alpha: float, sigma: list) -> tf.Tensor:
    """
    Compute the matrix-valued kernel K(x, z) as defined in Equations (20) and (21).
    
    K(x, z) = diag([K_(1)(x, z), K_(2)(x, z), ..., K_(n_x)(x, z)])
    
    where K_(a)(x, z) = exp(-1/2 * (x_(a) - z_(a))^2 / (alpha * sigma_(a)^2))
    
    Parameters:
    -----------
    x : tf.Tensor
        First input vector of shape (n_x, 1)
    z : tf.Tensor
        Second input vector of shape (n_x, 1)
    alpha : float
        Scaling parameter
    sigma : list
        Standard deviation list of length n_x
    
    Returns:
    --------
    tf.Tensor
        The matrix-valued kernel K(x, z) of shape (n_x, n_x) as a diagonal matrix
    """
    sigma_tensor = tf.constant(sigma, dtype=tf.float32)
    sigma_tensor = tf.reshape(sigma_tensor, (-1, 1))
    
    diff_squared = tf.square(x - z)
    exponent = -0.5 * diff_squared / (alpha * tf.square(sigma_tensor))
    diagonal_elements = tf.squeeze(tf.exp(exponent))
    K = tf.linalg.diag(diagonal_elements)
    return K


def matrix_valued_kernel_gradient(x_s: tf.Tensor, x: tf.Tensor,
                                  alpha: float, sigma: list) -> tf.Tensor:
    """
    Compute the gradient of the matrix-valued kernel with respect to x_s 
    as in Equation (23).
    
    ∂/∂x_s K_(a)(x_s, x) = -(x_s,(a) - x_(a))/(alpha * sigma_(a)^2) * K_(a)(x_s,(a), x_(a))
    
    Parameters:
    -----------
    x_s : tf.Tensor
        Source point vector of shape (n_x, 1)
    x : tf.Tensor
        Target point vector of shape (n_x, 1)
    alpha : float
        Scaling parameter
    sigma : list
        Standard deviation list of length n_x
    
    Returns:
    --------
    tf.Tensor
        Gradient vector of shape (n_x, 1)
    """
    sigma_tensor = tf.constant(sigma, dtype=tf.float32)
    sigma_tensor = tf.reshape(sigma_tensor, (-1, 1))
    
    diff = x_s - x
    diff_squared = tf.square(diff)
    exponent = -0.5 * diff_squared / (alpha * tf.square(sigma_tensor))
    K_a = tf.exp(exponent)
    gradient = -diff / (alpha * tf.square(sigma_tensor)) * K_a
    return gradient

def plot_gradient_and_arrow_pff(ax, x_point, z_point, arrows, title, 
                                is_matrix_valued=True, arrow_scale=1.0, point_size=80):
    """
    Plot visualization with ellipses and directional arrows.
    
    Args:
        ax: matplotlib axis object
        x_point: starting point coordinates (2x1 array)
        z_point: ending point coordinates (2x1 array)
        arrows: arrow direction magnitudes (2x1 array)
        title: plot title
        is_matrix_valued: if True, plot matrix-valued; if False, plot scalar
        arrow_scale: scaling factor for arrow lengths
        point_size: size of scatter plot points
    """
    
    # Common styling parameters
    ellipse_size = 1.0
    ellipse_alpha_base = 0.3
    ellipse_alpha_dark = 0.5
    arrow_head_width = 0.15
    arrow_head_length = 0.1
    arrow_linewidth = 2
    
    # Helper function to draw arrows
    def draw_arrow(start_point, dx, dy):
        ax.arrow(start_point[0, 0], start_point[1, 0], 
                dx, dy,
                head_width=arrow_head_width, 
                head_length=arrow_head_length, 
                fc='black', ec='black', 
                linewidth=arrow_linewidth, 
                zorder=6)
    
    # Helper function to create ellipse
    def create_ellipse(center, alpha):
        return Ellipse((center[0, 0], center[1, 0]), 
                      width=ellipse_size, 
                      height=ellipse_size, 
                      facecolor='gray', 
                      alpha=alpha, 
                      zorder=1 if alpha == ellipse_alpha_base else 2)
    
    # Plot reference line and data points
    ax.axhline(y=x_point[1, 0], color='k', linestyle=':', linewidth=1.5)
    ax.scatter(x_point[0, 0], x_point[1, 0], c='black', s=point_size, zorder=5)
    ax.scatter(z_point[0, 0], z_point[1, 0], c='black', s=point_size, zorder=5)
    
    if is_matrix_valued:
        # Matrix-valued: orthogonal arrows from both points
        ax.add_patch(create_ellipse(x_point, ellipse_alpha_base))
        draw_arrow(x_point, -arrows[0, 0] * arrow_scale, 0)  # Left
        draw_arrow(x_point, 0, -arrows[1, 0] * arrow_scale)  # Down
        
        ax.add_patch(create_ellipse(z_point, ellipse_alpha_dark))
        draw_arrow(z_point, arrows[0, 0] * arrow_scale, 0)   # Right
        draw_arrow(z_point, 0, arrows[1, 0] * arrow_scale)   # Up
        
    else:
        # Scalar: diagonal arrows from both points (opposite directions)
        ax.add_patch(create_ellipse(x_point, ellipse_alpha_base))
        draw_arrow(x_point, 
                  -arrows[0, 0] * arrow_scale, 
                  -arrows[1, 0] * arrow_scale)
        
        ax.add_patch(create_ellipse(z_point, ellipse_alpha_dark))
        draw_arrow(z_point, 
                  arrows[0, 0] * arrow_scale, 
                  arrows[1, 0] * arrow_scale)
    
    # Format plot
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.set_title(title, fontsize=18, pad=10)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_marginal_distribution(ax, idx_unobs, idx_obs, prior_particles, posterior_particles, 
                                truth_val, title, kernel_name):
    """
    Plot 2D marginal distribution for x_19 and x_20.
    """
    # Extract relevant dimensions
    prior_x19 = prior_particles[idx_unobs, :].numpy()
    prior_x20 = prior_particles[idx_obs, :].numpy()
    
    post_x19 = posterior_particles[idx_unobs, :].numpy()
    post_x20 = posterior_particles[idx_obs, :].numpy()
    
    # Plot prior particles
    ax.scatter(prior_x19, prior_x20, s=100, c='black', alpha=0.6, 
               marker='o', label='Prior', zorder=3)
    
    # Plot posterior particles
    ax.scatter(post_x19, post_x20, s=100, c='red', alpha=0.7,
               marker='o', label='Posterior', zorder=4)
    
    # Plot truth
    ax.scatter(truth_val[idx_unobs], truth_val[idx_obs], s=300, 
               c='green', marker='*', edgecolors='black', linewidths=1.5,
               label='Truth', zorder=5)
    
    # Compute spread statistics
    post_std_x19 = np.std(post_x19)
    post_std_x20 = np.std(post_x20)
    
    ax.set_xlabel('$x_{19}$ (unobserved)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$x_{20}$ (observed)', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\n({kernel_name})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text with statistics
    textstr = f'Posterior spread:\n$\sigma_{{19}}$ = {post_std_x19:.3f}\n$\sigma_{{20}}$ = {post_std_x20:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)