import tensorflow as tf
import numpy as np

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
    return np.ones((dim_inner, np_ens))


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