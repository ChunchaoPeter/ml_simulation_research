"""
Acoustic Multi-Target Tracking Model - Function-Based Implementation

Based on: Li & Coates (2017), "Particle Filtering with Invertible Particle Flow"
MATLAB Reference in the paper

================================================================================
STATE-SPACE MODEL DESCRIPTION
================================================================================

This is a NONLINEAR and NON-GAUSSIAN state-space model for multi-target acoustic tracking
implemented in TensorFlow.

STATE REPRESENTATION
--------------------
For C targets, the full state is:
    x = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ..., xC, yC, vxC, vyC]^T

Each target has 4 state variables:
    - (x, y): position in 2D space (meters)
    - (vx, vy): velocity in 2D space (meters/timestep)

MATLAB equivalent: Same format in Acoustic_example_initialization.m

MOTION MODEL (Linear - Constant Velocity)
-----------------------------------------
    x_k = Φ * x_{k-1} + w_k
    where w_k ~ N(0, Q) is process noise

For each target i, the motion model is:
    Φ = [[1, 0, 1, 0],     # x_k = x_{k-1} + vx_{k-1}
         [0, 1, 0, 1],     # y_k = y_{k-1} + vy_{k-1}
         [0, 0, 1, 0],     # vx_k = vx_{k-1}
         [0, 0, 0, 1]]     # vy_k = vy_{k-1}

MATLAB equivalent: Phi in Acoustic_example_initialization.m line 61

OBSERVATION MODEL (Nonlinear - Acoustic Sensors)
------------------------------------------------
    z_k = h(x_k) + v_k
    where v_k ~ N(0, R) is measurement noise

For sensor s located at position (s_x, s_y), the measurement is:
    z_s = Σ_{i=1}^{C} [Ψ / (r_i^α + d_0)] + v_s

where:
    - r_i = sqrt((x_i - s_x)^2 + (y_i - s_y)^2) is distance from target i to sensor s
    - Ψ = amplitude of sound at source (default: 10)
    - α = inverse power decay rate (default: 1)
    - d_0 = distance threshold (default: 0.1)

MATLAB equivalent: Acoustic_hfunc.m implements this measurement model

KEY DIFFERENCE FROM MATLAB: TWO PROCESS NOISE MATRICES
-------------------------------------------------------
The MATLAB code uses TWO different Q matrices (see Acoustic_example_initialization.m lines 66-74):

1. Q_real (for trajectory generation): Smaller, represents true process noise
   Q_real = 0.05 * [[1/3, 0, 0.5, 0],
                     [0, 1/3, 0, 0.5],
                     [0.5, 0, 1, 0],
                     [0, 0.5, 0, 1]]

2. Q (for filtering): Larger, accounts for model uncertainty
   Q = [[3, 0, 0.1, 0],
        [0, 3, 0, 0.1],
        [0.1, 0, 0.03, 0],
        [0, 0.1, 0, 0.03]]

NONLINEARITY
------------
The model is nonlinear because:
    - Measurements involve inverse distance: 1/(r^α + d_0)
    - Distance calculation involves square root: r = sqrt((x - s_x)^2 + (y - s_y)^2)
    - Multiple targets contribute additively to each sensor measurement

The nonlinearity makes the exact posterior non-Gaussian, even with Gaussian noise.

================================================================================
"""

import tensorflow as tf

def initialize_acoustic_model(
    n_targets=4,
    sensor_positions=None,
    process_noise_std=None,
    process_noise_std_real=None,
    measurement_noise_std=0.1,
    amplitude=10.0,
    inv_power=1.0,
    d0=0.1,
    sim_area_size=40.0
):
    """
    Initialize the Acoustic Model parameters.

    This function sets up all parameters needed for the acoustic multi-target tracking model,
    including state transition matrices, process noise covariances, and sensor positions.

    MATLAB Reference: Acoustic_example_initialization.m

    Args:
        n_targets: Number of targets to track (default: 4)
                  MATLAB: ps.setup.nTarget
        sensor_positions: Sensor positions as (n_sensors, 2) array
                         MATLAB: sensorsXY.mat
        process_noise_std: Process noise covariance for FILTER (4x4) or None for default
                          This should be LARGER to account for model uncertainty
                          MATLAB: Qii (line 70)
        process_noise_std_real: Process noise covariance for TRAJECTORY GENERATION (4x4)
                               This should be SMALLER - represents true process noise
                               MATLAB: Qii_real = 0.05 * Gamma (lines 66-68)
        measurement_noise_std: Standard deviation of measurement noise
                              Default 0.1 gives highly informative measurements
                              MATLAB: measvar_real = 0.01 (line 112), but we use sqrt(0.01) = 0.1
        amplitude: Amplitude of sound at source (Ψ)
                  Controls how loud the source is
                  MATLAB: Amp = 10 (line 42)
        inv_power: Inverse power decay rate (α)
                  Controls how fast amplitude decays with distance
                  MATLAB: invPow = 1 (line 43)
        d0: Distance threshold to prevent division by zero
           Prevents singularity when sensor is exactly at target position
           MATLAB: d0 = 0.1 (line 44)
        sim_area_size: Size of simulation area (default: 40m x 40m)
                      Targets move within [0, sim_area_size] x [0, sim_area_size]
                      MATLAB: simAreaSize = 40 (line 27)

    Returns:
        model_params: Dictionary containing all model parameters with keys:
                     - n_targets, state_dim_per_target, state_dim: Dimension info
                     - sensor_positions, n_sensors: Sensor configuration
                     - amplitude, inv_power, d0: Acoustic model parameters
                     - Phi: State transition matrix (state_dim x state_dim)
                     - Q: Process noise for filtering (state_dim x state_dim)
                     - Q_real: Process noise for trajectory generation (state_dim x state_dim)
                     - R: Measurement noise covariance (n_sensors x n_sensors)
                     - x0_initial_target_states: Default initial state for 4 targets
    """
    # ============================================================
    # STEP 1: Store basic state dimensions
    # ============================================================
    state_dim_per_target = 4  # [x, y, vx, vy] for each target
    state_dim = n_targets * state_dim_per_target  # Total state dimension (e.g., 16 for 4 targets)
    
    # ============================================================
    # STEP 2: Load or create sensor positions (5x5 grid by default)
    # MATLAB: sensorsXY.mat, lines 26-29
    # ============================================================
    if sensor_positions is None:
        # Create default 5x5 grid of sensors at [0, 10, 20, 30, 40] in both x and y
        # Total: 25 sensors uniformly distributed across the surveillance area
        # This matches the MATLAB file: 'PFPF/Acoustic_Example/sensorsXY.mat'
        # Sensor layout:
        #   (0,40)  (10,40)  (20,40)  (30,40)  (40,40)
        #   (0,30)  (10,30)  (20,30)  (30,30)  (40,30)
        #   (0,20)  (10,20)  (20,20)  (30,20)  (40,20)
        #   (0,10)  (10,10)  (20,10)  (30,10)  (40,10)
        #   (0,0)   (10,0)   (20,0)   (30,0)   (40,0)
        sensor_positions = tf.constant([
                        [ 0.,  0.],  # Row 1 (y=0): Bottom edge sensors
                        [10.,  0.],
                        [20.,  0.],
                        [30.,  0.],
                        [40.,  0.],
                        [ 0., 10.],  # Row 2 (y=10)
                        [10., 10.],
                        [20., 10.],
                        [30., 10.],
                        [40., 10.],
                        [ 0., 20.],  # Row 3 (y=20): Middle row
                        [10., 20.],
                        [20., 20.],  # Center sensor
                        [30., 20.],
                        [40., 20.],
                        [ 0., 30.],  # Row 4 (y=30)
                        [10., 30.],
                        [20., 30.],
                        [30., 30.],
                        [40., 30.],
                        [ 0., 40.],  # Row 5 (y=40): Top edge sensors
                        [10., 40.],
                        [20., 40.],
                        [30., 40.],
                        [40., 40.]
                    ], dtype=tf.float32)

    # sensor_positions shape: (n_sensors, 2) where each row is [x, y]
    n_sensors = sensor_positions.shape[0]  # Should be 25 for default 5x5 grid
    
    # ============================================================
    # STEP 3: Set acoustic model parameters (from paper Section V-A)
    # MATLAB: Acoustic_example_initialization.m, lines 42-44
    # ============================================================
    # These parameters define the acoustic measurement model: z_s = Σ [Ψ / (r^α + d_0)]
    amplitude = tf.constant(amplitude, dtype=tf.float32)  # Ψ = 10: Sound amplitude at source
    inv_power = tf.constant(inv_power, dtype=tf.float32)  # α = 1: Inverse power decay rate
    d0 = tf.constant(d0, dtype=tf.float32)  # d_0 = 0.1: Distance threshold (prevents division by zero)

    # ============================================================
    # STEP 4: Build state transition matrix Φ (block diagonal for all targets)
    # MATLAB: Phi (line 61), then blkdiag(Phi, A) for multiple targets (line 82)
    # ============================================================
    # For a single target with constant velocity model (dt=1):
    # [x_k  ]   [1 0 dt 0 ] [x_{k-1}  ]   [w_x ]
    # [y_k  ] = [0 1 0  dt] [y_{k-1}  ] + [w_y ]
    # [vx_k ]   [0 0 1  0 ] [vx_{k-1} ]   [w_vx]
    # [vy_k ]   [0 0 0  1 ] [vy_{k-1} ]   [w_vy]
    Phi_single = tf.constant([
        [1.0, 0.0, 1.0, 0.0],  # x_k = x_{k-1} + vx_{k-1}  (position updates with velocity)
        [0.0, 1.0, 0.0, 1.0],  # y_k = y_{k-1} + vy_{k-1}  (position updates with velocity)
        [0.0, 0.0, 1.0, 0.0],  # vx_k = vx_{k-1}           (velocity stays constant + noise)
        [0.0, 0.0, 0.0, 1.0]   # vy_k = vy_{k-1}           (velocity stays constant + noise)
    ], dtype=tf.float32)

    # Create block diagonal matrix for all targets
    # Result: Φ = diag(Φ_single, Φ_single, ..., Φ_single)
    # Shape: (state_dim, state_dim) = (16, 16) for 4 targets
    # Each target evolves independently according to the same constant velocity model
    Phi_blocks = [Phi_single for _ in range(n_targets)]
    Phi = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(block) for block in Phi_blocks]
    ).to_dense()
    
    # ============================================================
    # STEP 5: Build process noise covariance Q (block diagonal for all targets)
    # MATLAB: Two different Q matrices! (lines 63-74)
    #
    # IMPORTANT: The MATLAB code uses TWO different Q matrices:
    #   1. Q      - LARGER noise for filtering (accounts for model uncertainty)
    #   2. Q_real - SMALLER noise for trajectory generation (true process noise)
    # ============================================================

    # Q for FILTERING (larger, accounts for model uncertainty)
    # MATLAB: Qii (line 70)
    # This is what the filter THINKS the process noise is
    # It's intentionally larger to make the filter more robust to model mismatch
    if process_noise_std is None:
        Q_single = tf.constant([
            [3.0,  0.0,  0.1,  0.0 ],  # Position x variance: 3.0 (high uncertainty)
            [0.0,  3.0,  0.0,  0.1 ],  # Position y variance: 3.0 (high uncertainty)
            [0.1,  0.0,  0.03, 0.0 ],  # Velocity vx variance: 0.03 (lower uncertainty)
            [0.0,  0.1,  0.0,  0.03]   # Velocity vy variance: 0.03 (lower uncertainty)
        ], dtype=tf.float32)
    else:
        Q_single = tf.constant(process_noise_std, dtype=tf.float32)

    # Q_real for TRAJECTORY GENERATION (smaller, true process noise)
    # MATLAB: Qii_real = gammavar_real * Gamma (lines 66-68)
    # This is the ACTUAL noise used to generate trajectories
    # It's smaller because we want realistic target motion
    if process_noise_std_real is None:
        gammavar_real = 0.05  # Small noise variance
        # Gamma matrix encodes correlation between position and velocity noise
        # Based on continuous-time white noise acceleration model
        Gamma = tf.constant([
            [1/3,  0.0,  0.5,  0.0],  # Position-position and position-velocity correlation for x
            [0.0,  1/3,  0.0,  0.5],  # Position-position and position-velocity correlation for y
            [0.5,  0.0,  1.0,  0.0],  # Velocity-position and velocity-velocity correlation for x
            [0.0,  0.5,  0.0,  1.0]   # Velocity-position and velocity-velocity correlation for y
        ], dtype=tf.float32)
        Q_real_single = gammavar_real * Gamma  # Scale Gamma by small variance
    else:
        Q_real_single = tf.constant(process_noise_std_real, dtype=tf.float32)

    # Create block diagonal covariances for all targets
    # Result: Q = diag(Q_single, Q_single, ..., Q_single)
    # MATLAB: blkdiag(Q, Qii) loop (lines 81-86)
    # Shape: (state_dim, state_dim) = (16, 16) for 4 targets
    Q_blocks = [Q_single for _ in range(n_targets)]
    Q = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(block) for block in Q_blocks]
    ).to_dense()

    Q_real_blocks = [Q_real_single for _ in range(n_targets)]
    Q_real = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(block) for block in Q_real_blocks]
    ).to_dense()
    
    # ============================================================
    # STEP 6: Set measurement noise covariance R (diagonal, independent sensors)
    # MATLAB: measvar_real = 0.01, R_real = measvar_real*eye(nSensor) (line 130)
    # ============================================================
    # R = σ²_w * I, where σ_w = measurement_noise_std (default 0.1)
    # Shape: (n_sensors, n_sensors) = (25, 25) for default 5x5 grid
    # Diagonal matrix because sensors are assumed independent
    # Small noise (0.1^2 = 0.01) makes measurements highly informative
    R = tf.eye(n_sensors, dtype=tf.float32) * (measurement_noise_std ** 2)

    # ============================================================
    # STEP 7: Define default initial state for 4-target scenario
    # MATLAB: x0 = [12 6 0.001 0.001 32 32 -0.001 -0.005 20 13 -0.1 0.01 15 35 0.002 0.002]'
    # ============================================================
    # Initial state format: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
    # These are the default positions from the paper for 4 targets
    # Each target starts at different position with small initial velocity
    x0_initial_target_states = tf.expand_dims(tf.constant([
        12.0, 6.0, 0.001, 0.001,      # Target 1: starts at (12, 6) with velocity (0.001, 0.001)
        32.0, 32.0, -0.001, -0.005,   # Target 2: starts at (32, 32) with velocity (-0.001, -0.005)
        20.0, 13.0, -0.1, 0.01,       # Target 3: starts at (20, 13) with velocity (-0.1, 0.01)
        15.0, 35.0, 0.002, 0.002      # Target 4: starts at (15, 35) with velocity (0.002, 0.002)
    ], dtype=tf.float32), axis=1)  # Shape: (16, 1) column vector

    # ============================================================
    # Return all parameters as dictionary
    # ============================================================
    return {
        # Dimension information
        'n_targets': n_targets,
        'state_dim_per_target': state_dim_per_target,
        'state_dim': state_dim,

        # Sensor configuration
        'sensor_positions': sensor_positions,  # (n_sensors, 2)
        'n_sensors': n_sensors,

        # Acoustic model parameters
        'amplitude': amplitude,     # Ψ: sound amplitude at source
        'inv_power': inv_power,     # α: inverse power decay rate
        'd0': d0,                   # d_0: distance threshold
        'sim_area_size': sim_area_size,  # Surveillance area size

        # State-space model matrices
        'Phi': Phi,                 # State transition matrix (state_dim x state_dim)
        'Q': Q,                     # Process noise for FILTERING (state_dim x state_dim)
        'Q_real': Q_real,           # Process noise for TRAJECTORY GENERATION (state_dim x state_dim)
        'R': R,                     # Measurement noise covariance (n_sensors x n_sensors)
        'measurement_noise_std': measurement_noise_std,  # σ_w: measurement noise std

        # Initial state
        'x0_initial_target_states': x0_initial_target_states  # Default initial state (state_dim, 1)
    }

def state_transition(x_prev, model_params, use_real_noise=False):
    """
    State transition: x_k = Φ * x_{k-1} + w_k

    This is a LINEAR motion model (constant velocity).
    Each target evolves independently according to:
        x_k = x_{k-1} + vx_{k-1}  (position updates with velocity)
        y_k = y_{k-1} + vy_{k-1}
        vx_k = vx_{k-1}           (velocity stays constant + noise)
        vy_k = vy_{k-1}

    MATLAB Reference: AcousticPropagate.m
                      xp = mvnrnd((Phi*xp)', multiGamma)'

    Args:
        x_prev: Previous state, shape (state_dim, 1) - column vector
               Format: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        model_params: Dictionary of model parameters from initialize_acoustic_model()
        use_real_noise: If True, use Q_real (for trajectory generation)
                       If False, use Q (for filtering)
                       MATLAB: Q_real vs Q in propparams_real vs propparams

    Returns:
        Next state x_k, shape (state_dim, 1) - column vector
    """
    # ============================================================
    # Extract parameters from model_params dictionary
    # ============================================================
    Phi = model_params['Phi']          # State transition matrix (state_dim x state_dim)
    state_dim = model_params['state_dim']  # Total state dimension (e.g., 16 for 4 targets)

    # Choose which process noise covariance to use
    # Q_real: smaller, for generating realistic trajectories
    # Q:      larger, for filtering (accounts for model uncertainty)
    Q = model_params['Q_real'] if use_real_noise else model_params['Q']

    # ============================================================
    # Step 1: Deterministic state transition (constant velocity model)
    # x_next = Φ @ x_prev
    # MATLAB: Phi * xp
    # ============================================================
    x_next = tf.linalg.matmul(Phi, x_prev)

    # ============================================================
    # Step 2: Add Gaussian process noise w ~ N(0, Q)
    # To sample from N(0, Q), we use: w = chol(Q) * z, where z ~ N(0, I)
    # MATLAB: mvnrnd((Phi*xp)', multiGamma)' - samples from N(Phi*xp, multiGamma)
    # ============================================================

    # Sample from standard normal distribution N(0, I)
    z = tf.random.normal([state_dim, 1], dtype=tf.float32)  # Shape: (state_dim, 1)

    # Compute Cholesky decomposition: Q = L * L^T
    # where L is lower triangular
    Q_chol = tf.linalg.cholesky(Q)  # Shape: (state_dim, state_dim)

    # Transform to N(0, Q): w = L * z
    w = tf.linalg.matmul(Q_chol, z)  # Shape: (state_dim, 1)

    # ============================================================
    # Final state: x_k = Φ * x_{k-1} + w_k
    # ============================================================
    x_next = x_next + w

    return x_next

def observation_model(x, model_params):
    """
    NONLINEAR observation model: z_k = h(x_k) + v_k

    Each sensor measures the sum of acoustic amplitudes from all targets.
    The amplitude from target i to sensor s decays with distance:

    For each sensor s:
        z_s = Σ_{i=1}^{n_targets} [Ψ / (r_i^α + d_0)] + v_s

    where:
        r_i = sqrt((x_i - s_x)^2 + (y_i - s_y)^2)  (Euclidean distance)
        Ψ = amplitude at source (default 10)
        α = inverse power (default 1)
        d_0 = distance threshold (default 0.1)

    MATLAB Reference: Acoustic_hfunc.m

    Args:
        x: Current state, shape (state_dim, 1) - column vector
           Format: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
           MATLAB: xp with same format
        model_params: Dictionary of model parameters from initialize_acoustic_model()

    Returns:
        Observation z, shape (n_sensors, 1) - column vector
        MATLAB: y with shape (nSensor, nParticles)
    """
    # ============================================================
    # Extract parameters from model_params dictionary
    # ============================================================
    state_dim = model_params['state_dim']                    # Total state dimension
    sensor_positions = model_params['sensor_positions']      # (n_sensors, 2)
    n_sensors = model_params['n_sensors']                    # Number of sensors
    amplitude = model_params['amplitude']                    # Ψ: sound amplitude
    inv_power = model_params['inv_power']                    # α: decay rate
    d0 = model_params['d0']                                  # d_0: distance threshold
    measurement_noise_std = model_params['measurement_noise_std']  # σ_w

    # ============================================================
    # Step 1: Extract target positions from state vector
    # MATLAB: xx = xp(1:4:nTarget*4,:); xy = xp(2:4:nTarget*4,:); x = [xx;xy];
    # ============================================================
    # Flatten from (state_dim, 1) to (state_dim,)
    x_flat = tf.squeeze(x, axis=1)

    # Extract x-coordinates: indices [0, 4, 8, 12, ...] for all targets
    # tf.range(0, state_dim, 4) gives [0, 4, 8, 12, ...] for 4-state targets
    x_positions = tf.gather(x_flat, tf.range(0, state_dim, 4))  # Shape: (n_targets,)

    # Extract y-coordinates: indices [1, 5, 9, 13, ...] for all targets
    # tf.range(1, state_dim, 4) gives [1, 5, 9, 13, ...]
    y_positions = tf.gather(x_flat, tf.range(1, state_dim, 4))  # Shape: (n_targets,)

    # Stack to get (n_targets, 2) matrix of [x, y] coordinates
    target_positions = tf.stack([x_positions, y_positions], axis=1)

    # ============================================================
    # Step 2: Compute distances from all targets to all sensors
    # MATLAB: v = bsxfun(@minus,sensorsPos,permute(x,[1 3 2]));
    #         v = v.^2;
    #         v = v(1:nTarget,:,:)+v(nTarget+1:2*nTarget,:,:);
    #         v = sqrt(v);
    # ============================================================
    # We need to compute distance between every (sensor, target) pair
    # sensor_positions: (n_sensors, 2)
    # target_positions: (n_targets, 2)

    # Expand dimensions to enable broadcasting:
    # (n_sensors, 2) -> (n_sensors, 1, 2)
    sensors_expanded = tf.expand_dims(sensor_positions, axis=1)
    # (n_targets, 2) -> (1, n_targets, 2)
    targets_expanded = tf.expand_dims(target_positions, axis=0)

    # Compute differences via broadcasting: (n_sensors, n_targets, 2)
    # diff[s, i, :] = [x_i - s_x, y_i - s_y]
    diff = targets_expanded - sensors_expanded

    # Compute squared distances: sum over last dimension
    # distances_sq[s, i] = (x_i - s_x)^2 + (y_i - s_y)^2
    distances_sq = tf.reduce_sum(tf.square(diff), axis=2)  # Shape: (n_sensors, n_targets)

    # Take square root to get Euclidean distances
    # distances[s, i] = r_i,s = sqrt((x_i - s_x)^2 + (y_i - s_y)^2)
    distances = tf.sqrt(distances_sq)  # Shape: (n_sensors, n_targets)

    # ============================================================
    # Step 3: Compute amplitude contributions
    # MATLAB: v = likeparams.Amp./ (v.^likeparams.invPow+likeparams.d0);
    # ============================================================
    # For each (sensor, target) pair, compute: Ψ / (r^α + d_0)
    # amplitudes[s, i] = Ψ / (distances[s, i]^α + d_0)
    amplitudes = amplitude / (tf.pow(distances, inv_power) + d0)

    # ============================================================
    # Step 4: Sum contributions from all targets for each sensor
    # MATLAB: v = sum(v,1);
    # ============================================================
    # For each sensor s: z_s = Σ_{i=1}^{n_targets} amplitudes[s, i]
    z_true = tf.reduce_sum(amplitudes, axis=1)  # Shape: (n_sensors,)

    # ============================================================
    # Step 5: Add measurement noise v ~ N(0, σ²_w)
    # MATLAB: Noise is added during measurement generation in GenerateMeasurements.m
    # ============================================================
    noise = tf.random.normal(
        shape=[n_sensors],
        mean=0.0,
        stddev=measurement_noise_std,
        dtype=tf.float32
    )

    z = z_true + noise

    # Return as column vector (n_sensors, 1)
    return tf.reshape(z, [n_sensors, 1])


def compute_observation_jacobian(x, model_params):
    """
    Compute the Jacobian matrix H = ∂h/∂x of the acoustic observation model.

    MATLAB Reference:
        Acoustic_dh_dxfunc.m

    Observation model (inv_power = 1):
        h_s(x) = Σ_{i=1}^{n_targets} Ψ / (r_{si} + d0)

    where:
        r_{si} = sqrt((x_i - s_x)^2 + (y_i - s_y)^2)

    Jacobian structure:
        State order per target:
            [x_i, y_i, vx_i, vy_i]

        Measurement does NOT depend on velocity:
            ∂h/∂vx_i = 0
            ∂h/∂vy_i = 0

        Position derivatives:
            ∂h_s/∂x_i = -Ψ (x_i - s_x) / [r_{si} (r_{si} + d0)^2]
            ∂h_s/∂y_i = -Ψ (y_i - s_y) / [r_{si} (r_{si} + d0)^2]

    Args:
        x : Tensor, shape (state_dim, 1)
            State vector:
            [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]^T

        model_params : dict
            {
                'n_targets': int,
                'state_dim': int,
                'sensor_positions': Tensor (n_sensors, 2),
                'n_sensors': int,
                'amplitude': float (Ψ),
                'inv_power': int (assumed = 1),
                'd0': float
            }

    Returns:
        H : Tensor, shape (n_sensors, state_dim)
            Observation Jacobian matrix
    """
    # ============================================================
    # Unpack model parameters
    # ============================================================
    n_targets = model_params['n_targets']
    state_dim = model_params['state_dim']
    sensor_positions = model_params['sensor_positions']  # (n_sensors, 2)
    n_sensors = model_params['n_sensors']
    amplitude = model_params['amplitude']
    d0 = model_params['d0']
    
    # ============================================================
    # Step 1: Extract target positions from state vector
    # MATLAB:
    #   xx = xp(1:4:nTarget*4,:);
    #   xy = xp(2:4:nTarget*4,:);
    # ============================================================
    x_flat = tf.squeeze(x, axis=1)  # (state_dim,)

    x_positions = tf.gather(x_flat, tf.range(0, state_dim, 4))  # (n_targets,)
    y_positions = tf.gather(x_flat, tf.range(1, state_dim, 4))  # (n_targets,)

    # Stack positions as (x_i, y_i)
    target_positions = tf.stack([x_positions, y_positions], axis=1)  # (n_targets, 2)
    
        # ============================================================
    # Step 2: Compute sensor–target relative vectors and distances
    # MATLAB:
    #   mv = bsxfun(@minus, x, sensorsPos)
    #   v  = sqrt((x_i - s_x)^2 + (y_i - s_y)^2)
    # ============================================================
    sensors_expanded = tf.expand_dims(sensor_positions, axis=1)  # (n_sensors, 1, 2)
    targets_expanded = tf.expand_dims(target_positions, axis=0)  # (1, n_targets, 2)

    # diff[s, i, :] = [x_i - s_x, y_i - s_y]
    diff = targets_expanded - sensors_expanded  # (n_sensors, n_targets, 2)

    distances_sq = tf.reduce_sum(tf.square(diff), axis=2)  # (n_sensors, n_targets)
    distances = tf.sqrt(distances_sq)                      # r_{si}


    # ============================================================
    # Step 3: Compute scalar derivative factor
    # MATLAB:
    #   factor = -Amp ./ (r .* (r + d0).^2)
    # ============================================================
    factor = -amplitude / (distances * tf.square(distances + d0))
    factor = tf.expand_dims(factor, axis=2)  # (n_sensors, n_targets, 1)

    # ============================================================
    # Step 4: Position derivatives
    # MATLAB:
    #   v = (factor .* mv)'
    # ============================================================
    # position_derivatives[s, i, 0] = ∂h_s / ∂x_i
    # position_derivatives[s, i, 1] = ∂h_s / ∂y_i
    position_derivatives = factor * diff  # (n_sensors, n_targets, 2)

    # ============================================================
    # Step 5: Assemble full Jacobian matrix
    # MATLAB:
    #   dhdx(:,1:4:end) = ∂h/∂x
    #   dhdx(:,2:4:end) = ∂h/∂y
    # ============================================================
    H_rows = []
    for s in range(n_sensors):
        row_elements = []
        for i in range(n_targets):
            row_elements.extend([
                position_derivatives[s, i, 0],  # ∂h_s/∂x_i
                position_derivatives[s, i, 1],  # ∂h_s/∂y_i
                tf.constant(0.0, tf.float32),   # ∂h_s/∂vx_i
                tf.constant(0.0, tf.float32),   # ∂h_s/∂vy_i
            ])

        H_rows.append(tf.stack(row_elements))

    H = tf.stack(H_rows)  # (n_sensors, state_dim)

    return H


def simulate_trajectory(model_params, T,
                       keep_in_bounds=True, 
                       max_attempts=100):
    """
    Simulate a multi-target trajectory and observations.

    This version STRICTLY matches the original MATLAB behavior:
    - Initial state x0 is fixed (not sampled)
    - Uses real process noise Q_real
    - Rejects trajectories that leave the surveillance region

    MATLAB Reference:
        GenerateTracks.m
        GenerateMeasurements.m

    Time convention:
        States:        x_0, x_1, ..., x_T   (T+1 states)
        Observations:        z_1, ..., z_T   (T observations)

    Args:
        model_params : dict
            Must include:
                - x0 : initial state, shape (state_dim, 1)
                - state_dim
                - n_targets
                - n_sensors
                - sim_area_size
                - Q_real (used internally by state_transition)

        T : int
            Number of time steps / observations

        keep_in_bounds : bool
            If True, regenerate trajectory if any target leaves
            [5%, 95%] of the surveillance region

        max_attempts : int
            Maximum number of regeneration attempts

    Returns:
        states : Tensor, shape (state_dim, T+1)
        observations : Tensor, shape (n_sensors, T)
    """
    state_dim = model_params['state_dim']
    sim_area_size = model_params['sim_area_size']
    
    out_of_bounds = True
    while out_of_bounds:
        # ============================================================
        # Step 1: Initialize trajectory with FIXED initial state
        # MATLAB: x(:,1) = x0
        # ============================================================
        x = tf.identity(model_params['x0_initial_target_states'])  # shape (state_dim, 1)
        
        # Python lists (safe for rejection sampling)
        states = [tf.squeeze(x, axis=1)]
        observations = []

        # # Initialize storage
        # states_array = tf.TensorArray(dtype=tf.float32, size=T+1)
        # observations_array = tf.TensorArray(dtype=tf.float32, size=T)
        
        # # Store x_0
        # states_array = states_array.write(0, tf.squeeze(x, axis=1))
        
        out_of_bounds = False

        # ============================================================
        # Step 2: Propagate dynamics and generate measurements
        # MATLAB:
        #   for t = 2:T
        #       x(:,t) = AcousticPropagate(x(:,t-1), Q_real)
        #       z(:,t) = Acoustic_hfunc(x(:,t))
        #   end
        # ============================================================        
        for t in range(T):

            # ---- State transition (true dynamics) ----
            x = state_transition(x, model_params, use_real_noise=True)
            
            # Check if targets are within surveillance region
            if keep_in_bounds:
                x_flat = tf.squeeze(x, axis=1)
                
                # Extract positions [x1, x2, ...], [y1, y2, ...]
                x_positions = tf.gather(x_flat, tf.range(0, state_dim, 4))
                y_positions = tf.gather(x_flat, tf.range(1, state_dim, 4))
                
                lower_bound = 0.05 * sim_area_size
                upper_bound = 0.95 * sim_area_size
                
                x_out = tf.logical_or(
                    x_positions < lower_bound,
                    x_positions > upper_bound
                )
                y_out = tf.logical_or(
                    y_positions < lower_bound,
                    y_positions > upper_bound
                )
                
                # Reject entire trajectory if ANY target is out of bounds
                if tf.reduce_any(x_out) or tf.reduce_any(y_out):
                    out_of_bounds = True
                    break

           # ---- Generate observation ----
            z = observation_model(x, model_params)
            
            # ---- Store results ----
            # states_array = states_array.write(t + 1, tf.squeeze(x, axis=1))
            # observations_array = observations_array.write(t, tf.squeeze(z, axis=1))
            states.append(tf.squeeze(x, axis=1))
            observations.append(tf.squeeze(z, axis=1))
    # ------------------------------------------------------------
    # Valid trajectory found
    # ------------------------------------------------------------
    states = tf.stack(states, axis=1)
    observations = tf.stack(observations, axis=1)
    return states, observations
