import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

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
    
    MATLAB Reference: Acoustic_example_initialization.m
    
    Args:
        n_targets: Number of targets to track (default: 4)
                  MATLAB: ps.setup.nTarget
        sensor_positions: Sensor positions as (n_sensors, 2) array or None to load default
                         MATLAB: sensorsXY.mat
        process_noise_std: Process noise covariance for FILTER (4x4) or None for default
                          MATLAB: Qii (line 70)
        process_noise_std_real: Process noise covariance for TRAJECTORY GENERATION (4x4)
                               MATLAB: Qii_real = 0.05 * Gamma (lines 66-68)
        measurement_noise_std: Standard deviation of measurement noise
                              MATLAB: measvar_real = 0.01 (line 112)
        amplitude: Amplitude of sound at source (Ψ)
                  MATLAB: Amp = 10 (line 42)
        inv_power: Inverse power decay rate (α)
                  MATLAB: invPow = 1 (line 43)
        d0: Distance threshold
           MATLAB: d0 = 0.1 (line 44)
        sim_area_size: Size of simulation area (default: 40m x 40m)
                      MATLAB: simAreaSize = 40 (line 27)
        seed: Random seed for reproducibility
    
    Returns:
        model_params: Dictionary containing all model parameters
    """
    # Store basic parameters
    state_dim_per_target = 4  # [x, y, vx, vy] for each target
    state_dim = n_targets * state_dim_per_target  # Total state dimension
    
    # ============================================================
    # Load or create sensor positions (5x5 grid by default)
    # MATLAB: sensorsXY.mat, lines 26-29
    # ============================================================
    if sensor_positions is None:
        # We use the same data set as from to load default 25x2 grid from MATLAB file:'PFPF/Acoustic_Example/sensorsXY.mat'
        sensor_positions = tf.constant([
                        [ 0.,  0.],
                        [10.,  0.],
                        [20.,  0.],
                        [30.,  0.],
                        [40.,  0.],
                        [ 0., 10.],
                        [10., 10.],
                        [20., 10.],
                        [30., 10.],
                        [40., 10.],
                        [ 0., 20.],
                        [10., 20.],
                        [20., 20.],
                        [30., 20.],
                        [40., 20.],
                        [ 0., 30.],
                        [10., 30.],
                        [20., 30.],
                        [30., 30.],
                        [40., 30.],
                        [ 0., 40.],
                        [10., 40.],
                        [20., 40.],
                        [30., 40.],
                        [40., 40.]
                    ], dtype=tf.float32)

    # Convert to TensorFlow constant (n_sensors, 2)
    n_sensors = sensor_positions.shape[0]
    
    # ============================================================
    # Acoustic model parameters (from paper Section V-A)
    # MATLAB: Acoustic_example_initialization.m, lines 42-44
    # ============================================================
    amplitude = tf.constant(amplitude, dtype=tf.float32)  # Ψ = 10
    inv_power = tf.constant(inv_power, dtype=tf.float32)  # α = 1
    d0 = tf.constant(d0, dtype=tf.float32)  # d_0 = 0.1
    
    # ============================================================
    # Build state transition matrix Φ (block diagonal for all targets)
    # MATLAB: Phi (line 61), then blkdiag(Phi, A) for multiple targets (line 82)
    # ============================================================
    Phi_single = tf.constant([
        [1.0, 0.0, 1.0, 0.0],  # x_k = x_{k-1} + vx_{k-1}
        [0.0, 1.0, 0.0, 1.0],  # y_k = y_{k-1} + vy_{k-1}
        [0.0, 0.0, 1.0, 0.0],  # vx_k = vx_{k-1}
        [0.0, 0.0, 0.0, 1.0]   # vy_k = vy_{k-1}
    ], dtype=tf.float32)
    
    # Create block diagonal matrix for all targets
    Phi_blocks = [Phi_single for _ in range(n_targets)]
    Phi = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(block) for block in Phi_blocks]
    ).to_dense()
    
    # ============================================================
    # Build process noise covariance Q (block diagonal for all targets)
    # MATLAB: Two different Q matrices! (lines 63-74)
    # ============================================================
    
    # Q for FILTERING (larger, accounts for model uncertainty)
    if process_noise_std is None:
        Q_single = tf.constant([
            [3.0,  0.0,  0.1,  0.0 ],
            [0.0,  3.0,  0.0,  0.1 ],
            [0.1,  0.0,  0.03, 0.0 ],
            [0.0,  0.1,  0.0,  0.03]
        ], dtype=tf.float32)
    else:
        Q_single = tf.constant(process_noise_std, dtype=tf.float32)
    
    # Q_real for TRAJECTORY GENERATION (smaller, true process noise)
    if process_noise_std_real is None:
        gammavar_real = 0.05
        Gamma = tf.constant([
            [1/3,  0.0,  0.5,  0.0],
            [0.0,  1/3,  0.0,  0.5],
            [0.5,  0.0,  1.0,  0.0],
            [0.0,  0.5,  0.0,  1.0]
        ], dtype=tf.float32)
        Q_real_single = gammavar_real * Gamma
    else:
        Q_real_single = tf.constant(process_noise_std_real, dtype=tf.float32)
    
    # Create block diagonal covariances for all targets
    Q_blocks = [Q_single for _ in range(n_targets)]
    Q = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(block) for block in Q_blocks]
    ).to_dense()
    
    Q_real_blocks = [Q_real_single for _ in range(n_targets)]
    Q_real = tf.linalg.LinearOperatorBlockDiag(
        [tf.linalg.LinearOperatorFullMatrix(block) for block in Q_real_blocks]
    ).to_dense()
    
    # ============================================================
    # Measurement noise covariance R (diagonal, independent sensors)
    # MATLAB: measvar_real = 0.01, R_real = measvar_real*eye(nSensor) (line 130)
    # ============================================================
    R = tf.eye(n_sensors, dtype=tf.float32) * (measurement_noise_std ** 2)
    
    x0_initial_target_states = tf.expand_dims(tf.constant([
        12.0, 6.0, 0.001, 0.001,      # Target 1: starts at (12, 6)
        32.0, 32.0, -0.001, -0.005,   # Target 2: starts at (32, 32)
        20.0, 13.0, -0.1, 0.01,       # Target 3: starts at (20, 13)
        15.0, 35.0, 0.002, 0.002      # Target 4: starts at (15, 35)
    ], dtype=tf.float32), axis=1)
    # Return all parameters as dictionary
    return {
        'n_targets': n_targets,
        'state_dim_per_target': state_dim_per_target,
        'state_dim': state_dim,
        'sensor_positions': sensor_positions,
        'n_sensors': n_sensors,
        'amplitude': amplitude,
        'inv_power': inv_power,
        'd0': d0,
        'sim_area_size': sim_area_size,
        'Phi': Phi,
        'Q': Q,
        'Q_real': Q_real,
        'measurement_noise_std': measurement_noise_std,
        'R': R,
        'x0_initial_target_states': x0_initial_target_states
    }

def state_transition(x_prev, model_params, use_real_noise=False):
    """
    State transition: x_k = Φ * x_{k-1} + w_k
    
    MATLAB Reference: AcousticPropagate.m
    
    Args:
        x_prev: Previous state, shape (state_dim, 1) - column vector
        model_params: Dictionary of model parameters
        use_real_noise: If True, use Q_real (for trajectory generation)
                       If False, use Q (for filtering)
    
    Returns:
        Next state x_k, shape (state_dim, 1) - column vector
    """
    Phi = model_params['Phi']
    Q = model_params['Q_real'] if use_real_noise else model_params['Q']
    state_dim = model_params['state_dim']
    
    # Step 1: Deterministic state transition
    x_next = tf.linalg.matmul(Phi, x_prev)
    
    # Step 2: Add Gaussian process noise w ~ N(0, Q)
    z = tf.random.normal([state_dim, 1], dtype=tf.float32)
    Q_chol = tf.linalg.cholesky(Q)
    w = tf.linalg.matmul(Q_chol, z)
    
    # Final state: x_k = Φ * x_{k-1} + w_k
    x_next = x_next + w
    
    return x_next

def observation_model(x, model_params):
    """
    NONLINEAR observation model: z_k = h(x_k) + v_k
    
    MATLAB Reference: Acoustic_hfunc.m
    
    Args:
        x: Current state, shape (state_dim, 1) - column vector
        model_params: Dictionary of model parameters
    
    Returns:
        Observation z, shape (n_sensors, 1) - column vector
    """
    state_dim = model_params['state_dim']
    sensor_positions = model_params['sensor_positions']
    n_sensors = model_params['n_sensors']
    amplitude = model_params['amplitude']
    inv_power = model_params['inv_power']
    d0 = model_params['d0']
    measurement_noise_std = model_params['measurement_noise_std']
    
    # Step 1: Extract target positions from state vector
    x_flat = tf.squeeze(x, axis=1)
    
    x_positions = tf.gather(x_flat, tf.range(0, state_dim, 4))
    y_positions = tf.gather(x_flat, tf.range(1, state_dim, 4))
    
    target_positions = tf.stack([x_positions, y_positions], axis=1)
    
    # Step 2: Compute distances from all targets to all sensors
    sensors_expanded = tf.expand_dims(sensor_positions, axis=1)
    targets_expanded = tf.expand_dims(target_positions, axis=0)
    
    diff = targets_expanded - sensors_expanded
    distances_sq = tf.reduce_sum(tf.square(diff), axis=2)
    distances = tf.sqrt(distances_sq)
    
    # Step 3: Compute amplitude contributions
    amplitudes = amplitude / (tf.pow(distances, inv_power) + d0)
    
    # Step 4: Sum contributions from all targets for each sensor
    z_true = tf.reduce_sum(amplitudes, axis=1)
    
    # Step 5: Add measurement noise
    noise = tf.random.normal(
        shape=[n_sensors],
        mean=0.0,
        stddev=measurement_noise_std,
        dtype=tf.float32
    )
    
    z = z_true + noise
    
    # Return as column vector (n_sensors, 1)
    return tf.reshape(z, [n_sensors, 1])

if __name__ == "__main__":
    # Example usage
    model_params = initialize_acoustic_model()
    
    # Initial target states (example)
    x0_initial_target_states = model_params['x0_initial_target_states']
    
    print('x0_initial_target_states', x0_initial_target_states)
    
    # Simulate one step
    x_next = state_transition(x0_initial_target_states, model_params, use_real_noise=True)
    z = observation_model(x=x_next, model_params=model_params)
    
    print('Next state x_next:', x_next.numpy())
    print('Observation z:', z.numpy())