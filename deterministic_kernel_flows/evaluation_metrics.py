import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from itertools import permutations

tfd = tfp.distributions


################################## ------ Plot function -------##################################################

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

    d_p(X, \hat{X}) = \left(\frac{1}{C} \min_{\pi \in \Pi} \sum_{c=1}^C d(x_c, \hat{x}_{\pi(c)})^p\right)^{1/p}
    
    Args:
        true_states: TensorFlow tensor, shape (state_dim,) or (state_dim, T)
        estimated_states: TensorFlow tensor, same shape
        num_targets: number of target 
        n_target: Dimension of each target's state; # Each target has 2 state dimensions (e.g., x, y)
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


def find_acoustic_position(estimates, ground_truth):
    """
    Extracts position coordinates from acoustic estimates and ground truth data.
    
    This function processes 4 acoustic sources by extracting their 2D position 
    coordinates from interleaved data arrays. It assumes that position data for 
    each source is stored as 2 consecutive rows at indices [0:2], [4:6], [8:10], 
    and [12:14], suggesting the input arrays contain other acoustic parameters 
    (e.g., velocity, acceleration) interspersed with position data.
    
    Args:
        estimates: A tensor or array where position estimates are stored at 
                   indices [i*4:i*4+2] for each source i (shape: [16+, num_samples])
        ground_truth: A tensor or array with the same structure as estimates, 
                      containing ground truth position values
    
    Returns:
        tuple: A tuple containing:
            - estimate_position: Concatenated tensor of all estimated positions 
                                (shape: [8, num_samples])
            - ground_truth_position: Concatenated tensor of all ground truth positions 
                                    (shape: [8, num_samples])
    
    Example:
        If processing 4 sound sources with x,y coordinates stored every 4 rows,
        this extracts rows [0:2, 4:6, 8:10, 12:14] from both inputs and 
        concatenates them into [8, :] shaped tensors.
    """
    estimate_position = []
    ground_truth_position = []
    for i in range(4):
        index_p = i * 4
        position_s = estimates[index_p:index_p+2, :]
        estimate_position.append(position_s)
        position_g = ground_truth[index_p:index_p+2, :]
        ground_truth_position.append(position_g)
    estimate_position = tf.concat(estimate_position, axis=0)
    ground_truth_position = tf.concat(ground_truth_position, axis=0)
    return estimate_position, ground_truth_position


def compute_position_error_noraml_acousitc(estimates, ground_truth, n_targets):
    errors = []
    for i in range(n_targets):
        x_idx = i * 4
        y_idx = i * 4 + 1
        x_err = estimates[x_idx, :] - ground_truth[x_idx, :]
        y_err = estimates[y_idx, :] - ground_truth[y_idx, :]
        error = tf.sqrt(x_err**2 + y_err**2)
        errors.append(error)
    errors = tf.stack(errors, axis=0)
    mean_error = tf.reduce_mean(errors, axis=0)
    return errors, mean_error
