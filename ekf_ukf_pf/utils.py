"""
Utility functions for filter demonstrations and comparisons.

This module provides common functions used across PF, EKF, and UKF demos:
- Model setup and data generation
- Filter-specific function factories
- Performance metrics computation
- Plotting utilities
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Optional, Dict, List
import tracemalloc
import time

tfd = tfp.distributions


# ============================================================================
# Model Setup and Data Generation
# ============================================================================

def create_model(dt: float = 1.0,
                 process_noise_std_pos: float = 0.1,
                 process_noise_std_vel: float = 0.1,
                 range_noise_std: float = 50.0,
                 bearing_noise_std: float = 0.005,
                 seed: int = 42):
    """
    Create a RangeBearingModel with specified parameters.

    Args:
        dt: Time step
        process_noise_std_pos: Process noise standard deviation for position
        process_noise_std_vel: Process noise standard deviation for velocity
        range_noise_std: Range measurement noise standard deviation
        bearing_noise_std: Bearing measurement noise standard deviation
        seed: Random seed

    Returns:
        RangeBearingModel instance
    """
    from range_bearing_model import RangeBearingModel

    return RangeBearingModel(
        dt=dt,
        process_noise_std_pos=process_noise_std_pos,
        process_noise_std_vel=process_noise_std_vel,
        range_noise_std=range_noise_std,
        bearing_noise_std=bearing_noise_std,
        seed=seed
    )


def generate_trajectory(model, T: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate a trajectory using the model.

    Args:
        model: RangeBearingModel instance
        T: Number of time steps

    Returns:
        Tuple of (true_states, observations)
        - true_states: shape (4, T+1)
        - observations: shape (2, T)
    """
    return model.simulate_trajectory(T=T)


# ============================================================================
# Common Filter Functions
# ============================================================================

def create_state_transition_fn(model):
    """
    Create state transition function for the given model.

    Args:
        model: RangeBearingModel instance

    Returns:
        State transition function
    """
    def state_transition_fn(x, u=None):
        return tf.matmul(model.A, x)
    return state_transition_fn


def create_observation_fn():
    """
    Create observation function for range-bearing measurements.

    Returns:
        Observation function
    """
    def observation_fn(x):
        x_pos, y_pos = x[0, 0], x[2, 0]
        r = tf.sqrt(x_pos**2 + y_pos**2)
        theta = tf.atan2(y_pos, x_pos)
        return tf.stack([r, theta])[:, tf.newaxis]
    return observation_fn


def create_state_jacobian_fn(model):
    """
    Create state transition Jacobian function (for EKF).

    Args:
        model: RangeBearingModel instance

    Returns:
        State transition Jacobian function
    """
    def state_jacobian_fn(x, u=None):
        return model.A
    return state_jacobian_fn


def create_observation_jacobian_fn(model):
    """
    Create observation Jacobian function (for EKF).

    Args:
        model: RangeBearingModel instance

    Returns:
        Observation Jacobian function
    """
    def observation_jacobian_fn(x):
        return model.compute_observation_jacobian(x)
    return observation_jacobian_fn


# ============================================================================
# Particle Filter Specific Functions
# ============================================================================

def create_pf_functions(model, x0, Sigma0, dtype=tf.float64):
    """
    Create all functions needed for Particle Filter.

    Args:
        model: RangeBearingModel instance
        x0: Initial state
        Sigma0: Initial covariance
        dtype: TensorFlow dtype for PF

    Returns:
        Dictionary with keys:
        - state_transition_fn
        - observation_fn
        - process_noise_sampler
        - observation_likelihood_fn
        - x0_sampler
    """
    # State transition (cast to dtype)
    def state_transition_fn_pf(x, u=None):
        return tf.matmul(tf.cast(model.A, dtype), x)

    # Observation
    def observation_fn_pf(x):
        x_pos, y_pos = x[0, 0], x[2, 0]
        r = tf.sqrt(x_pos**2 + y_pos**2)
        theta = tf.atan2(y_pos, x_pos)
        return tf.stack([r, theta])[:, tf.newaxis]

    # Process noise sampler
    def process_noise_sampler(num_samples):
        noise = tf.random.normal([4, num_samples], dtype=dtype)
        Q_sqrt = tf.linalg.cholesky(tf.cast(model.Q, dtype))
        return tf.matmul(Q_sqrt, noise)

    # Observation likelihood
    def observation_likelihood_fn(z, x):
        z_pred = observation_fn_pf(x)
        v = z - z_pred
        R = tf.cast(model.R, dtype)
        R_inv = tf.linalg.inv(R)
        exponent = -0.5 * tf.matmul(tf.transpose(v), tf.matmul(R_inv, v))[0, 0]
        return tf.exp(exponent)

    # Initial state sampler
    def x0_sampler(num_samples):
        x0_casted = tf.cast(x0, dtype)
        Sigma0_casted = tf.cast(Sigma0, dtype)
        x0_flat = tf.reshape(x0_casted, [-1])
        initial_dist = tfd.MultivariateNormalTriL(
            loc=x0_flat,
            scale_tril=tf.linalg.cholesky(Sigma0_casted)
        )
        return tf.transpose(initial_dist.sample(num_samples))

    return {
        'state_transition_fn': state_transition_fn_pf,
        'observation_fn': observation_fn_pf,
        'process_noise_sampler': process_noise_sampler,
        'observation_likelihood_fn': observation_likelihood_fn,
        'x0_sampler': x0_sampler
    }


def compute_path_degeneracy(ancestry_history):
    """
    Compute path degeneracy measure for Particle Filter.

    For each time t, compute how many distinct ancestors of the
    final generation (time T) are still present at time t.

    Args:
        ancestry_history: Tensor of shape (num_particles, T+1),
            ancestry_history[i, t] is the ancestor index at time t-1
            of particle i at time t (with column 0 = [0,1,...,N-1]).

    Returns:
        num_unique_ancestors: Tensor of shape (T+1,),
            where entry t is the number of distinct ancestors at time t
            that lead to the final generation.
    """
    ancestry_history = tf.convert_to_tensor(ancestry_history, dtype=tf.int32)

    num_particles = ancestry_history.shape[0]
    T_plus_1 = ancestry_history.shape[1]
    T = T_plus_1 - 1

    # Start from final generation at time T
    ancestors = tf.range(num_particles, dtype=tf.int32)
    num_unique = []

    # Walk backwards in time: T, T-1, ..., 0
    for t in range(T, -1, -1):
        unique_ancestors = tf.unique(ancestors)[0]
        num_unique.append(tf.size(unique_ancestors))

        if t > 0:
            # ancestry_history[:, t] maps particle index at time t
            # to its parent index at time t-1.
            parents_t = ancestry_history[:, t]
            ancestors = tf.gather(parents_t, ancestors)

    # Reverse to get [0, 1, ..., T]
    num_unique = tf.stack(num_unique[::-1])
    return tf.cast(num_unique, tf.float64)


# ============================================================================
# Performance Metrics
# ============================================================================

def compute_position_error(filtered_states: tf.Tensor,
                          true_states: tf.Tensor,
                          exclude_first: bool = True) -> tf.Tensor:
    """
    Compute position error between filtered and true states.

    Args:
        filtered_states: Filtered state estimates, shape (4, T+1)
        true_states: True states, shape (4, T+1)
        exclude_first: Whether to exclude the first state (t=0)

    Returns:
        Position errors, shape (T,) if exclude_first else (T+1,)
    """
    if exclude_first:
        pos_error = tf.sqrt(
            (filtered_states[0, 1:] - true_states[0, 1:])**2 +
            (filtered_states[2, 1:] - true_states[2, 1:])**2
        )
    else:
        pos_error = tf.sqrt(
            (filtered_states[0, :] - true_states[0, :])**2 +
            (filtered_states[2, :] - true_states[2, :])**2
        )
    return pos_error


def compute_position_rmse(filtered_states: tf.Tensor,
                         true_states: tf.Tensor,
                         exclude_first: bool = True) -> float:
    """
    Compute position RMSE.

    Args:
        filtered_states: Filtered state estimates, shape (4, T+1)
        true_states: True states, shape (4, T+1)
        exclude_first: Whether to exclude the first state (t=0)

    Returns:
        Position RMSE as a float
    """
    pos_error = compute_position_error(filtered_states, true_states, exclude_first)
    rmse = tf.sqrt(tf.reduce_mean(pos_error**2))
    return rmse.numpy()


def compute_metrics(filtered_states: tf.Tensor,
                   true_states: tf.Tensor,
                   name: str,
                   verbose: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.

    Args:
        filtered_states: Filtered state estimates, shape (4, T+1)
        true_states: True states, shape (4, T+1)
        name: Name of the filter (for printing)
        verbose: Whether to print metrics

    Returns:
        Dictionary with keys: pos_rmse, vel_rmse, avg_pos_error, pos_error
    """
    filtered_states = tf.cast(filtered_states, tf.float32)
    true_states = tf.cast(true_states, tf.float32)

    # Position RMSE
    pos_error = tf.sqrt(
        (filtered_states[0, 1:] - true_states[0, 1:])**2 +
        (filtered_states[2, 1:] - true_states[2, 1:])**2
    )
    pos_rmse = tf.sqrt(tf.reduce_mean(pos_error**2))

    # Velocity RMSE
    vel_error = tf.sqrt(
        (filtered_states[1, 1:] - true_states[1, 1:])**2 +
        (filtered_states[3, 1:] - true_states[3, 1:])**2
    )
    vel_rmse = tf.sqrt(tf.reduce_mean(vel_error**2))

    # Average position error
    avg_pos_error = tf.reduce_mean(pos_error)

    if verbose:
        print(f"\n{name} Metrics:")
        print(f"  Position RMSE: {pos_rmse.numpy():.4f}")
        print(f"  Velocity RMSE: {vel_rmse.numpy():.4f}")
        print(f"  Avg Position Error: {avg_pos_error.numpy():.4f}")

    return {
        'pos_rmse': pos_rmse.numpy(),
        'vel_rmse': vel_rmse.numpy(),
        'avg_pos_error': avg_pos_error.numpy(),
        'pos_error': pos_error.numpy()
    }


def measure_performance(filter_fn: Callable,
                       name: str = "Filter") -> Tuple[float, float, any]:
    """
    Measure runtime and memory usage of a filter.

    Args:
        filter_fn: Function that runs the filter (takes no arguments)
        name: Name of the filter (for printing)

    Returns:
        Tuple of (runtime, peak_memory_mb, filter_output)
    """
    tracemalloc.start()
    start_time = time.time()

    result = filter_fn()

    runtime = time.time() - start_time
    peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()

    print(f"{name} Runtime: {runtime:.4f} seconds")
    print(f"{name} Peak Memory: {peak_memory:.2f} MB")

    return runtime, peak_memory, result


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_2d_trajectory(true_states: np.ndarray,
                      filtered_states: Dict[str, np.ndarray],
                      ax: Optional[plt.Axes] = None,
                      title: str = '2D Trajectory') -> plt.Axes:
    """
    Plot 2D trajectory comparison.

    Args:
        true_states: True states, shape (4, T+1)
        filtered_states: Dictionary mapping filter names to filtered states
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot true trajectory
    ax.plot(true_states[0, :], true_states[2, :],
            'g-', linewidth=2, label='True', alpha=0.7)

    # Plot filtered trajectories
    colors = {'EKF': 'r--', 'UKF': 'b--', 'PF': 'm--'}
    for name, states in filtered_states.items():
        linestyle = colors.get(name, 'k--')
        ax.plot(states[0, :], states[2, :],
                linestyle, linewidth=2, label=name, alpha=0.7)

    # Mark start
    ax.plot(true_states[0, 0], true_states[2, 0],
            'go', markersize=10, label='Start')

    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    return ax


def plot_position_error(pos_errors: Dict[str, np.ndarray],
                       ax: Optional[plt.Axes] = None,
                       title: str = 'Position Error Over Time') -> plt.Axes:
    """
    Plot position error over time for multiple filters.

    Args:
        pos_errors: Dictionary mapping filter names to position errors
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Predicted': 'r-', 'Filtered': 'b-', 'EKF': 'r-',
              'UKF': 'b-', 'PF': 'g-'}

    for name, errors in pos_errors.items():
        time_steps = np.arange(1, len(errors) + 1)
        linestyle = colors.get(name, 'k-')
        ax.plot(time_steps, errors, linestyle,
                linewidth=2, label=name, alpha=0.7)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Position Error')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_state_component(true_states: np.ndarray,
                        filtered_states: Dict[str, np.ndarray],
                        component_idx: int,
                        component_name: str,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a single state component (x, y, vx, or vy).

    Args:
        true_states: True states, shape (4, T+1)
        filtered_states: Dictionary mapping filter names to filtered states
        component_idx: Index of state component (0=x, 1=vx, 2=y, 3=vy)
        component_name: Name of component for labeling
        ax: Matplotlib axes (creates new if None)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    T = true_states.shape[1] - 1
    state_time = np.arange(0, T + 1)

    # Plot true trajectory
    ax.plot(state_time, true_states[component_idx, :],
            'g-', linewidth=2, label='True', alpha=0.7)

    # Plot filtered trajectories
    colors = {'EKF': 'r--', 'UKF': 'b--', 'PF': 'm--'}
    for name, states in filtered_states.items():
        linestyle = colors.get(name, 'k--')
        ax.plot(state_time, states[component_idx, :],
                linestyle, linewidth=2, label=name, alpha=0.7)

    ax.set_xlabel('Time step')
    ax.set_ylabel(component_name)
    ax.set_title(component_name)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_full_comparison(true_states: np.ndarray,
                        filtered_states: Dict[str, np.ndarray],
                        pos_errors: Dict[str, np.ndarray],
                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a comprehensive 2x2 plot with trajectory and state components.

    Args:
        true_states: True states, shape (4, T+1)
        filtered_states: Dictionary mapping filter names to filtered states
        pos_errors: Dictionary mapping filter names to position errors
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 2D Trajectory
    plot_2d_trajectory(true_states, filtered_states, ax=axes[0, 0])

    # Position Error
    plot_position_error(pos_errors, ax=axes[0, 1])

    # X Position
    plot_state_component(true_states, filtered_states, 0, 'x position', ax=axes[1, 0])

    # Y Position
    plot_state_component(true_states, filtered_states, 2, 'y position', ax=axes[1, 1])

    plt.tight_layout()
    return fig


def plot_velocity_comparison(true_states: np.ndarray,
                            filtered_states: Dict[str, np.ndarray],
                            figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a 2x1 plot showing X and Y velocity components.

    Args:
        true_states: True states, shape (4, T+1)
        filtered_states: Dictionary mapping filter names to filtered states
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # X Speed
    plot_state_component(true_states, filtered_states, 1, 'X Speed', ax=axes[0])

    # Y Speed
    plot_state_component(true_states, filtered_states, 3, 'Y Speed', ax=axes[1])

    plt.tight_layout()
    return fig


def plot_performance_comparison(metrics: Dict[str, Dict[str, float]],
                               metric_names: List[str] = ['pos_rmse', 'runtime', 'memory'],
                               metric_labels: List[str] = ['Position RMSE', 'Runtime (s)', 'Peak Memory (MB)'],
                               figsize: Tuple[int, int] = (18, 5)) -> plt.Figure:
    """
    Create bar charts comparing filter performance metrics.

    Args:
        metrics: Dictionary mapping filter names to metric dictionaries
        metric_names: List of metric keys to plot
        metric_labels: List of metric labels for plots
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
    if len(metric_names) == 1:
        axes = [axes]

    filters = list(metrics.keys())
    colors = {'EKF': '#E74C3C', 'UKF': '#3498DB', 'PF': '#2ECC71', 'PF (N=100)': '#2ECC71'}
    filter_colors = [colors.get(f, '#95A5A6') for f in filters]

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        values = [metrics[f][metric_name] for f in filters]
        bars = ax.bar(filters, values, color=filter_colors,
                     alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric_name == 'runtime':
                label = f'{val:.3f}s'
            elif metric_name == 'memory':
                label = f'{val:.1f} MB'
            else:
                label = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2, height + height*0.02,
                   label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_pf_degeneracy(num_unique_ancestors: np.ndarray,
                      ax: Optional[plt.Axes] = None,
                      title: str = 'Path degeneracy (lineage diversity over time)') -> plt.Axes:
    """
    Plot particle filter path degeneracy.

    Args:
        num_unique_ancestors: Array of unique ancestor counts over time
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    time_axis = np.arange(len(num_unique_ancestors))
    ax.plot(time_axis, num_unique_ancestors, marker='o')
    ax.set_xlabel('time t')
    ax.set_ylabel('number of unique ancestors')
    ax.set_title(title)
    ax.grid(True)

    return ax


def plot_particle_count_sensitivity(pf_results: List[Dict],
                                   figsize: Tuple[int, int] = (18, 5)) -> plt.Figure:
    """
    Plot PF performance sensitivity to particle count.

    Args:
        pf_results: List of dictionaries with keys: particles, rmse, runtime, memory
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    particles = [r['particles'] for r in pf_results]
    rmses = [r['rmse'] for r in pf_results]
    runtimes = [r['runtime'] for r in pf_results]
    memories = [r['memory'] for r in pf_results]

    # RMSE vs Particles
    axes[0].plot(particles, rmses, 'o-', linewidth=2, markersize=8, color='#2ECC71')
    axes[0].set_xlabel('Number of Particles', fontsize=12)
    axes[0].set_ylabel('Position RMSE', fontsize=12)
    axes[0].set_title('PF Accuracy vs Particle Count', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Runtime vs Particles
    axes[1].plot(particles, runtimes, 'o-', linewidth=2, markersize=8, color='#E74C3C')
    axes[1].set_xlabel('Number of Particles', fontsize=12)
    axes[1].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[1].set_title('PF Runtime vs Particle Count', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Memory vs Particles
    axes[2].plot(particles, memories, 'o-', linewidth=2, markersize=8, color='#3498DB')
    axes[2].set_xlabel('Number of Particles', fontsize=12)
    axes[2].set_ylabel('Peak Memory (MB)', fontsize=12)
    axes[2].set_title('PF Memory vs Particle Count', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
