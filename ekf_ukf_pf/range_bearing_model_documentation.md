# Range-Bearing Non-Linear State-Space Model

## Mathematical Formulation

This document provides a detailed mathematical description of the Range-Bearing tracking model, a non-linear and non-Gaussian state-space model commonly used in radar tracking and robot localization.

---

## 1. Model Overview

The Range-Bearing model is a **state-space model** with:
- **Linear motion model** (constant velocity in Cartesian coordinates)
- **Non-linear observation model** (polar coordinates: range and bearing)

The non-linearity in the observation function, combined with additive Gaussian noise, produces a **non-Gaussian posterior distribution**, making this model suitable for testing non-linear filtering algorithms such as:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)

---

## 2. State Space

### State Vector

The state at time $t$ is a 4-dimensional vector representing 2D position and velocity:

$$
\mathbf{x}_t = \begin{bmatrix} x_t \\ \dot{x}_t \\ y_t \\ \dot{y}_t \end{bmatrix} \in \mathbb{R}^4
$$

where:
- $x_t$: position in x-direction (horizontal)
- $\dot{x}_t$: velocity in x-direction
- $y_t$: position in y-direction (vertical)
- $\dot{y}_t$: velocity in y-direction

---

## 3. Motion Model (State Transition)

The motion model describes how the state evolves over time. We use a **linear constant velocity model**:

$$
\mathbf{x}_t = A \mathbf{x}_{t-1} + \mathbf{w}_t
$$

where $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q)$ is the **process noise**.

### State Transition Matrix

$$
A = \begin{bmatrix}
1 & \Delta t & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & \Delta t \\
0 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

This matrix implements the discrete-time constant velocity model:

$$
\begin{aligned}
x_t &= x_{t-1} + \Delta t \cdot \dot{x}_{t-1} + w_{x,t} \\
\dot{x}_t &= \dot{x}_{t-1} + w_{\dot{x},t} \\
y_t &= y_{t-1} + \Delta t \cdot \dot{y}_{t-1} + w_{y,t} \\
\dot{y}_t &= \dot{y}_{t-1} + w_{\dot{y},t}
\end{aligned}
$$

### Process Noise Covariance

$$
Q = \begin{bmatrix}
\sigma_{x}^2 & 0 & 0 & 0 \\
0 & \sigma_{\dot{x}}^2 & 0 & 0 \\
0 & 0 & \sigma_{y}^2 & 0 \\
0 & 0 & 0 & \sigma_{\dot{y}}^2
\end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

where:
- $\sigma_{x}^2, \sigma_{y}^2$: variance of position noise
- $\sigma_{\dot{x}}^2, \sigma_{\dot{y}}^2$: variance of velocity noise

**Note**: Process noise affects both position and velocity, modeling uncertainty in the motion.

---

## 4. Observation Model (Measurement)

The observation model is **non-linear**, converting Cartesian state to polar coordinates:

$$
\mathbf{z}_t = h(\mathbf{x}_t) + \mathbf{v}_t
$$

where $\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R)$ is the **measurement noise**.

### Observation Function

$$
h(\mathbf{x}_t) = \begin{bmatrix} r_t \\ \theta_t \end{bmatrix} = \begin{bmatrix} \sqrt{x_t^2 + y_t^2} \\ \arctan2(y_t, x_t) \end{bmatrix} \in \mathbb{R}^2
$$

where:
- $r_t$: **range** (distance from origin to object)
- $\theta_t$: **bearing** (angle from positive x-axis)

### Measurement Components

**Range (distance):**
$$
r_t = \sqrt{x_t^2 + y_t^2}
$$

**Bearing (angle):**
$$
\theta_t = \arctan2(y_t, x_t) = \begin{cases}
\arctan\left(\frac{y_t}{x_t}\right) & \text{if } x_t > 0 \\
\arctan\left(\frac{y_t}{x_t}\right) + \pi & \text{if } x_t < 0, y_t \geq 0 \\
\arctan\left(\frac{y_t}{x_t}\right) - \pi & \text{if } x_t < 0, y_t < 0 \\
+\frac{\pi}{2} & \text{if } x_t = 0, y_t > 0 \\
-\frac{\pi}{2} & \text{if } x_t = 0, y_t < 0 \\
\text{undefined} & \text{if } x_t = 0, y_t = 0
\end{cases}
$$

The $\arctan2$ function returns angles in the range $(-\pi, \pi]$.

### Measurement Noise Covariance

$$
R = \begin{bmatrix}
\sigma_r^2 & 0 \\
0 & \sigma_\theta^2
\end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

where:
- $\sigma_r^2$: variance of range measurement noise
- $\sigma_\theta^2$: variance of bearing measurement noise (in radians)

---

## 5. Observation Jacobian (for EKF)

For the Extended Kalman Filter, we need the **Jacobian matrix** of the observation function:

$$
H_t = \frac{\partial h}{\partial \mathbf{x}}\bigg|_{\mathbf{x} = \mathbf{x}_t} = \begin{bmatrix}
\frac{\partial r}{\partial x} & \frac{\partial r}{\partial \dot{x}} & \frac{\partial r}{\partial y} & \frac{\partial r}{\partial \dot{y}} \\
\frac{\partial \theta}{\partial x} & \frac{\partial \theta}{\partial \dot{x}} & \frac{\partial \theta}{\partial y} & \frac{\partial \theta}{\partial \dot{y}}
\end{bmatrix}
$$

### Partial Derivatives for Range

For $r = \sqrt{x^2 + y^2}$:

$$
\frac{\partial r}{\partial x} = \frac{x}{\sqrt{x^2 + y^2}} = \frac{x}{r} = \cos(\theta)
$$

$$
\frac{\partial r}{\partial y} = \frac{y}{\sqrt{x^2 + y^2}} = \frac{y}{r} = \sin(\theta)
$$

$$
\frac{\partial r}{\partial \dot{x}} = 0, \quad \frac{\partial r}{\partial \dot{y}} = 0
$$

### Partial Derivatives for Bearing

For $\theta = \arctan2(y, x)$:

$$
\frac{\partial \theta}{\partial x} = \frac{-y}{x^2 + y^2} = \frac{-y}{r^2} = -\frac{\sin(\theta)}{r}
$$

$$
\frac{\partial \theta}{\partial y} = \frac{x}{x^2 + y^2} = \frac{x}{r^2} = \frac{\cos(\theta)}{r}
$$

$$
\frac{\partial \theta}{\partial \dot{x}} = 0, \quad \frac{\partial \theta}{\partial \dot{y}} = 0
$$

### Complete Jacobian Matrix

$$
H_t = \begin{bmatrix}
\cos(\theta_t) & 0 & \sin(\theta_t) & 0 \\
-\frac{\sin(\theta_t)}{r_t} & 0 & \frac{\cos(\theta_t)}{r_t} & 0
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

---

## 6. Initial State Distribution

The initial state $\mathbf{x}_0$ is sampled from a Gaussian distribution:

$$
\mathbf{x}_0 \sim \mathcal{N}(\mathbf{\mu}_0, P_0)
$$

where:
- $\mathbf{\mu}_0 = \mathbf{0}$ (zero mean)
- $P_0 = \text{diag}(\sigma_{x_0}^2, \sigma_{\dot{x}_0}^2, \sigma_{y_0}^2, \sigma_{\dot{y}_0}^2)$ (diagonal covariance)

Typically:
- $\sigma_{x_0} = \sigma_{y_0} = 100$ (initial position uncertainty)
- $\sigma_{\dot{x}_0} = \sigma_{\dot{y}_0} = 10$ (initial velocity uncertainty)

---

## 7. Trajectory Generation

Following the convention from `lgss_sample.py` in kf_lgssm folder, a trajectory of length $T$ consists of:

### States: $\mathbf{x}_{0:T}$

$$
\{\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\}
$$

- Total: $T+1$ states
- Shape: $(4, T+1, N)$ where $N$ is the number of samples/trajectories

### Observations: $\mathbf{z}_{1:T}$

$$
\{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T\}
$$

- Total: $T$ observations
- Shape: $(2, T, N)$
- **Important**: There is no observation $\mathbf{z}_0$ for the initial state $\mathbf{x}_0$

### Generation Process

1. Sample initial state: $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, P_0)$
2. For $t = 1, 2, \ldots, T$:
   - State transition: $\mathbf{x}_t = A\mathbf{x}_{t-1} + \mathbf{w}_t$, where $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q)$
   - Generate observation: $\mathbf{z}_t = h(\mathbf{x}_t) + \mathbf{v}_t$, where $\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R)$


### Note:

The system is **observable** if the state can be uniquely determined from observations. For the Range-Bearing model:

The **observability matrix** is rank-deficient because range and bearing do not provide information about velocities $\dot{x}$ and $\dot{y}$ instantaneously. However, the system is **asymptotically observable** over time because the changing positions allow velocity estimation.
