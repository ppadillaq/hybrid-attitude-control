# hybrid-attitude-control
Simulation and visualization of hybrid spacecraft attitude control using quaternions, rigid-body dynamics, and hybrid control laws for global attitude stabilization.

<p align="center">
  <img src="assets/hybrid_attitude_control.gif"
       alt="Hybrid spacecraft attitude control simulation"
       width="900">
</p>

The visualization shows the spacecraft attitude evolution together with the
logic variable, quaternion state, angular velocity, and cumulative control
effort.

## Overview

This project implements and visualizes hybrid attitude control algorithms for
rigid spacecraft using unit quaternions and nonlinear rigid-body dynamics.

The implementation is based on the hybrid control framework proposed by
Mayhew, Sanfelice, and Teel [1] for robust global asymptotic attitude
stabilization.

The project explores several key concepts in nonlinear and hybrid control,
including:

- Quaternion-based attitude representation and kinematics
- Nonlinear rigid-body rotational dynamics
- Lyapunov-based stability analysis
- Hybrid dynamical systems with continuous flows and discrete jumps
- Hysteresis-based switching logic
- Global asymptotic attitude stabilization
- Avoidance of the quaternion unwinding phenomenon
- Energy-based nonlinear attitude control
- Backstepping control
- Robustness against quaternion measurement noise
- Chattering and the effect of hysteresis
- Control-effort analysis
- Comparison of different hybrid controller configurations
- Synchronized 3D spacecraft visualization using PyVista

The project combines mathematical modelling, nonlinear control theory,
numerical simulation, and 3D visualization to provide both a technical
implementation and an intuitive demonstration of hybrid spacecraft attitude
control.

This repository is designed as both:

- a technical simulation project
- a research-oriented visual demonstrator for hybrid nonlinear control

## Motivation

Classical continuous attitude controllers on SO(3) face topological limitations that prevent global asymptotic stabilization using smooth feedback alone.

Hybrid control overcomes this issue by introducing a discrete logic variable that enables global stabilization without unwinding.

This project aims to provide an intuitive and visual implementation of these ideas for engineering, learning, and research purposes.

## Mathematical Model
### Attitude Representation

The spacecraft attitude is represented using unit quaternions:

```math
q = (\eta, \epsilon) \in \mathbb{S}^3
```

where:

- $\eta$ is the scalar part
- $\epsilon$ is the vector part

### Kinematics

```math
\dot{q} = \frac{1}{2} q \otimes \nu(\omega)
```

where:

- $\omega$ is the angular velocity
- $\otimes$ denotes quaternion multiplication

### Rigid-Body Dynamics

```math
J\dot{\omega} = -\omega \times J\omega + u
```

where:

- $J$ is the inertia matrix
- $u$ is the control torque

### Hybrid Logic

A binary logic variable is introduced:
```math
h \in \{-1, 1\}
```
with flow and jump sets:
```math
C = \{(q,h): h\eta \geq -\delta\} \\
D = \{(q,h): h\eta \leq -\delta\}
```
and jump map:
```math
h^+ = -h
```
This switching mechanism avoids unwinding and guarantees global stabilization.


## References

[1] C. G. Mayhew, R. G. Sanfelice, and A. R. Teel,
"Robust global asymptotic attitude stabilization of a rigid body by
quaternion-based hybrid feedback," 2009.