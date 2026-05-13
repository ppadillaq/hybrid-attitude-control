# hybrid-attitude-control
Simulation and visualization of hybrid spacecraft attitude control using quaternions, rigid-body dynamics, and hybrid control laws for global attitude stabilization.

## Overview

This project explores hybrid attitude control for rigid spacecraft using quaternion-based representations and switching logic inspired by the work of Mayhew, Sanfelice, and Teel.

The main objective is to demonstrate global asymptotic attitude stabilization while avoiding the unwinding phenomenon through hybrid control techniques on SO(3).

The project combines:

- Quaternion-based attitude kinematics
- Rigid-body rotational dynamics
- Hybrid control laws with flow and jump sets
- Lyapunov-based stability analysis
- 3D spacecraft visualization using PyVista
- Interactive simulation and control analysis in Python

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