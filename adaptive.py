"""
n-body Simulation and Adaptive Integrators Module

This module provides numerical methods for simulating the dynamics of N-body
gravitational systems. It includes:

- Standard fixed-step schemes: Euler, Euler-Cromer, Leapfrog, Runge-Kutta 4, Forest-Ruth, PEFRL.
- Adaptive timestep methods: RKF45, adaptive Leapfrog, and adaptive symplectic integrators.
- Functions to compute forces, accelerations, energies, and angular momentum.
- Utilities for center-of-mass correction and trajectory handling.
- Kepler orbit functions for two-body comparisons.

All functions are documented in NumPy/SciPy style.

Dependencies
------------
numpy
scipy

Usage
-----
Import the module and call `run_scheme` or `run_adaptive_scheme` with the desired
numerical scheme and initial conditions.
"""

import numpy as np
import scipy as sci
import scipy.integrate
from functions import *
import time

# ============================
#     Adaptive Step Size
# ============================

def findh(rs, vs, h0 = np.inf):
    """
    Estimate a next timestep from pairwise relative speeds and separations.

    Parameters
    ----------
    rs : array, shape (N, dim). Positions of N bodies.
    vs : array, shape (N, dim). Velocities of N bodies.
    h0 : float, optional upper bound for returned timestep (default: +inf).

    Returns
    -------
    h : float
    """

    h = h0
    N = len(rs)
    for i in range(N):
        for j in range(i+1, N):
                r_mag = np.linalg.norm(rs[i] - rs[j])
                v_mag = np.linalg.norm(vs[i] - vs[j])
                h = min(h, r_mag / v_mag)
    return h
    
def AdaptiveLeapfrog(t0, r0s, v0s, h0, G, masses, tolerance = 1e-3, safety_factor = 0.9, min_scale = 0.1, max_scale = 5):
    """
    Perform a single adaptive timestep using the Leapfrog integrator.

    Parameters
    ----------
    t0 : float. Current time.
    r0s : ndarray, shape (N, 3). Positions of all particles at t0.
    v0s : ndarray, shape (N, 3). Velocities of all particles at t0.
    h0 : float. Current timestep.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.
    tolerance : float, optional. Error tolerance (default: 1e-3).
    safety_factor : float, optional. Scaling factor for timestep adaptation (default: 0.9).
    min_scale : float, optional. Minimum allowed timestep scaling factor (default: 0.1).
    max_scale : float, optional. Maximum allowed timestep scaling factor (default: 5).

    Returns
    -------
    t1 : float. Updated time.
    h1 : float. Next suggested timestep.
    rs : ndarray, shape (N, 3). Updated positions.
    vs : ndarray, shape (N, 3). Updated velocities.
    """

    vs_half = v0s + 0.5 * dv_dt(r0s, G, masses) * h0
    r1s = r0s + vs_half * h0
    v1s = vs_half + 0.5 * dv_dt(r1s, G, masses) * h0

    # estimate local truncation error
    error = max(1e-7, np.linalg.norm(r1s - r0s))
    
    if error <= tolerance:
        h1 = h0 * max(0.001, min(max_scale, max(min_scale, safety_factor * (tolerance / error)**0.2)))
        t1 = t0 + h0
        rs = r1s
        vs = v1s
    else:
        t1 = t0
        h1 = h0 * max(0.001, min(max_scale, max(min_scale, safety_factor * (tolerance / error)**0.2)))
        rs = r0s
        vs = v0s
    
    return t1, h1, rs, vs
    
def RKF45Step(t0, r0s, v0s, h0, G, masses, tolerance = 1e-6, safety_factor = 0.9, min_scale = 0.1, max_scale = 5):
    """
    Perform a single timestep using the adaptive Runge-Kutta-Fehlberg 4(5) method.

    Parameters
    ----------
    t0 : float. Current time.
    r0s : ndarray, shape (N, 3). Positions of all particles at t0.
    v0s : ndarray, shape (N, 3). Velocities of all particles at t0.
    h0 : float. Current timestep.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.
    tolerance : float, optional. Error tolerance for adaptive step (default: 1e-6).
    safety_factor : float, optional. Scaling factor for timestep adaptation (default: 0.9).
    min_scale : float, optional. Minimum allowed timestep scaling factor (default: 0.1).
    max_scale : float, optional. Maximum allowed timestep scaling factor (default: 5).

    Returns
    -------
    t1 : float. Updated time.
    h1 : float. Next suggested timestep.
    r1s : ndarray, shape (N, 3). Updated positions.
    v1s : ndarray, shape (N, 3). Updated velocities.
    """

    # butcher tablaeu for RKF45
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    A = np.array([[0, 0, 0, 0, 0, 0],
                  [1/4, 0, 0, 0, 0, 0],
                  [3/32, 9/32, 0, 0, 0, 0],
                  [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                  [439/216, -8, 3680/513, -845/4104, 0, 0],
                  [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]])
    b1 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    b2 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    
    # transform to one long vector 
    w0 = vec_to_w(r0s, v0s)
    
    k1 = all_derivatives(w0, t0, G, masses)

    k2 = all_derivatives(w0 + A[1,0] * h0 * k1, t0 + c[1] * h0, G, masses)

    k3 = all_derivatives(w0 + A[2,0] * h0 * k1 + A[2,1] * h0 * k2, t0 + c[2] * h0, G, masses)

    k4 = all_derivatives(w0 + A[3,0] * h0 * k1 + A[3,1] * h0 * k2 + A[3,2] * h0 * k3, t0 + c[3] * h0, G, masses)

    k5 = all_derivatives(w0 + A[4,0] * h0 * k1 + A[4,1] * h0 * k2 + A[4,2] * h0 * k3 - A[4,3] * h0 * k4, t0 + c[4] * h0, G, masses)

    k6 = all_derivatives(w0 + A[5,0] * h0 * k1 + A[5,1] * h0 * k2 - A[5,2] * h0 * k3 + A[5,3] * h0 * k4 + A[5,4] * h0 * k5, t0 + c[5] * h0, G, masses)

    ks = np.array([k1, k2, k3, k4, k5, k6])
    
    w1_4th = w0 + h0 * (k1 * b1[0] + k2 * b1[1] + k3 * b1[2] + k4 * b1[3] + k5 * b1[4] + k6 * b1[5])
    w1_5th = w0 + h0 * (k1 * b2[0] + k2 * b2[1] + k3 * b2[2] + k4 * b2[3] + k5 * b2[4] + k6 * b2[5])
    
    error = np.abs(w1_5th - w1_4th)
    
    w1 = w0 
    t1 = t0
    
    if np.max(error) <= tolerance:
        t1 += h0
        w1 = w1_5th
    
    if np.max(error) == 0: 
        h1 = h0 * max_scale
    else:
        h1 = h0 * max(0.001, min(max_scale, max(min_scale, safety_factor * (tolerance / np.max(error))**0.2)))
    
    r1s, v1s = w_to_vec(w1)
    
    return t1, h1, r1s, v1s

def run_adaptive_scheme(scheme, t0, T, h0, r0s, v0s, G, masses, tolerance = 1e-3):
    """
    Integrate an n-body system using an adaptive timestep scheme.

    Parameters
    ----------
    scheme : callable. Function implementing a single adaptive timestep (e.g., AdaptiveLeapfrog or RKF45Step).
    t0 : float. Initial time.
    T : float. Final integration time.
    h0 : float. Initial timestep.
    r0s : ndarray, shape (N, 3). Initial positions of particles.
    v0s : ndarray, shape (N, 3). Initial velocities of particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.
    tolerance : float, optional. Error tolerance for adaptive step (default: 1e-3).

    Returns
    -------
    trajectories: tuple
        - t_vals : list of floats. Times at which the solution was evaluated.
        - rs_traj : ndarray, shape (M, N, 3). Trajectory of particle positions.
        - vs_traj : ndarray, shape (M, N, 3). Trajectory of particle velocities.
        - E_traj : ndarray, shape (M,). Total energy of the system over time.
        - am_traj : ndarray, shape (M, N, 3). Angular momentum of particles over time.
        - times : float. Cumulative CPU time spent integrating.
    """
    
    # reposition centre of mass to origin with no momentum 
    rcom, vcom = CentreOfMass(r0s, v0s, masses)
    r0s -= rcom
    v0s -= vcom
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    # Initialize our saved trajectories to be blank 
    t_vals = []
    rs_traj = [] 
    vs_traj = [] 
    E_traj = []
    am_traj = []
    times = 0 
    
    # run scheme for requried number of steps 
    while t <= T:
        t1 = time.time()
        t, h, rs, vs = scheme(t, rs, vs, h, G, masses, tolerance)  # Update step
        times += time.time() - t1
        E = TotalEnergy(rs, vs, G, masses)
        am = AngMomentum(rs, vs, masses) # Calculate angular momentum 
        
        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj = E_traj + [E]
        am_traj = am_traj + [am]
        
    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    E_traj = np.array(E_traj)
    am_traj = np.array(am_traj)
    
    # reposition centre of mass to origin with no momentum 
    rs_traj = np.array([rs + rcom for rs in rs_traj])
    vs_traj = np.array([vs + vcom for vs in vs_traj])
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)


def run_adaptive(scheme, t0, T, h0, r0s, v0s, G, masses, scaleh = 0.005):
    """
    Integrate an n-body system with a symplectic integrator and dynamically scaled timestep.

    Parameters
    ----------
    scheme : callable. Symplectic numerical scheme function.
    t0 : float. Initial time.
    T : float. Final integration time.
    h0 : float. Initial timestep.
    r0s : ndarray, shape (N, 3). Initial positions.
    v0s : ndarray, shape (N, 3). Initial velocities.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.
    scaleh : float, optional. Scale factor for timestep adaptation (default: 0.005).

    Returns
    -------
    trajectories: tuple
        - t_vals : list of floats. Times at which the solution was evaluated.
        - rs_traj : ndarray, shape (M, N, 3). Trajectory of particle positions.
        - vs_traj : ndarray, shape (M, N, 3). Trajectory of particle velocities.
        - E_traj : ndarray, shape (M,). Total energy of the system over time.
        - am_traj : ndarray, shape (M, N, 3). Angular momentum of particles over time.
        - times : float. Cumulative CPU time spent integrating.
    """
    
    # reposition centre of mass to origin with no momentum 
    rcom, vcom = CentreOfMass(r0s, v0s, masses)
    r0s -= rcom
    v0s -= vcom
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    # Initialize our saved trajectories to be blank 
    t_vals = []
    rs_traj = [] 
    vs_traj = [] 
    E_traj = []
    am_traj = []
    times = 0 
    
    # run scheme for requried number of steps 
    while t <= T:
        t1 = time.time()
        rs, vs = scheme(rs, vs, h, G, masses)  # Update step
        h = scaleh * findh(rs, vs)
        t += h
        times += time.time() - t1
        E = TotalEnergy(rs, vs, G, masses)
        am = AngMomentum(rs, vs, masses) # Calculate angular momentum 
        
        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj = E_traj + [E]
        am_traj = am_traj + [am]
        
    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    E_traj = np.array(E_traj)
    am_traj = np.array(am_traj)
    
    # reposition centre of mass to origin with no momentum 
    rs_traj = np.array([rs + rcom for rs in rs_traj])
    vs_traj = np.array([vs + vcom for vs in vs_traj])
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)



def run_adaptive_symplectic(scheme, t0, T, h0, r0s, v0s, G, masses, scaleh = 0.005):
    """
    Integrate an n-body system with a symplectic integrator using adaptive timestep averaging.

    Parameters
    ----------
    scheme : callable. Symplectic numerical scheme function.
    t0 : float. Initial time.
    T : float. Final integration time.
    h0 : float. Initial timestep.
    r0s : ndarray, shape (N, 3). Initial positions.
    v0s : ndarray, shape (N, 3). Initial velocities.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.
    scaleh : float, optional. Scale factor for timestep adaptation (default: 0.005).

    Returns
    -------
    trajectories: tuple
        - t_vals : list of floats. Times at which the solution was evaluated.
        - rs_traj : ndarray, shape (M, N, 3). Trajectory of particle positions.
        - vs_traj : ndarray, shape (M, N, 3). Trajectory of particle velocities.
        - E_traj : ndarray, shape (M,). Total energy of the system over time.
        - am_traj : ndarray, shape (M, N, 3). Angular momentum of particles over time.
        - times : float. Cumulative CPU time spent integrating.
    """
    
    # reposition centre of mass to origin with no momentum 
    rcom, vcom = CentreOfMass(r0s, v0s, masses)
    r0s -= rcom
    v0s -= vcom
    
    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    # Initialize our saved trajectories to be blank 
    t_vals = []
    rs_traj = [] 
    vs_traj = [] 
    E_traj = []
    am_traj = []
    times = 0 
    
    rs, vs = scheme(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        t1 = time.time()

        h = scaleh * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = scheme(rs, vs, h, G, masses)
        h_bar = scaleh * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = scheme(rs, vs, h_new, G, masses)
        
        t += h_new

        times += time.time() - t1

        E = TotalEnergy(rs, vs, G, masses)
        am = AngMomentum(rs, vs, masses) # Calculate angular momentum 
        
        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj = E_traj + [E]
        am_traj = am_traj + [am]
        
    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    E_traj = np.array(E_traj)
    am_traj = np.array(am_traj)
    
    # reposition centre of mass to origin with no momentum 
    rs_traj = np.array([rs + rcom for rs in rs_traj])
    vs_traj = np.array([vs + vcom for vs in vs_traj])
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)