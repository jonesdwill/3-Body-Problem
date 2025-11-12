"""
Solvers and utility functions for n-body simulations using Forest-Ruth.

Contains:
- timestep and distance utilities (findh, maxDist, findR, distCalculator)
- a simple rasterization helper (fillGrid)
- several solver loops that rely on external project primitives:
  TotalEnergy, TotalKE, KineticEnergy, AngMomentum, Optimised_FR_Step, etc.
"""

import numpy as np
import pandas as pd
from functions import *
from schemes import *
from plot import *
from Kepler import *
from adaptive import *

def findh(rs: np.array, vs, h0 = np.inf):
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

def maxDist(rs):
    """
    Compute the maximum pairwise distance among points.

    Parameters
    ----------
    rs : array, shape (N, dim). Positions of N bodies.

    Returns
    -------
    d : float. Maximum Euclidean distance between any two bodies.
    """

    d = -1
    N = len(rs)
    for i in range(N): 
        for j in range(i+1, N):
            d = max(d, np.linalg.norm(rs[i]-rs[j]))
    return d

def findR(v0s, E0, masses):
    """ Estimate radius R from initial kinetic energy and total energy.

    Parameters
    ----------
    v0s : Initial velocities (passed to KineticEnergy).
    E0 : float. Reference total energy.
    masses : array. N-body masses used by KineticEnergy.

    Returns
    -------
    R : float. Computed characteristic radius. May be negative/inf if KE == E0.
    """

    ke = np.sum(KineticEnergy(v0s, masses))
    R = 5 / (2 * (ke - E0))
    return R

def distCalculator(rs):
    """
    Return a normalized list of pairwise distances (almost, excludes last element).

    Parameters
    ----------
    rs : array_like, shape (N, dim). Positions of N bodies.

    Returns
    -------
    normalized_dists : ndarray
    """

    N = len(rs)
    Rs = []
    for i in range(N):
        for j in range(i + 1, N):
            Rs.append(np.linalg.norm(rs[i] - rs[j]))
    Rs = np.array(Rs)
    return Rs[:-1] / np.sum(Rs)

def fillGrid(path, gridsize = 1000):
    """
    Rasterize a path of 2D points to a binary grid.

    Parameters
    ----------
    path : iterable of (x, y). Sequence of 2D positions to rasterize.
    gridsize : int, optional width and height of the square grid (default 1000).

    Returns
    -------
    grid : ndarray, shape (gridsize, gridsize). Binary grid with ones at visited cells.
    """

    grid = np.zeros((gridsize, gridsize))

    for pos in path:

        grid_x = int(2 * (pos[0] - 1e-15) * gridsize)
        grid_y = int(2 * (pos[1] - 1e-15) * gridsize)

        grid[grid_x][grid_y] = 1

    return grid

def fullSolver(T, C, r0s, v0s, G, masses, hlim = 1e-3, Elim = 0.01, h0 = 1e-10, t0 = 0):
    """
        Time-integrate an n-body system using the "Optimised_FR_Step" update and an adaptive timestep.

        Parameters
        ----------
        T : float. Final simulation time.
        C : float. Safety coefficient used when proposing timesteps (h = C * findh(...)).
        r0s, v0s : array. Initial positions and velocities (shape (N, dim)).
        G : float. Gravitational constant or parameter passed to energy/step functions.
        masses : array. Masses of bodies.
        hlim : float, optional. Minimum allowable timestep; solver stops with stability=2 if h_new < hlim.
        Elim : float, optional. Relative energy tolerance; solver stops with stability=3 if exceeded.
        h0 : float, optional. Initial timestep seed.
        t0 : float, optional. Initial time.

        Returns
        -------
        result : tuple (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)
            - t_vals : list of times
            - rs_traj : ndarray of position arrays saved at each step
            - vs_traj : ndarray of velocity arrays saved at each step
            - E_traj : list of relative energy errors at each saved step
            - am_traj : list of angular momentum diagnostics
            - times : float, cumulative CPU time spent inside the timed section
        stability : int
            Status code:
              0 - system unbounded (max distance exceeds threshold distance of 10)
              1 - reached final time (t+h_new > T)
              2 - timestep fell below hlim
              3 - energy error exceeded Elim
        """

    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    t_vals = [t0]
    rs_traj = [r0s] 
    vs_traj = [v0s] 
    E_traj = [0]
    am_traj = [AngMomentum(rs, vs, masses)]
    times = 0 

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        t1 = time.time()

        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        times += time.time() - t1

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs((E - E0) / E0hat)

        if h_new < hlim: 
            stability = 2
            break 
        if t+h_new > T: 
            stability =  1
            break 
        if relE > Elim:
            stability = 3
            break 
        if maxDist(rs) > 10:
            stability = 0
            break 
        
        t += h_new

        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj.append(relE)
        am_traj.append(AngMomentum(rs, vs, masses))

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times), stability 


def optimisedSolver(T, C, r0s, v0s, G, masses, hlim = 1e-3, Elim = 0.01, h0 = 1e-10, t0 = 0):
    """
    Streamlined solver that only stores positions and tracks maximum relative energy.

    Compared to fullSolver this function returns stability, the maximum
    observed relative energy error and the final time t reached.

    Parameters
    ----------
    T, C, r0s, v0s, G, masses : see fullSolver
    hlim, Elim, h0, t0 : see fullSolver

    Returns
    -------
    stability : int. Status code (same semantics as fullSolver).
    maxE : float. Maximum observed relative energy error over the simulation.
    t : float. Final simulation time reached.
    """

    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    maxE = 0

    rs_traj = [r0s] 

    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs(np.sum(E) - E0) / np.abs(E0hat)

        if h_new < hlim: 
            stability = 2
            break 
        if t+h_new > T: 
            stability =  1
            break 
        if relE > Elim:
            stability = 3
            break 
        if maxDist(rs) > 10:
            stability = 0
            break 
        
        t += h_new

        rs_traj = rs_traj + [rs] 
        maxE = max(relE, maxE)

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    
    return stability,  maxE, t

def shapeSolver(T, C, r0s, v0s, G, masses, defaultVar = 10000, hlim = 1e-3, Elim = 0.01, h0 = 1e-10, t0 = 0):
    """
    Solver variant that computes a crude 'shape variance' by rasterizing the path.

    When the solver reaches T it fills a grid using `distCalculator` applied to each
    saved configuration and returns the fraction of grid cells occupied as `variance`.
    Otherwise behaves like `optimisedSolver` with the same stability codes.

    Parameters
    ----------
    defaultVar : int, optional. Default variance returned if the solver stops before rasterizing (default 10000).
    Other parameters : same as optimisedSolver/fullSolver.

    Returns
    -------
    stability : int. Status code (same semantics as fullSolver).
    variance : float. Sum of filled grid cells (a crude measure of area/coverage) or defaultVar.
    maxE : float. Maximum observed relative energy error.
    t : float. Final simulation time reached.
    """

    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    variance = defaultVar
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    maxE = 0

    rs_traj = [r0s] 

    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for requried number of steps 
    while t <= T:
        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs(np.sum(E) - E0) / np.abs(E0hat)

        if h_new < hlim: 
            stability = 2
            break 
        if t+h_new > T: 
            stability =  1
            path = [distCalculator(rs) for rs in rs_traj]
            grid = fillGrid(path, gridsize = 1000)
            variance = np.sum(grid)
            break 
        if relE > Elim:
            stability = 3
            break 
        if maxDist(rs) > 10:
            stability = 0
            break 
        
        t += h_new

        rs_traj = rs_traj + [rs] 
        maxE = max(relE, maxE)

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    
    return stability, variance, maxE, t



def NStepsSolver(NSteps, C, r0s, v0s, G, masses, hlim = 1e-6, Elim = 0.01, h0 = 1e-10, t0 = 0):
    """
    Run the integration for a fixed number of steps (useful for profiling or tests).

    This variant ignores energy and distance stopping criteria and only stops once
    `totalSteps > NSteps`. It returns the same trajectory data shape as fullSolver
    together with a stability code (1 when the requested number of steps is reached).

    Parameters
    ----------
    NSteps : int. Number of integration steps to perform (stops when totalSteps > NSteps).
    C, r0s, v0s, G, masses, hlim, Elim, h0, t0 : see fullSolver

    Returns
    -------
    result : tuple (t_vals, rs_traj, vs_traj, E_traj, am_traj, times)
    stability : int. Status code (1 when the requested number of steps is reached).
    """

    # Make a copy of initial values
    rs = np.copy(r0s)
    vs = np.copy(v0s)
    t = t0
    h = h0
    
    E0 = TotalEnergy(r0s, v0s, G, masses)
    E0hat = E0 
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(TotalKE(v0s, masses) + np.abs(r0s, G, masses))) 
    if E0hat == 0: E0hat = 1

    t_vals = [t0]
    rs_traj = [r0s] 
    vs_traj = [v0s] 
    E_traj = [0]
    am_traj = [AngMomentum(rs, vs, masses)]
    times = 0 

    totalSteps = 0

    rs, vs = Optimised_FR_Step(rs, vs, h, G, masses) 
    
    # run scheme for required number of steps
    while 1 < 2:
        t1 = time.time()

        h = C * findh(rs, vs) # proposed timestep

        rs_bar, vs_bar = Optimised_FR_Step(rs, vs, h, G, masses)
        h_bar = C * findh(rs_bar, vs_bar) # proposed timestep

        h_new = (h + h_bar) / 2
        
        rs, vs = Optimised_FR_Step(rs, vs, h_new, G, masses)

        times += time.time() - t1

        E = TotalEnergy(rs, vs, G, masses) # Calculate Energy 
        relE = np.abs((E - E0) / E0hat)

        # if h_new < hlim: 
        #     stability = 2
        #     break 
        if totalSteps > NSteps: 
            stability =  1
            break 
        # if relE > Elim:
        #     stability = 3
        #     break 
        # if maxDist(rs) > 10:
        #     stability = 0
        #     break 
        
        t += h_new

        ## append values to trajectories 
        t_vals = t_vals + [t]
        rs_traj = rs_traj + [rs] 
        vs_traj = vs_traj + [vs]
        E_traj.append(relE)
        am_traj.append(AngMomentum(rs, vs, masses))

        totalSteps += 1

    # Make trajectories into numpy arrays
    rs_traj = np.array(rs_traj)
    vs_traj = np.array(vs_traj) 
    
    return (t_vals, rs_traj, vs_traj, E_traj, am_traj, times), stability 

