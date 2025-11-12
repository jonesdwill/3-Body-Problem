"""
N-Body Problem Governing Equations

This module provides functions to compute forces, energies, angular momentum,
and to manage positions and velocities for a system of N gravitating bodies.
"""

import numpy as np

# ============================
#      Governing Equations
# ============================

def Force_i(rs, i, G, masses):
    """
    Compute total gravitational force on particle i due to all other particles.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of all particles.
    i : int. Index of the particle to compute the force on.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of all particles.

    Returns
    -------
    F_i : ndarray, shape (3,). Total force acting on particle i.
    """
    
    ri = rs[i] # get position of i-th mass 
    F_i = np.zeros(ri.shape) # create empty force vector for i-th mass 
    
    # loop over other masses
    for j, rj in enumerate(rs):
        
        if i != j:
            assert np.linalg.norm(rj - ri), 'Collision' # masses are at the same position 
            Fij = G * masses[i] * masses[j] * (rj - ri) / ((np.linalg.norm(rj - ri))** 3) # force of j-th mass on i-th mass
            F_i += Fij # sum forces
            
    return F_i

def Force(rs, G, masses):
    """
    Compute gravitational forces on particles.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of all particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of all particles.

    Returns
    -------
    Fs : ndarray, shape (N, 3). Forces on each particle.
    """
    
    N = len(rs)

    Fs = np.zeros_like(rs).astype('float64') # empty vector of forces 

    for i in range(N):
        for j in range(i+1, N):
            rij = rs[j] - rs[i]
            rij_mag = np.linalg.norm(rij)
            F = masses[i] * masses[j] * (rij) / (rij_mag ** 3)
            Fs[i] += F
            Fs[j] += - F
        
    return G * Fs

def dr_dt(vs):
    """ Compute derivative of positions (velocity) for all particles. """
    return vs

def dv_dt(rs, G, masses):
    """
    Compute acceleration of each particle due to gravitational forces.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of all particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of all particles.

    Returns
    -------
    dvdt : ndarray, shape (N, 3). Accelerations of each particle.
    """

    N = len(rs)

    Fs = np.zeros_like(rs).astype('float64') # empty vector of forces 

    for i in range(N):
        for j in range(i+1, N):
            rij = rs[j] - rs[i]
            F = (rij) / (np.linalg.norm(rij) ** 3)
            Fs[i] += F * masses[j]
            Fs[j] += - F * masses[i]
        
    return G * Fs

def w_to_vec(w):
    """ Convert a flat 1D vector to position and velocity arrays. """
    W = np.reshape(w, (len(w) // 3, 3))
    rs, vs = np.split(W, 2) 
    return rs, vs

def vec_to_w(rs, vs):
    """ Flatten positions and velocities into a single 1D vector. """
    return np.concatenate((rs.flatten(), vs.flatten()))
    
def all_derivatives(w, t, G, masses):
    """
    Compute time derivatives of positions and velocities for all particles.

    Parameters
    ----------
    w : ndarray, shape (2*N*3,). Flattened positions and velocities.
    t : float. Current time (unused in Newtonian gravity).
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of all particles.

    Returns
    -------
    derivs : ndarray, shape (2*N*3,). Flattened derivatives of positions and velocities.
    """
 
    rs, vs = w_to_vec(w) # separate positions and velocities
    
    drdt = dr_dt(vs) # find velocity of all particles
    dvdt = dv_dt(rs, G, masses) # find acceleration of all particles
    
    derivs = vec_to_w(drdt, dvdt)
    # derivs = np.concatenate((drdt,dvdt)).flatten() # reformat to be used by scipy integrate
    
    return derivs

# ============================
#         Re-position
# ============================

def CentreOfMass(rs, vs, masses):
    """
    Compute center of mass position and velocity of the system.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    vs : ndarray, shape (N, 3). Velocities of particles.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    rcom : ndarray, shape (3,). Position of center of mass.
    vcom : ndarray, shape (3,). Velocity of center of mass.
    """
    
    rcom = sum([rs[i] * masses[i] for i in range(len(masses))]) / np.sum(masses)
    vcom = sum([vs[i] * masses[i] for i in range(len(masses))]) / np.sum(masses)
    return rcom, vcom

def Centralise(rs_traj, i):
    """
    Reposition a given particle at the origin and shift all other particles accordingly.

    Parameters
    ----------
    rs_traj : ndarray, shape (T, N, 3). Trajectory of all particle positions.
    i : int. Index of the particle to centralise.

    Returns
    -------
    rs_traj : ndarray, shape (T, N, 3). Adjusted trajectory with particle i at the origin.
    """
    
    ri = np.copy(rs_traj[:,i,:])
    
    for j in range(rs_traj.shape[1]):
        rs_traj[:,j,:] -= ri
        
    return rs_traj

# ============================
#          Energy 
# ============================

def KE(vs, i, masses):
    """
    Compute kinetic energy of a single particle.

    Parameters
    ----------
    vs : ndarray, shape (N, 3). Velocities of particles.
    i : int. Index of particle.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    ke : float. Kinetic energy of particle i.
    """
        
    ke = 0.5 * masses[i] * np.linalg.norm(vs[i]) ** 2
    return ke

def KineticEnergy(vs, masses):
    """
    Compute kinetic energy of each particle.

    Parameters
    ----------
    vs : ndarray, shape (N, 3). Velocities of particles.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    kes : ndarray, shape (N,). Kinetic energy of each particle.
    """
        
    ke = 0.5 * masses.T @ np.array([np.linalg.norm(v) ** 2  for v in vs])
    return ke

def TotalKE(vs, masses):
    """
    Compute total kinetic energy of the system.

    Parameters
    ----------
    vs : ndarray, shape (N, 3). Velocities of particles.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    KE_total : float. Total kinetic energy of the system.
    """

    ke = 0.5 * masses.T @ np.array([np.linalg.norm(v) ** 2  for v in vs])
    return ke

def PE(rs, i, G, masses):
    """
    Compute potential energy of a single particle.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    i : int. Index of particle.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    pe : float. Potential energy of particle i.
    """

    ri = rs[i]
    U = 0
    for j, rj in enumerate(rs):
        if i != j:
            # print(ri, rj, np.linalg.norm(rj - ri))
            Uij = masses[i] * masses[j] / np.linalg.norm(rj - ri)
            U += Uij 

    return - G * U

def PotentialEnergy(rs, G, masses):
    """
    Compute potential energy of each particle.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of all particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    pes : ndarray, shape (N,). Potential energy of each particle.
    """

    N = len(rs)

    pes = [PE(rs, i, G, masses) for i in range(N)]
        
    return pes / 2

def TotalPE(rs, G, masses):
    """
    Compute total potential energy of the system.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    U_total : float. Total potential energy.
    """

    N = len(rs)

    U = 0

    for i in range(N):
        for j in range(i+1, N):
            U += masses[i] * masses[j] / np.linalg.norm(rs[j] - rs[i])
        
    return - G * U

def RelativeEnergy(E_traj):
    """
    Compute relative change in total energy over time as a percentage.

    Parameters
    ----------
    E_traj : ndarray, shape (T, N). Trajectory of particle energies over time.

    Returns
    -------
    dE : ndarray, shape (T,). Relative energy error (%) compared to initial total energy.
    """

    Et = np.sum(E_traj, axis = 1)
    E0 = Et[0] # initial energy 
    E0hat = E0 # scaling coefficient 
    
    ## conditions to avoid dividing by zero 
    if E0hat == 0: E0hat = np.max(np.abs(E_traj[0])) 
    if E0hat == 0: E0hat = 1
    
    dE = np.abs(Et - E0) / np.abs(E0hat)
    
    return dE * 100

def Energies(rs, vs, G, masses):
    """
    Compute total energy (kinetic + potential) of the system.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    vs : ndarray, shape (N, 3). Velocities of particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    E_total : ndarray, shape (N,). Total energy per particle.
    """

    ke = KineticEnergy(vs, masses)
    pe = PotentialEnergy(rs, G, masses)
    return ke + pe

def TotalEnergy(rs, vs, G, masses):
    """
    Compute total energy of the system.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    vs : ndarray, shape (N, 3). Velocities of particles.
    G : float. Gravitational constant.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    E_total : float. Total energy of the system.
    """

    return TotalKE(vs, masses) + TotalPE(rs, G, masses)

# ============================
#      Angular Momentum
# ============================

def AM(rs, vs, i, masses):
    """
    Compute angular momentum of a single particle.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    vs : ndarray, shape (N, 3). Velocities of particles.
    i : int. Index of particle.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    l : ndarray, shape (3,). Angular momentum vector of particle i.
    """
    l = np.cross(rs[i], masses[i]*vs[i])
    return l

def AngMomentum(rs, vs, masses):
    """
    Compute angular momentum of all particles.

    Parameters
    ----------
    rs : ndarray, shape (N, 3). Positions of particles.
    vs : ndarray, shape (N, 3). Velocities of particles.
    masses : ndarray, shape (N,). Masses of particles.

    Returns
    -------
    L : ndarray, shape (N, 3). Angular momentum vectors of all particles.
    """
        
    L = [AM(rs, vs, i, masses) for i in range(len(masses))]
    return np.array(L)

def RelativeAngMomentum(am_traj):
    """
    Compute relative change in total angular momentum over time (%).

    Parameters
    ----------
    am_traj : ndarray, shape (T, N, 3). Trajectory of angular momenta for each particle.

    Returns
    -------
    dL : ndarray, shape (T,). Relative angular momentum change (%) compared to initial value.
    """
        
    Lt = np.sum(am_traj, axis = 1) # total angular momentum of the system
    L0 = Lt[0] # initial angular momentum
    L0hat = L0 # scaling coefficient 
    
    ## conditions to avoid dividing by zero 
    if L0hat == 0: L0hat = np.max(np.abs(am_traj))
    if L0hat == 0: L0hat = 1
        
    # scale schange in angular momentum
    dL = np.abs(Lt - L0) / np.abs(L0hat)
    return dL * 100
