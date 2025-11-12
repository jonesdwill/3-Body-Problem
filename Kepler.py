"""
Kepler Problem

This module provides functions for solving Kepler's equation and computing
orbital positions for a two-body problem.
"""

import numpy as np

def SolveKepler(e, M):
    """
    Solve Kepler's equation E - e*sin(E) = M for the eccentric anomaly E using Newton-Raphson iteration.

    Parameters
    ----------
    e : float. Orbital eccentricity (0 <= e < 1).
    M : float. Mean anomaly (in radians).

    Returns
    -------
    E : float. Eccentric anomaly (in radians).
    """

    E0 = M
    while True:
        E1 = E0 - (E0 - e * np.sin(E0) - M) / (1 - e * np.cos(E0))
        if np.linalg.norm(E1 - E0) < 1e-8:
            return E1
        E0 = E1

def KeplerOrbit(a, e, mu, t):
    """
    Compute the position in 3D of a body in a Keplerian orbit at time t.

    Parameters
    ----------
    a : float. Semi-major axis of the orbit.
    e : float. Eccentricity of the orbit.
    mu : float. Standard gravitational parameter (G*(M+m)).
    t : float or ndarray. Time(s) at which to evaluate the orbit.

    Returns
    -------
    pos : ndarray, shape (len(t), 3). Cartesian coordinates (x, y, z) of the body at time t.
    """

    T = KeplerPeriod(a, mu)
    M = 2 * np.pi * t / T
    E = SolveKepler(e, M)
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))

def SemiMajorAxis(mu, r0s, v0s):
    """
    Estimate the semi-major axis of an orbit from initial position and velocity.

    Parameters
    ----------
    mu : float. Standard gravitational parameter (G*(M+m)).
    r0s : ndarray, shape (N, 3). Initial positions of bodies; the second body is used for calculation.
    v0s : ndarray, shape (N, 3). Initial velocities of bodies; the second body is used for calculation.

    Returns
    -------
    a : float. Semi-major axis of the orbit.
    """

    r = r0s[1, :]
    v = v0s[1, :]
    R = np.linalg.norm(r)
    a = mu * R / (2 * mu - R * (v[0] ** 2 + v[1] ** 2))
    return a

def KeplerPeriod(a, mu): return 2 * np.pi * np.sqrt(a**3 / mu)