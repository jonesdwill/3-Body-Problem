# Stability of Selective Solutions to the 3-Body Problem 
 The general three body problem, which describes three masses interacting freely and purely
 by gravitational attraction, has been a focus of research in classical mechanics for the past 300
 years. Previous analytical work by Euler and Lagrange describes five families of periodic, elliptical, 
 solutions to the problem. Furthermore, modern numerical techniques have unlocked the discovery of many 
 new periodic solutions, such as the distinct ‘figure-8’ orbit. This thesis builds
 a framework for analysing the stability of such orbits, in both the traditional and Lyapunov
 sense. It is confirmed that the figure-8 is stable, and Euler and Lagrange’s circular solutions are
 unstable. A region of stability is found for the figure-8, and the maximum Lyapunov exponents
 of Euler and Lagrange’s solutions are accurately calculated. The implications of the results are
 discussed with reference to Celestial mechanics and the N-body problem

# Framework
This repo includes a framework for solving the 3-body problem up to time T, using a selection of candidate schemes. Most usefully, it includes a time-adaptive Symplective integrator based on the Forest-ruth algorithm, to solve a generalised N-body system.

**functions.py** Governing equations, as well as primitives for Energy and Angular Momentum.

**plot.py** Utilities for plotting N-bodies. 

**schemes.py** Contains selection of candidate numerical integrators.

**Kepler.py** Non-numerical solution to Kepler orbits (N=2). Useful for testing.

**FRSolver.py** Solves the N-Body Equations using Forest-Ruth.

**adaptive.py** Solves the N-Body Equations using a time-adaptive Forest-Ruth. 

Please Note: This project is designed for my Master's thesis, and not as a well-interpretable package. As a result, many of the notebooks and Scripts have not been optimised, have redundant functionality, and serve to document progression. 

# Report
A write-up and citations are found in the pdf '[Stability of Selective Solutions to the 3-Body Problem.pdf](Stability%20of%20Selective%20Solutions%20to%20the%203-Body%20Problem.pdf)'.
