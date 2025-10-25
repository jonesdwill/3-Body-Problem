# Stability of Selective Solutions to the 3-Body Problem 
 The general three body problem, which describes three masses interacting freely and purely
 by gravitational attraction, has been a focus of research in classical mechanics for the past 300
 years. Previous analytical work by Euler and Lagrange describes five families of periodic, ellip
tical, solutions to the problem. Furthermore, modern numerical techniques have unlocked the
 discovery of many new periodic solutions, such as the distinct ‘figure-8’ orbit. This thesis builds
 a framework for analysing the stability of such orbits, in both the traditional and Lyapunov
 sense. It is confirmed that the figure-8 is stable, and Euler and Lagrange’s circular solutions are
 unstable. A region of stability is found for the figure-8, and the maximum Lyapunov exponents
 of Euler and Lagrange’s solutions are accurately calculated. The implications of the results are
 discussed with reference to Celestial mechanics and the N-body problem

# Framework
This repo includes various order functions for solving the 3-body problem. Most usefully, it includes a time-adaptive Symplective integrator based on the Forest-ruth algorithm, to solve a generalised N-body system.

**FRSolver.py** Solves the N-Body Equations using Forest-Ruth.
**adaptive.py** Solves the N-Body Equations using a time-adapive Forest-Ruth (changing step size). 

# Write-up
Write up and citations are found in the pdf file. 
