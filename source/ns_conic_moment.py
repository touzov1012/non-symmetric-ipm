# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 08:59:28 2021

Optimization over the moment cone (dual of the SOS cone) where points

    x = E[fi(x)]
    
where fi form a basis for space of (n, 2d) polynomials. In this application,
the basis is the Lagrange basis on a unisolvent set of size (2d + n choose n).

Coordinates of any function f with respect to fi is simply the evaluation of
f(xi) where xi is the ith interpolation point.

@author: alex
"""

import numpy as np
from ns_conic import *
from ns_util_poly import *

def Pmat(n, d):
    """
    Encode coordinates, with respect to (2d + n choose n) interpolant basis,
    of some basis for (n, d) polynomials as columns in a tall matrix. This basis
    is declared implicitly through the point evaluations (coordinates) at the
    unisolvent points.
    
    L(x) = P'diag(x)P
    
    so that
    
    L(q) = P'diag(q)P = pp'
    
    where q is the basis of Lagrange polynomials on the interpolant points
    and so when passing x: xi is the desired evalutation of the interpolant basis
    for the ith interpolant point.
    """
    
    pts = UnisolventPoints(n, 2*d)
    V = ChebyVandermonde(pts, d)
    [P, R] = np.linalg.qr(V)
    
    return P

def InitDelta(A, b, c, g1):
    """
    Find the initialization parameter for building the starting point (x, s)
    where x = delta * 1 and s = -delta^-1 g(1)
    """
    
    a = np.sum(A, axis=1)
    dP = np.amax((1 + np.abs(b)) / (1 + np.abs(a)))
    dD = np.amax((1 + np.abs(g1)) / (1 + np.abs(c)))
    
    return np.sqrt(dP * dD)
    

def g(x, P):
    """
    Gradient of the barrier at L(x)
    """
    
    return -np.diag(P @ np.linalg.inv(P.T @ np.diag(x) @ P) @ P.T)
    

def H(x, P):
    """
    Hessian of the barrier at L(x)
    """
    
    L = P.T @ np.diag(x) @ P
    
    [evals, evecs] = np.linalg.eigh(L)
    
    if (evals <= 0).any():
        return np.nan
    
    Linv = evecs @ np.diag(1.0 / evals) @ evecs.T
    
    G = P @ Linv @ P.T
    return G * G


def Solve(A, b, c, n, d):
    """
    Solve the moment program using non-symmetric cone alg where n is the number
    of polynomial variats and d is the largest degree of the polynomials.
    
    The rows of A should be coordinates with respect to the interpolant basis
    on the points described above. That is, if I want to model f(x) = b, then the
    coordinates of f(x) with respect to the interpolant basis are f(xi) for the
    ith coordinate.
    
    Thus, the conic constraint is there to enforce that any sequence of moments
    x is a valid one, that is a convex combination of interpolant basis
    point evaluations.
    """
    
    P = Pmat(n, d)
    dH = lambda x: H(x, P)
    dg = lambda x: g(x, P)
    
    e = np.ones(A.shape[1])
    g1 = dg(e)
    
    delta = InitDelta(A, b, c, g1)
    
    x0 = delta * e
    s0 = -g1 / delta
    
    nu = Choose(n + d, d)
    
    return NSSolve(A, b, c, dH, dg, nu, x0 = x0, s0 = s0)
    

def TestProblem():
    """
    Minimize the famous 6 hump camel back function.
    Opt: -1.0316
    """
    
    F = lambda x: (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4+4*x[1]**2)*x[1]**2
    
    n = 2
    d2 = 6
    d = 3
    
    pts = UnisolventPoints(n, d2)
    
    c = np.array([F(pts[i,:]) for i in range(0, pts.shape[0])])
    A = np.ones((1,len(c)))
    b = np.ones(1)
    
    return [A, b, c, n, d]
    
    