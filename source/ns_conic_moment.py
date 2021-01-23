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
from ns_structs import *

def Pmat(dData):
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
    
    pts = UnisolventPoints(dData.n, dData.d2)
    V = ChebyVandermonde(pts, dData.d)
    [P, R] = np.linalg.qr(V)
    
    return P

def InitDelta(dData, g1):
    """
    Find the initialization parameter for building the starting point (x, s)
    where x = delta * 1 and s = -delta^-1 g(1)
    """
    
    a = np.sum(dData.A, axis=1)
    dP = np.amax((1 + np.abs(dData.b)) / (1 + np.abs(a)))
    dD = np.amax((1 + np.abs(g1)) / (1 + np.abs(dData.c)))
    
    return np.sqrt(dP * dD)
    

def gHc(x, P):
    """
    Gradient, Hessian, Cone check at L(x)
    """
    
    L = P.T @ np.diag(x) @ P
    
    [evals, evecs] = np.linalg.eigh(L)
    
    check = (evals > 0).all()
    
    Linv = evecs @ np.diag(1.0 / evals) @ evecs.T
    
    G = P @ Linv @ P.T
    
    return [-np.diag(G), G * G, check]


def Solve(dData):
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
    
    P = Pmat(dData)
    dgHc = lambda x: gHc(x, P)
    
    e = np.ones(dData.A.shape[1])
    g1 = dgHc(e)[0]
    
    delta = InitDelta(dData, g1)
    
    x0 = delta * e
    s0 = -g1 / delta
    
    nu = P.shape[1]
    
    dInit = INIT(np.zeros(dData.A.shape[0]), x0, s0)
    dCone = CONE(dgHc, nu)
    
    return NSSolve(dData, dInit, dCone)
    

def TestProblem():
    """
    Minimize the famous 6 hump camel back function.
    Opt: -1.0316
    """
    
    F = lambda x: (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4+4*x[1]**2)*x[1]**2
    
    n = 2
    d = 3
    d2 = 6
    
    pts = UnisolventPoints(n, d2)
    
    c = np.array([F(pts[i,:]) for i in range(0, pts.shape[0])])
    A = np.ones((1,len(c)))
    b = np.ones(1)
    
    dData = DATA(A, b, c)
    dData.n = n
    dData.d = d
    dData.d2 = d2
    
    return dData
    
    