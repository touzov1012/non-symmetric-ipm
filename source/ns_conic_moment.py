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
from ns_util_linalg import *
from ns_structs import *

def Pmat(dData):
    """
    Encode coordinates, with respect to (2d + n choose n) interpolant basis,
    of some basis for (n, d) polynomials as columns in a tall matrix. This basis
    is declared implicitly through the point evaluations (coordinates) at the
    unisolvent points.
    
    L(x) = P'diag(g*x)P
    
    for some domain restriction g(x) >= 0 where L() is the canonical map to Sn
    
    L(q) = P'diag(g*q)P = gpp'
    
    where q is the basis of Lagrange polynomials on the interpolant points
    and so when passing x: xi acts as the desired evalutation of the interpolant basis
    for the ith interpolant point.
    
    For simplicity, this function returns a single P matrix, but if we have
    multiple domain restrictions gi(x) >= 0, we could speed up by returning multiple
    P matrices where column counts are based on the degree of gi.
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
    

def gHc(x, P, g):
    """
    Gradient, Hessian, Cone check at L(x).
    x = basis evaluations
    P = described as above used for mapping into PSD cone
    g = array of domain restrictions, polynomial functions including g0(x) = 1
        g is a collection of vectors representing gi evaluated at each unisol.
        point. Each vector is a row.
    """
    
    # fill 3D array along diagonal with g * x
    Gx = DiagExpand(g * x)
    
    L = P.T @ Gx @ P
    
    [evals, evecs] = np.linalg.eigh(L)
    
    check = (evals > 0).all()
    
    if check == False:
        return [None, None, check]
    
    Linv = evecs @ DiagExpand(1.0 / evals) @ evecs.transpose((0,2,1))
    
    G = P @ Linv @ P.T
    
    Gd = np.diagonal(G, axis1=-1, axis2=-2)
    
    grad = -np.sum(Gd * g, axis=0)
    hess = np.sum(g[:,:,None] * g[:,None,:] * G * G, axis=0)
    
    return [grad, hess, check]


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
    dgHc = lambda x: gHc(x, P, dData.g)
    
    e = np.ones(dData.A.shape[1])
    g1 = dgHc(e)[0]
    
    delta = InitDelta(dData, g1)
    
    x0 = delta * e
    s0 = -g1 / delta
    
    # barrier parameter
    nu = P.shape[1] * dData.g.shape[0]
    
    dInit = INIT(np.zeros(dData.A.shape[0]), x0, s0)
    dCone = CONE(dgHc, nu)
    
    return NSSolve(dData, dInit, dCone)
    

def TestProblem():
    """
    Minimize the butcher polynomial over the unit cube.
    Opt: -4.6667
    """
    
    F = lambda x: x[5]*x[1]**2 + x[4]*x[2]**2 - x[0]*x[3]**2 + x[3]**3 + x[3]**2 - 1/3*x[0] + 4/3*x[3]
    
    g0 = lambda x: 1
    g1 = lambda x: x[0] + 1
    g2 = lambda x: 1 - x[0]
    g3 = lambda x: x[1] + 1
    g4 = lambda x: 1 - x[1]
    g5 = lambda x: x[2] + 1
    g6 = lambda x: 1 - x[2]
    g7 = lambda x: x[3] + 1
    g8 = lambda x: 1 - x[3]
    g9 = lambda x: x[4] + 1
    g10 = lambda x: 1 - x[4]
    g11 = lambda x: x[5] + 1
    g12 = lambda x: 1 - x[5]
    
    n = 6
    d = 2
    d2 = d*2
    
    pts = UnisolventPoints(n, d2)
    
    g = FnVandermonde(pts, [F,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12])
    
    c = g[:,0]
    A = np.ones((1,len(c)))
    b = np.ones(1)
    
    dData = DATA(A, b, c)
    dData.n = n
    dData.d = d
    dData.d2 = d2
    dData.g = g[:,1:].T
    
    return dData
    
    