# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:23:48 2021

@author: alex
"""

import time
import numpy as np
from ns_structs import *

def sqrtmatsym(A):
    """
    matrix square root
    """
    
    [evals, evecs] = np.linalg.eigh(A)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T

def sqrtmatsyminv(A):
    """
    matrix square root
    """
    
    [evals, evecs] = np.linalg.eigh(A)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

def RelativeGap(dSoln, dCone):
    """
    The complementary gap for a solution (x, s)
    """
    
    return (np.dot(dSoln.x, dSoln.s) + dSoln.xtau * dSoln.stau) / (dCone.nu + 1)

def ProxInfo(dSoln, dCone):
    """
    Calculate all proximity info.
    inside: is x inside the cone
    H: hessian at x
    g: gradient at x
    phi: descent direction
    pdist: distance to central path
    """
    
    [g0, H0, inside] = dCone.gHc(dSoln.x)
    
    if dSoln.xtau <= 0 or dSoln.stau <= 0:
        inside = False
    
    if inside == False:
        return [False, np.nan, np.nan, np.nan, np.inf]
    
    g = np.append(g0, -1.0 / dSoln.xtau)
    
    H = np.zeros((H0.shape[0]+1,H0.shape[1]+1))
    H[:-1,:-1] = H0
    H[-1:,-1:] = 1.0 / (dSoln.xtau * dSoln.xtau)
    
    gap = RelativeGap(dSoln, dCone)
    
    phi = np.append(dSoln.s, dSoln.stau) + gap * g
    
    pdist = np.linalg.norm(sqrtmatsyminv(H) @ phi)
    
    return [inside, H, g, phi, pdist]

def SplitToSoln(dSoln, yxs):
    """
    Split a vector among the components of dSoln
    """
    
    m = len(dSoln.y)
    n = len(dSoln.x)
    
    dSoln.y = yxs[0:m]
    dSoln.x = yxs[m:(m+n)]
    dSoln.xtau = yxs[m+n]
    dSoln.s = yxs[(m+n+1):-1]
    dSoln.stau = yxs[-1]
    

def LineSearch(dSoln, dDelta, dCone, radius, limit):
    """
    Line search for the largest dDelta such that dSoln in the proxi ball
    """
    
    dSoln.MAD(dDelta, limit)
    limit = -limit
    
    while ProxInfo(dSoln, dCone)[4] >= radius * RelativeGap(dSoln, dCone):
        limit /= 2
        dSoln.MAD(dDelta, limit)
    

def SelfDualNewtonSystem(data):
    """
    Create the self dual embedding with a known interior point for the problem
    
    min cx
        Ax = b
        x >= 0
        
    The embedding will be of the form
    
    min -c*x
        A*x* = 0
    -A*'y + Cx* - S = c*
    x,S >= 0
    
    with known solution x=S=1, y=0
    
    The returned quantity is the right hand side of the linear system
    solved during a newton step, that is the optimality conditions
    xs = mu are appended in linearized form
    """
    
    n = data.A.shape[1]
    m = data.A.shape[0]
    
    A_star = np.c_[data.A,-data.b]
    C = np.zeros((n+1,n+1))
    C[0:n,n] = data.c
    C[n,0:n] = -C[0:n,n].T
    
    yA = np.r_[np.zeros((m,m)), -A_star.T, np.zeros((n+1, m))]
    xA = np.r_[A_star, C, np.eye(n+1)]
    sA = np.r_[np.zeros((m, n+1)), -np.eye(n+1), np.eye(n+1)]
    
    return np.c_[yA, xA, sA]

def UpdateStats(dStats, dData, dSoln, dCone, dInit):
    """
    Update current solution stats
    """
    
    [inside, H, g, phi, proxi] = ProxInfo(dSoln, dCone)
    
    dStats.inside = inside
    dStats.H = H
    dStats.g = g
    dStats.phi = phi
    dStats.proxi = proxi
    dStats.gap = RelativeGap(dSoln, dCone)
    dStats.elapsed = time.time() - dStats.stime
    dStats.nsteps += 1
    dStats.pres = np.linalg.norm(dData.A @ dSoln.x - dData.b * dSoln.xtau, np.inf)
    dStats.dres = np.linalg.norm(dData.A.T @ dSoln.y - dData.c * dSoln.xtau + dSoln.s, np.inf)
    dStats.gres = np.abs(np.dot(dData.b, dSoln.y) - np.dot(dData.c, dSoln.x) - dSoln.stau)
    dStats.pval = np.dot(dData.c, dSoln.x) / max(dSoln.xtau, 0.0000001)
    dStats.dval = np.dot(dData.b, dSoln.y) / max(dSoln.xtau, 0.0000001)
    
    selfFeas = dStats.pres < dInit.eps and dStats.dres < dInit.eps and dStats.gres < dInit.eps
    
    if selfFeas:
        if np.abs(dStats.pval - dStats.dval) < dInit.eps:
            dStats.status = 1 # optimal PD
        elif dSoln.xtau < dInit.eps * 0.01 and dSoln.stau > dInit.eps * 0.01:
            dStats.status = 2 # infeasible P/D
        elif dSoln.xtau < dInit.eps * 0.1 and dSoln.stau < dInit.eps * 0.1:
            dStats.status = 3 # ill-posed, pathological
        else:
            dStats.status = 0 # still working
    else:
        dStats.status = 0 # working
        

def NSSolve(dData, dInit, dCone):
    """
    Solve the non-symmetric conic problem with the Skajaa Ye PCA
    dData: problem data
    dInit: initialization data
    dCone: cone data
    """
    
    m = dData.A.shape[0]
    n = dData.A.shape[1]
    nplus = n + 1
    
    nA = SelfDualNewtonSystem(dData)
    nb = np.zeros(nA.shape[0])
    
    dSoln = SOLN(dInit.y0, dInit.x0, dInit.s0, 1.0, 1.0)
    
    dStats = STATS(False, np.nan, np.nan, np.nan, 0.0, 0.0, time.time(), 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    dDelta = SOLN(np.zeros(m), np.zeros(n), np.zeros(n), 0.0, 0.0)
    UpdateStats(dStats, dData, dSoln, dCone, dInit)
    
    print(dStats)
    
    iterates = 0
    maxItr = 1000
    
    while dStats.status == 0 and iterates < maxItr:
        # predictor phase
        nb[:-nplus] = -nA[:-nplus,:] @ np.r_[dSoln.y, dSoln.x, dSoln.xtau, dSoln.s, dSoln.stau]
        nb[-nplus:] = -np.r_[dSoln.s, dSoln.stau]
        
        nA[(m+nplus):,m:(m+nplus)] = dStats.gap * dStats.H
        
        SplitToSoln(dDelta, np.linalg.solve(nA, nb))
        LineSearch(dSoln, dDelta, dCone, dInit.beta, 4)
        
        # corrector phase
        for j in range(0,dInit.correctors):
            UpdateStats(dStats, dData, dSoln, dCone, dInit)
            
            nb[:] = 0
            nb[-nplus:] = -dStats.phi
            
            nA[(m+nplus):,m:(m+nplus)] = dStats.gap * dStats.H
            
            SplitToSoln(dDelta, np.linalg.solve(nA, nb))
            LineSearch(dSoln, dDelta, dCone, dInit.eta, 4)
        
        UpdateStats(dStats, dData, dSoln, dCone, dInit)
        
        print(dStats)
    
    if dStats.status == 0:
        dStats.status = 3
        print(dStats)
    
    return [dSoln, dStats]
