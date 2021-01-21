# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:23:48 2021

@author: alex
"""

import numpy as np

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

def DualityGap(x, s):
    """
    The complementary gap for a solution (x, s)
    """
    
    return np.dot(x, s) / len(x)

def Phi(x, s, g):
    """
    Descent direction for barrier in corrector phase
    g: gradient of the barrier
    """
    
    return s + DualityGap(x, s) * g(x)

def Prox(x, s, H, g):
    """
    Calculate the proximity to the central path
    H: Hessian of barrier
    g: gradient of barrier
    """
    h = H(x)
    
    if np.isnan(h).any():
        return float('inf')
    
    phi = Phi(x, s, g)
    
    return np.linalg.norm(sqrtmatsyminv(h) @ phi)

def extg(g, x):
    """
    Evaluate extended gradient for self dual embedding
    """
    
    n = len(x)
    hg = np.zeros(n)
    hg[:-1] = g(x[:-1])
    hg[-1:] = -1.0 / x[-1:]
    
    return hg

def extH(H, x):
    """
    Evaluate extended Hessian for self dual embedding
    """
    
    n = len(x)
    hH = np.zeros((n,n))
    hH[:-1,:-1] = H(x[:-1])
    hH[-1:,-1:] = 1.0 / (x[-1:] * x[-1:])
    
    return hH

def LineSearch(F, beta, x, s, dx, ds, L, U, iterates):
    """
    Line search minimum value of alpha for F(x0 + d * alpha)
    """
    
    nx = x + dx * U
    ns = s + ds * U
    gap = DualityGap(nx, ns)
    
    U_val = F(nx, ns)
    
    if iterates <= 0:
        if U_val < beta * gap:
            return U
        else:
            return L
        
    
    if U_val >= beta * gap:
        return LineSearch(F, beta, x, s, dx, ds, L, (U + L) / 2, iterates - 1)
    else:
    	return U

def SelfDualNewtonSystem(A, b, c):
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
    
    n = A.shape[1]
    m = A.shape[0]
    
    A_star = np.c_[A,-b]
    C = np.zeros((n+1,n+1))
    C[0:n,n] = c
    C[n,0:n] = -C[0:n,n].T
    
    yA = np.r_[np.zeros((m,m)), -A_star.T, np.zeros((n+1, m))]
    xA = np.r_[A_star, C, np.eye(n+1)]
    sA = np.r_[np.zeros((m, n+1)), -np.eye(n+1), np.eye(n+1)]
    
    return np.c_[yA, xA, sA]

def NSSolve(A, b, c, H, g, eps = 0.000001, beta = 0.2, xi = 0.5):
    """
    Solve the non-symmetric conic problem with the Skajaa Ye PCA
    (A,b,c): data
    H: Hessian of barrier
    g: gradient of barrier
    beta: central path ball radius
    xi: corrector ball scaling
    """
    
    m = A.shape[0]
    nt = A.shape[1]
    n = nt + 1
    
    eta = beta * xi
    kx = eta + np.sqrt(2 * eta * eta + n)
    
    ap = 0.02 / kx
    ac = 1
    
    nA = SelfDualNewtonSystem(A, b, c)
    nb = np.zeros(nA.shape[0])
    
    y = np.zeros(m)
    x = np.ones(n)
    s = np.ones(n)
    
    gp = lambda u: extg(g, u)
    Hp = lambda u: extH(H, u)
    Pp = lambda u, w: Prox(u, w, Hp, gp)
    
    itr = 0
    
    while DualityGap(x, s) >= eps:
        # predictor phase
        
        itr += 1
        
        hess = Hp(x)
        mu = DualityGap(x, s)
        
        #todo: double check
        nb[:-n] = -nA[:-n,:] @ np.r_[y, x, s]
        nb[-n:] = -s
        
        nA[(m+n):,m:(m+n)] = mu * hess
        
        sol = np.linalg.solve(nA, nb)
        
        dy = sol[0:m]
        dx = sol[m:(m+n)]
        ds = sol[(m+n):]
        
        ls_alpha = LineSearch(Pp, beta, x, s, dx, ds, ap, ap + 2, 3)
        
        #print(str(ls_alpha) + " " + str(ap))
        
        y = y + ls_alpha * dy
        x = x + ls_alpha * dx
        s = s + ls_alpha * ds
        
        # corrector phase
        
        hess = Hp(x)
        mu = DualityGap(x, s)
        
        nb[:] = 0
        nb[-n:] = -Phi(x, s, gp)
        
        nA[(m+n):,m:(m+n)] = mu * hess
        
        sol = np.linalg.solve(nA, nb)
        
        dy = sol[0:m]
        dx = sol[m:(m+n)]
        ds = sol[(m+n):]
        
        ls_alpha = LineSearch(Pp, eta, x, s, dx, ds, ac, ac + 2, 3)
        
        #print(str(ls_alpha) + " " + str(ap))
        
        y = y + ls_alpha * dy
        x = x + ls_alpha * dx
        s = s + ls_alpha * ds
    
    print("Newton steps: " + str(itr))
    
    if x[-1] > 2 * s[-1] and x[-1] > eps:
        """
        P optimal
        """
        x = x[0:-1] / x[-1]
        print("Optimal Value: " + str(np.dot(c, x)) + " with x = \n" + str(x))
        return x
    elif s[-1] > 2 * x[-1] and s[-1] > eps:
        """
        P/D inf
        """
        print("Primal or Dual Infeasible")
        return None
    else:
        """
        Pathology encountered
        """
        return None
