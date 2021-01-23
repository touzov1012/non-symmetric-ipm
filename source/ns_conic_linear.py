# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:24:15 2021

@author: alex
"""

import numpy as np
from ns_conic import *
from ns_structs import *

def gHc(x):
    """
    Gradient, Hessian, Cone check
    """
    
    return [-1.0 / x, np.diag(1.0 / (x * x)), (x > 0).all()]


def Solve(dData):
    """
    Solve the linear program using non-symmetric cone alg.
    Note: This isnt optimal for LP, but used for tests...
    """
    
    m = dData.A.shape[0]
    n = dData.A.shape[1]
    
    dInit = INIT(np.zeros(m), np.ones(n), np.ones(n))
    dCone = CONE(gHc, n)
    
    return NSSolve(dData, dInit, dCone)


def TestLP():
    """
    Generate a test LP problem in standard form
    opt: 6450
    x = (700, 700, 50)
    """
    
    A = np.array([[0.4, 0.30, 0.2, -1,  0,  0, 0, 0, 0],
                  [0.4, 0.35, 0.2,  0, -1,  0, 0, 0, 0],
                  [0.2, 0.35, 0.6,  0,  0, -1, 0, 0, 0],
                  [  1,    0,   0,  0,  0,  0, 1, 0, 0],
                  [  0,    1,   0,  0,  0,  0, 0, 1, 0],
                  [  0,    0,   1,  0,  0,  0, 0, 0, 1]])
    
    b = np.array([500, 300, 300, 700, 700, 700])
    
    c = np.array([5, 4, 3, 0, 0, 0, 0, 0, 0])
    
    return DATA(A, b, c)
    
