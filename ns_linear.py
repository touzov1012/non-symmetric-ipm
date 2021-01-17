# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:24:15 2021

@author: alex
"""

import numpy as np
from ns_utility import *

def g(x):
    """
    Gradient of the log barrier function
    """
    
    return -1.0 / x

def H(x):
    """
    Hessian of the log barrier function
    """
    
    return np.diag(1.0 / (x * x))

def Solve(A, b, c):
    """
    Solve the linear program using non-symmetric cone alg.
    Note: This isnt optimal for LP, but used for tests...
    """
    
    return NSSolve(A, b, c, H, g)


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
    
    return [A, b, c]
    