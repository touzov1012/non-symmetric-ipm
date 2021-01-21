# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:39:41 2021

@author: alex
"""

import numpy as np

def Partition(n, d):
    """
    create all ways to decompose d as a sum of n positive integers
    """
    
    x = np.zeros(n, dtype = int)
    x[0] = d
    x = x[None]
    
    if d == 0 or n == 1:
        return x
    
    for i in range(1,d+1):
        xp = Partition(n-1, i)
        xs = np.c_[np.repeat(d - i, xp.shape[0])[None].T, xp]
        x = np.r_[x, xs]
    
    return x


def ChebyPts2(d):
    """
    Get the Chebyshev points in the interval [-1,1] for basis up to degree d
    """
    
    return np.polynomial.chebyshev.chebpts2(d+1)

def Padua(d):
    """
    Generate bivariate Padua points up to degree d
    """
    
    C0 = ChebyPts2(d)
    C1 = ChebyPts2(d+1)
    
    CE0 = C0[::2]
    CE1 = C1[::2]
    CO0 = C0[1::2]
    CO1 = C1[1::2]
    
    A1 = np.array(np.meshgrid(CE0,CO1)).T.reshape(-1,2)
    A2 = np.array(np.meshgrid(CO0,CE1)).T.reshape(-1,2)
    
    return np.r_[A1,A2]

def ChebyGrid(n, d):
    """
    Create a grid from the Chebyshev points in n dimensions for degree up to d
    and return the points of the grid as rows of a 2D array
    """
    
    x = []
    for i in range(n):
        x.append(ChebyPts2(d+i))
    
    return np.array(np.meshgrid(*x)).T.reshape(-1,n)

def ChebyVandermonde(n, d):
    """
    Create a vandermonde matrix with (d+1)^n rows and U columns
    where U is the dimension of poly in n variats up to degree d.
    Points are taken from the Chebyshev grid function
    """
    
    grid = ChebyGrid(n, d)
    
    P = Partition(n+1, d)
    P = P[:,1:]
    
    U = P.shape[0]
    ind = np.eye(2*d+1)
    
    V = np.ones((grid.shape[0], U))
    
    for i in range(U):
        for j in range(n):
            V[:,i] *= np.polynomial.chebyshev.chebval(grid[:,j], ind[:,P[i,j]])
    
    return V

def MaxVolumeSubMat(A):
    """
    Given a full rank m x n matrix with m > n, try to find the indices
    of the submatrix which has maximum volume using a greedy row search
    """
    
    A = A.copy()
    ind = np.array([],dtype=int)
    
    m = A.shape[0]
    n = A.shape[1]
    
    last_row = np.zeros(n)
    last_val = 1
    
    for i in range(n):
        
        best_row = A[0,:] - np.dot(A[0,:],last_row) / last_val * last_row
        best_val = np.dot(best_row, best_row)
        best_ind = 0
        
        for j in range(m):
            test_row = A[j,:] - np.dot(A[j,:],last_row) / last_val * last_row
            test_val = np.dot(test_row, test_row)
            
            if test_val > best_val:
                best_row = test_row
                best_val = test_val
                best_ind = j
                
            A[j,:] = test_row
        
        last_row = best_row
        last_val = best_val
        
        ind = np.append(ind, best_ind)
    
    return ind
            
    
    
    
    
