# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 18:04:18 2021

@author: alex
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class SOLN:
    """
    Data used for storing iterations and final solution to conic programs.
    (y,x,s): primal/dual solutions
    (xtau, stau): self dual additional vars
    """
    
    y: np.ndarray
    x: np.ndarray
    s: np.ndarray
    xtau: float
    stau: float
    
    def MAD(self, other, fac):
        self.y += other.y * fac
        self.x += other.x * fac
        self.s += other.s * fac
        self.xtau += other.xtau * fac
        self.stau += other.stau * fac

@dataclass
class DATA:
    """
    Data for the problem instance.
    (A,b,c): LHS, RHS, and objective
    """
    
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray

@dataclass
class STATS:
    """
    Various metrics of quality of current iterate.
    """
    
    inside: bool
    H: np.ndarray
    g: np.ndarray
    phi: np.ndarray
    proxi: float
    gap: float
    stime: float
    elapsed: float
    nsteps: int
    pres: float
    dres: float
    gres: float
    pval: float
    dval: float
    status: int
    
    def __repr__(self):
        
        stats = ''
        if self.status == 1:
            stats = 'P-D optimal'
        elif self.status == 2:
            stats = 'P/D infeasible'
        elif self.status == 3:
            stats = 'ill-posed / pathological'
        else:
            stats = 'working...'
        
        return repr(f'in-cone: {self.inside}, proxi: {self.proxi:.6f}, gap: {self.gap:.6f}, elapsed: {self.elapsed:.3f}, newton-steps: {self.nsteps}, pval: {self.pval:.4f}, dval: {self.dval:.4f}, status: {stats}')
    
@dataclass
class INIT:
    """
    Data used for initializing the solver.
    eps: target gap
    beta: predictor ball size
    eta: corrector ball size
    correctors: corrector steps
    (y0, x0, s0): initial point
    """
    
    y0: np.ndarray
    x0: np.ndarray
    s0: np.ndarray
    eps: float = 0.000001
    beta: float = 0.2
    eta: float = 0.1
    correctors: int = 1
    
@dataclass
class CONE:
    """
    Parameters for a given cone
    gHc: function of primal x, returns gradient and Hessain and if lies in cone.
    nu: barrier parameter
    """
    
    gHc: staticmethod
    nu: int
    