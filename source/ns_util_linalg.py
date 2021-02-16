# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:11:22 2021

Utility functions for anything linear algebra / tensor structure related.

@author: alex
"""

import numpy as np

def DiagExpand(A):
    """
    Paste A on the diagonal of an array one dimension higher.
    """
    
    G = np.zeros(A.shape + A.shape[-1:])
    Gd = np.diagonal(G, axis1=-2, axis2=-1)
    Gd.setflags(write=True)
    Gd[:] = A
    
    return G