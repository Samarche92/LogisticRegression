#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module containing useful functions for ex2 script
"""

import numpy as np

def sigmoid(z):
    g=np.zeros(np.size(z))
    g=1/(1+np.exp(-z))
    return g

def costFunction(theta, X, y):
    
    m=len(X)
    h=sigmoid(np.matmul(X,theta))
    J=-y*np.log(h)-(1-y)*np.log(1-h)
    J=np.sum(J)/m
    
    grad=np.matmul(np.transpose(X),h-y)
    grad=grad/m
    
    return (J,grad)