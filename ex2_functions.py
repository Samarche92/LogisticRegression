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
    
    np.ndarray.flatten(grad)
    return (J,grad)

def objective(theta,X,y):
    return costFunction(theta,X,y)[0]

def grad(theta,X,y):
    res=costFunction(theta,X,y)[1]
    n=len(res)
    np.ndarray.flatten(res)
    return res