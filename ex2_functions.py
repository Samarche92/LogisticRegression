#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module containing useful functions for ex2 script
"""

import numpy as np

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    
    m=len(X)
    n=len(theta)
    theta=np.reshape(theta,[n,1]) #reshaping into vector
    h=sigmoid(np.matmul(X,theta))
    J=-y*np.log(h)-(1-y)*np.log(1-h)
    J=np.sum(J)/m
    
    grad=np.matmul(np.transpose(X),h-y)
    grad=grad/m
    
    np.ndarray.flatten(grad)
    return (J,grad)

def predict(theta,X):
    m=len(X)
    p=np.zeros([m,1])
    test=sigmoid(np.matmul(X,theta))
    pos=np.where(test>=0.5)
    pos=pos[0]
    p[pos]=1
    return p