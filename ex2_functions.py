#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module containing useful functions for ex2 script
"""

import numpy as np
from matplotlib import pyplot as plt 

def plotData(X,y,label_1='feature 1',label_2='feature 2'):
    """ plot training data """
    # find positive and negative cases
    
    pos=np.where(y)
    pos=pos[0]
    neg=np.where(1-y)
    neg=neg[0]

    #plot data
    print('Plotting data\n')
    #plt.figure()
    plt.xlabel(label_1) 
    plt.ylabel(label_2) 
    plt.plot(X[pos,0],X[pos,1],'yo') 
    plt.plot(X[neg,0],X[neg,1],'k+')
    plt.legend(['Positive','Negative'])
    plt.show()

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    """ compute cost function and gradient """
    
    m=len(X)
    n=len(theta)
    theta=np.reshape(theta,[n,1]) #reshaping into vector
    h=sigmoid(np.matmul(X,theta))
    J=-y*np.log(h)-(1-y)*np.log(1-h)
    J=np.sum(J)/m
    
    grad=np.matmul(np.transpose(X),h-y)
    grad=grad/m
    
    return (J,grad)

def predict(theta,X):
    """ Given input data and model parameters,
    outputs predicted classification """
    
    m=len(X)
    p=np.zeros([m,1])
    test=sigmoid(np.matmul(X,theta))
    pos=np.where(test>=0.5)
    pos=pos[0]
    p[pos]=1
    return p

def mapFeature(x1,x2,degree=6):
    """ maps the two input features to quadratic features"""

    out = np.ones_like(x1)
    for i in range(1,degree+1):
        for j in range(i+1):
            out=np.column_stack((out,x1**(i-j)*x2**j)) 
    
    return out

def costFunctionReg(theta, X, y, L):
    """ compute regularized cost function and gradient """
    
    m = len(y) # number of training examples
    n=len(theta)
    theta=np.reshape(theta,[n,1])

    (J,grad)=costFunction(theta,X,y)
    
    theta2=np.copy(theta)
    theta2[0]=0
    
    J+=L*np.linalg.norm(theta2)**2/(2*m)
    grad+=L*theta2/m
    
    return (J,grad)

def plotDecisionBoundary(theta,X,y,label_1='feature 1',label_2='feature 2'):
    """ plot training data and decision boundary """   
    
    if np.shape(X)[1]<=3:
        plot_x = np.linspace(np.min(X[:,1])-2,np.max(X[:,1])+2)
        plot_y = theta[0]+theta[1]*plot_x
        plot_y = -plot_y/theta[2]
        plt.plot(plot_x,plot_y)
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5)
        v = np.linspace(-1, 1.5)
        
        z=np.zeros([len(u),len(v)])
        
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j]=np.matmul(mapFeature(u[i],v[j]),theta)
        
        plt.contour(u,v,np.transpose(z),0)
        
    plt.legend(['Positive','Negative','decision boundary'])
    plt.show()
    
        
    