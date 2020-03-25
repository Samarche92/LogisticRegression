#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for regularized logistic regression
"""
import numpy as np
from matplotlib import pyplot as plt 
import ex2_functions as fct2
from scipy.optimize import minimize


## Load Data
# The first two columns contains the test scores and the third column
# contains the label.

data=np.loadtxt(fname="ex2data2.txt",delimiter=",")
X = data[:,0:2]
m=len(X)
y = data[:,2]

y=np.reshape(y,[m,1]) 

fct2.plotData(X,y,'test score 1', 'test score 2')
wait = input("Program paused. Press enter to continue \n")

# Add Polynomial Features

X = fct2.mapFeature(X[:,0],X[:,1])

# Initialize fitting parameters
initial_theta = np.zeros([np.shape(X)[1],1])

# Set regularization parameter L to 1
L = 1;

# Compute and display initial cost and gradient for regularized logistic regression

(cost, grad) = fct2.costFunctionReg(initial_theta, X, y, L)

print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')