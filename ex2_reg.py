#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for regularized logistic regression
"""
import numpy as np
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
num=0

fct2.plotData(X,y,'test score 1', 'test score 2')
wait = input("Program paused. Press enter to continue \n")

# Add Polynomial Features

X = fct2.mapFeature(X[:,0],X[:,1])
n=np.shape(X)[1]

# Initialize fitting parameters
initial_theta = np.zeros([n,1])

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

wait = input("Program paused. Press enter to continue \n")

# Compute and display cost and gradient with all-ones theta and L = 10
test_theta = np.ones([np.shape(X)[1],1])
(cost, grad) = fct2.costFunctionReg(test_theta, X, y, 10)

print('\nCost at test theta (with L = 10): {}\n'.format(cost))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

wait = input("Program paused. Press enter to continue \n")

#  Run minimize to obtain the optimal theta
#  This function will return the solution in object res

res=minimize(lambda t : fct2.costFunctionReg(t,X,y,L)[0],np.ndarray.flatten(initial_theta),
            jac=lambda t : np.ndarray.flatten(fct2.costFunctionReg(t,X,y,L)[1]),
            options={'maxiter':400,'disp':True})

# Retrieve solution from res object
theta=np.reshape(res.x,[n,1])
cost=res.fun

fct2.plotDecisionBoundary(theta,X,y,'test score 1', 'test score 2')
