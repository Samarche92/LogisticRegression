#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for logistic regression project
"""

import numpy as np
from matplotlib import pyplot as plt 
import ex2_functions as fct2


## Load Data
# The first two columns contains the exam scores and the third column
# contains the label.

data=np.loadtxt(fname="ex2data1.txt",delimiter=",")
X = data[:,0:2]
y = data[:,2]

m=len(X) # number of training examples
n = np.size(X,1)

y=np.reshape(y,[m,1])  #reshaping into vector

# find positive and negative cases
pos=np.where(y)
pos=pos[0]
neg=np.where(1-y)
neg=neg[0]

#plot data
print('Plotting data\n')

plt.figure()
plt.xlabel("Exam 1 score") 
plt.ylabel("Exam 2 score") 
plt.plot(X[pos,0],X[pos,1],'yo') 
plt.plot(X[neg,0],X[neg,1],'k+')
plt.legend(['admitted','not admitted'])
plt.show()

wait = input("Program paused. Press enter to continue \n")

# Add ones to training set for the intercept term
X=np.column_stack((np.ones([m,]),data[:,0:2])) 

initial_theta=np.zeros([n+1,1]) #initialize fitting parameters

# Compute and display initial cost and gradient
(cost, grad) = fct2.costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

# Compute and display cost and gradient with non-zero theta
test_theta = np.reshape(np.array([-24, 0.2, 0.2]),[3,1])
#test_theta = np.reshape(test_theta,[3,1]) #reshaping into vector
(cost, grad) = fct2.costFunction(test_theta, X, y);

print('\nCost at test theta: {}\n'.format(cost))
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

wait = input("Program paused. Press enter to continue \n")