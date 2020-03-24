#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for logistic regression project
"""

import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm


## Load Data
# The first two columns contains the exam scores and the third column
# contains the label.

data=np.loadtxt(fname="ex2data1.txt",delimiter=",")
X = data[:,0:2]
y = data[:,2]

m=len(X) # number of training examples

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