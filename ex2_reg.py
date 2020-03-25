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

data=np.loadtxt(fname="ex2data2txt",delimiter=",")
X = data[:,0:2]
y = data[:,2]
