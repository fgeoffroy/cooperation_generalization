from __future__ import division
import numpy as np
from random import *
from copy import *
import math as math
from scipy.optimize import root
from Parameters import *



def wr(number):		#Truncating floats in file
	return repr(float('%.6f' % number))



def encrypting(param, alphas):
	def func(x):
		y = 0
		for i in range(InputComplexity):
			for j in range(NbOfInputPerParam):
				y += alphas[i,j] * math.cos((i+1) * math.pi * x[j])
		y += MaxValueGameParameter / 2
		y -= param
		outputV = np.zeros(NbOfInputPerParam)
		outputV[0] = y
		return outputV

	isSolutionConform = False
	while not isSolutionConform:
		sol = root(func, np.random.uniform(0, 1, NbOfInputPerParam))
		if sol.success == 1 and (sol.x > 0).all() and (sol.x < 1).all():
			isSolutionConform = True
	return sol.x
