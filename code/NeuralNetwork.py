from __future__ import division
import numpy as np
import math as math
from random import *
from Utilities import *
from Parameters import *
import sys


class NeuralNetwork:

	def __init__(self, playerType):
		if playerType == "Recipient":
			nbParam = 3		#{b; c; ra} for the Recipient
		elif playerType == "Investor":
			nbParam = 2		#{b; c} for the Investor
		nbInputs = NbOfInputPerParam * nbParam + NbOfConfoundingFeatures
		self.activationInput = np.ones(nbInputs + 1)
		self.activationHidden = None
		self.activationOutput = None
		self.weightIn = np.random.normal(0, 0.1, [nbInputs + 1, NbOfHiddenNeurons])	# +1 for the bias
		self.weightOut = np.random.normal(0, 0.1, [NbOfHiddenNeurons + 1, 1])	# idem


	def forwardActivation(self, inputMatrix):	#2 dimension numpy vector as input (+ its row number)
		# input activations, adding bias neuron
		subSetSize = inputMatrix.shape[0]
		self.activationInput = np.append(inputMatrix, np.ones((subSetSize, 1)), axis=1)

		# hidden activations
		self.activationHidden = np.dot(self.activationInput, self.weightIn)
		self.activationHidden = self.sigmoid(self.activationHidden)
		self.activationHidden = np.append(self.activationHidden, np.ones((subSetSize, 1)), axis=1)

		# output activations
		self.activationOutput = np.dot(self.activationHidden, self.weightOut)
		if IsSigmoidOutput:
			self.activationOutput = self.sigmoid(self.activationOutput)
			output = self.activationOutput
		else:
			output = self.activationOutput
			# Constrain between 0 and 1 in the absence of sigmoid function in the output
			output[output < 0] = 0
			output[output > 1] = 1
		return output



	def backward(self, targetV, selectionStrengthV, updateSubSetSize):
		subSetSize = targetV.shape[0]
		#Error, target result
		deltaOutput = self.activationOutput - targetV 		#This works for both sigmoidal activation + cross-entropy error AND linear activation + mean squared error
		deltaOutput *= selectionStrengthV	#Here we take into account the fact that selection strength might vary between games

		#Backward deltas
		deltaHidden = self.weightOut[:-1].T * deltaOutput		#No need to compute the delta of the biais unit
		deltaHidden *= self.activationHidden[:, :-1] * (1 - self.activationHidden[:, :-1])

		#Updating weights
		gradWeightOut = np.dot(self.activationHidden.T, deltaOutput)
		gradWeightOut /= updateSubSetSize
		self.weightOut -= gradWeightOut * LearningRate

		gradWeightIn = np.dot(self.activationInput.T, deltaHidden)
		gradWeightIn /= updateSubSetSize
		self.weightIn -= gradWeightIn * LearningRate


	def sigmoid(self, array):
		return 1 / (1 + np.exp(-array))
