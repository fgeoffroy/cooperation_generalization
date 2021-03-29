from __future__ import division
import math as math
from random import *
import numpy as np
from Utilities import *
from Parameters import *

class Game:

	def __init__(self, typeOfSet, type, index, alphaBV, alphaCV, alphaRaV):
		self.typeOfSet = typeOfSet
		self.type = type		# 0: Selfish ; 1: Cooperative ; 2: Interdependent ; 3: Wasteful
		self.index = index
		if self.type == 2:
			self.commonInterest = False		#Determines whether there is a common interest in an Interdependent opportunity
		else:
			self.commonInterest = None

		#Randomizing the game parameters (general constraints: -1 < c < 1 ; 0 < b < 1 ; 0 < ra < 1)
		if self.type == 0:		#SELFISH c < 0 ; b > 0 ; ra = 0
			self.c = - random()
			self.b = random()
			self.ra = 0
		elif self.type == 1:	#COOPERATIVE c > 0 ; b > c ; ra = 0
			self.c = 1
			self.b = 0
			while self.b <= self.c:
				self.c = random()
				self.b = random()
			self.ra = 0
		elif self.type == 2:	#INTERDEPENDENT c > 0 ; two possibilities for ra depending on whether there is a common interest (ra = 0 < c or ra > c) ; b > c - ra
			self.c = random()
			self.b = random()
			self.ra1 = 0
			self.ra2 = random()
			while not ( ( self.ra2 > self.c ) and ( self.b > self.c ) ):
				self.c = random()
				self.b = random()
				self.ra2 = random()
			self.raV = [0, self.ra2]
		elif self.type == 3:	#WASTEFUL c > 0 ; b < c ; ra = 0
			self.c = 0
			self.b = 1
			while self.b >= self.c:
				self.c = random()
				self.b = random()
			self.ra = 0

		#Generating the ANN's inputs derived from the game parameters
		if InputComplexity > 0:
			self.inputCostV = encrypting((self.c + 1)/2, alphaCV)	#Normalizing c in [0, 1]
			self.inputBenefitV = encrypting(self.b, alphaBV)
			if type == 2:
				self.inputAutomaticRewardV = [encrypting(self.raV[0], alphaRaV), encrypting(self.raV[1], alphaRaV)]
			else:
				self.inputAutomaticRewardV = encrypting(self.ra, alphaRaV)
		else:
			self.inputCostV = np.array([(self.c + 1)/2])			#Normalizing c in [0, 1]
			self.inputBenefitV = np.array([self.b])
			if type == 2:
				self.inputAutomaticRewardV = [np.array([self.raV[0]]), np.array([self.raV[1]])]
			else:
				self.inputAutomaticRewardV = np.array([self.ra])

		#Generating the ANN's inputs generated derived from spurious features. (Note that an Interdependent Game has two different possible sets of spurious features for the Recipient, depending on the value of the automatic reward)
		if type == 2:
			self.inputConfoundingV = [np.random.rand(NbOfConfoundingFeatures), np.random.rand(NbOfConfoundingFeatures)]
			if GameStructure == "CoEvolution":
				# self.investorInputsV = np.concatenate((self.inputBenefitV, self.inputCostV))
				self.investorInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputConfoundingV[0]))
		else:
			self.inputConfoundingV = np.random.rand(NbOfConfoundingFeatures)
			#self.recipientInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputAutomaticRewardV))
			self.recipientInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputAutomaticRewardV, self.inputConfoundingV))
			if GameStructure == "CoEvolution":
				#self.investorInputsV = np.concatenate((self.inputBenefitV, self.inputCostV))
				self.investorInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputConfoundingV))




	def randomizeCommonInterest(self):
		# For an Interdependent game, randomizes the common interest (i.e. one of the two possible automatic reward is randomly choosen)
		# Updates the value of the automatic reward and the ANN's inputs accordingly
		rand = randint(0, 1)
		if rand == 0:
			self.commonInterest = False
		else:
			self.commonInterest = True
		self.ra = self.raV[rand]
		# self.recipientInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputAutomaticRewardV[rand]))
		self.recipientInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputAutomaticRewardV[rand], self.inputConfoundingV[rand]))



	def setCommonInterest(self, isCommonInterest):
		# For an Interdependent game, set it to have a common interest (i.e. ra > c) or no common interest (i.e. ra < c)
		# Updates the value of the automatic reward and the ANN's inputs accordingly
		if isCommonInterest:
			self.commonInterest = True
			ind = 1
		else:
			self.commonInterest = False
			ind = 0
		self.ra = self.raV[ind]
		# self.recipientInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputAutomaticRewardV[ind]))
		self.recipientInputsV = np.concatenate((self.inputBenefitV, self.inputCostV, self.inputAutomaticRewardV[ind], self.inputConfoundingV[ind]))
