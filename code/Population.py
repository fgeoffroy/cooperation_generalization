from __future__ import division
import sys
from random import *
from copy import *
from NeuralNetwork import *
from Game import *
from Parameters import *
from Utilities import *
import numpy as np
import math as math
from scipy.optimize import root
import os






class Population:

	def __init__(self):
		# There are two players: the Investor and the Recipient
		self.Recipient = NeuralNetwork("Recipient")
		self.metricsRecipient = np.zeros(10)	#nb of each type of metrics// 0: SELFISH True Negatives ; 1: SELFISH False Positives ; 2: COOPERATIVE True Positives ; 3: COOPERATIVE False Negatives ; 4: INTERDEPENDENT True Positives ; 5: INTERDEPENDENT False Positives ; 6: INTERDEPENDENT True Negatives ; 7: INTERDEPENDENT False Negatives ; 8: WASTEFUL True Negatives ; 9: WASTEFUL False Positives
		if GameStructure == "CoEvolution":
			self.Investor = NeuralNetwork("Investor")
		self.metricsInvestor = np.zeros(12)	#nb of each type of metrics// 0: SELFISH True Positives ; 1: SELFISH False Negatives ; 2: COOPERATIVE True Negatives ; 3: COOPERATIVE False Positives ; 4: COOPERATIVE True Positives ; 5: COOPERATIVE False Negatives ; 6: INTERDEPENDENT True Positives ; 7: INTERDEPENDENT False Negatives ; 8: WASTEFUL True Negatives ; 9: WASTEFUL False Positives ; 10: WASTEFUL True Positives ; 11: WASTEFUL False Negatives


		#List of parameters for encrypting b and c into input(s) for neural networks (according to JBs model)
		self.alphasBenefit, self.alphasCost, self.alphasAutomaticReward = self.initiate_encrypting()

		#Creation of the training set and the test set. Distinguish several types and their numbers. Create the object Game with a function that makes it random
		self.trainingSet = self.generate_set("Training")
		self.testSet = self.generate_set("Test")

		# Verification of the training set size
		if len(self.trainingSet) != SelfishNb + CoopNb + InterNb + WasteNb:
			sys.exit("Warning : Error in training set construction")

		shuffle(self.trainingSet)
		self.trainingSetSize = len(self.trainingSet)


		r = random()
		resultFolder = os.getcwd() + "/../results/"
		self.separatedSelectionMetricsName = resultFolder + "metrics_separated_selection" + repr(r) + ".txt"
		self.selectionMetricsName = resultFolder + "metrics_selection" + repr(r) + ".txt"
		self.separatedSelectionName = resultFolder + "totalSet_separated_selection" + repr(r) + ".txt"
		self.totalSetName = resultFolder + "totalSet_selection" + repr(r) + ".txt"
		if HeatMapFile: self.selectionHeatMapName = resultFolder + "heatmap_selection" + repr(r) + ".txt"


	def initiate_encrypting(self):
		if InputComplexity > 0:
			alphaMax = (MaxValueGameParameter / 3) * math.sqrt(6 / (InputComplexity * NbOfInputPerParam))
			alphasBenefit = self.generate_alphas(alphaMax)
			alphasCost = self.generate_alphas(alphaMax)
			alphasAutomaticReward = self.generate_alphas(alphaMax)
		else:
			alphasBenefit = None
			alphasCost = None
			alphasAutomaticReward = None
		return alphasBenefit, alphasCost, alphasAutomaticReward


	def generate_alphas(self, alphaMax):
		isAlphaVConform = False
		while not isAlphaVConform:
			#Generating the alpha vector
			alphas = np.random.uniform(-alphaMax, alphaMax, (InputComplexity, NbOfInputPerParam))

			#Checking if this particular alpha vector is conform: if it can generate solution in the whole [0, 1] range
			#Checking 0
			def func0(x):
				y = 0
				for i in range(InputComplexity):
					for j in range(NbOfInputPerParam):
						y += alphas[i,j] * math.cos((i+1) * math.pi * x[j])
				y += MaxValueGameParameter / 2
				outputV = np.zeros(NbOfInputPerParam)	#trick to use the "root" function to solve a system of 1 equation and NbOfInputPerParam variables
				outputV[0] = y
				return outputV
			isAlphaVConform0 = False
			for i in range(100):
				sol = root(func0, np.random.uniform(0, 1, NbOfInputPerParam))
				if sol.success == 1 and (sol.x > 0).all() and (sol.x < 1).all():
					isAlphaVConform0 = True
					break
			#Checking 1
			def func1(x):
				outputV = func0(x)
				outputV[0] = outputV[0] - 1
				return outputV
			isAlphaVConform1 = False
			for i in range(100):
				sol = root(func1, np.random.uniform(0, 1, NbOfInputPerParam))
				if sol.success == 1 and (sol.x > 0).all() and (sol.x < 1).all():
					isAlphaVConform1 = True
					break

			isAlphaVConform = isAlphaVConform0 and isAlphaVConform1
		return alphas


	def generate_set(self, typeOfSet):
		set = []

		if typeOfSet == "Training":
			countAllOpp = np.array([SelfishNb, CoopNb, InterNb, WasteNb])
		else:
			countAllOpp = np.repeat(1000, 4)

		index = 1
		for type in [0, 1, 2, 3]:		# 0: Selfish ; 1: Cooperative ; 2: Interdependent ; 3: Wasteful
			for i in range(countAllOpp[type]):
				opp = Game(typeOfSet, type, index, self.alphasBenefit, self.alphasCost, self.alphasAutomaticReward)
				set.append(opp)
				index += 1

		return set


	def evolve(self):
		self.separated_selection()
		self.create_totalSet_file(self.separatedSelectionName)
		if HeatMapFile: self.create_heatmap_file(self.selectionHeatMapName)
		self.selection()
		self.create_totalSet_file(self.totalSetName)




	def separated_selection(self):
		self.create_metrics_file(self.separatedSelectionMetricsName)
		for i in range(1, NbOfIterationsSeparatedSelection + 1):	#to include the last iteration
			self.randomizeInterdependentGames()
			for j in range(0, self.trainingSetSize , BatchSize):
				if j + BatchSize <= self.trainingSetSize - 1:
					subSetSize = BatchSize
					subSet = self.trainingSet[j : j + subSetSize]
				else:
					subSetSize = self.trainingSetSize - j
					subSet = self.trainingSet[j:]
					shuffle(self.trainingSet)

				typeV = [opp.type for opp in subSet]
				typeV = np.vstack(typeV)
				costV = [opp.c for opp in subSet]
				costV = np.vstack(costV)
				automaticRewardV = [opp.ra for opp in subSet]
				automaticRewardV = np.vstack(automaticRewardV)
				commonInterestV = [opp.commonInterest for opp in subSet]
				commonInterestV = np.vstack(commonInterestV)

				#Recipient (forward, then target and backward)
				rewardV = self.rewarding(subSet)
				rewardedV = rewardV + automaticRewardV - costV >= 0
				targetRewardV = np.zeros((subSetSize, 1))	# in the preselection phase, we train the Recipient to reward r=0 for every interaction
				rewardSelectionStrengthV = np.ones((subSetSize, 1))
				self.Recipient.backward(targetRewardV, rewardSelectionStrengthV, subSetSize)

				if GameStructure == "InvestorDarwinianDemon" or GameStructure == "RecipientAlone":
					testV = np.zeros((subSetSize, 1))
				elif GameStructure == "CoEvolution":
					#Investor (forward, then target and backward)
					testV = self.investing(subSet)
					targetTestV = np.zeros((subSetSize, 1))	# in the preselection phase, we train the Investor to never "pay to see" (i.e. probabilistic output = 0)
					testSelectionStrengthV = np.ones((subSetSize, 1))
					self.Investor.backward(targetTestV, testSelectionStrengthV, subSetSize)

				#Metrics
				self.update_metrics(typeV, commonInterestV, rewardedV, testV > 0)

			if i % PrintLag == 0:
				self.normalize_metrics()
				self.update_metrics_file(i, self.separatedSelectionMetricsName)
				self.metricsRecipient = np.zeros(10)
				self.metricsInvestor = np.zeros(12)


	def selection(self):
		self.create_metrics_file(self.selectionMetricsName)
		if HeatMapFile: self.update_heatmap_file(0, self.selectionHeatMapName)
		for i in range(1, NbOfIterations + 1):	#to include the last iteration
			self.randomizeInterdependentGames()
			for j in range(0, self.trainingSetSize , BatchSize):
				if j + BatchSize <= self.trainingSetSize - 1:
					subSetSize = BatchSize
					subSet = self.trainingSet[j : j + subSetSize]
				else:
					subSetSize = self.trainingSetSize - j
					subSet = self.trainingSet[j:]
					shuffle(self.trainingSet)

				#FORWARD
				rewardV = self.rewarding(subSet)
				typeV = [opp.type for opp in subSet]
				typeV = np.vstack(typeV)
				costV = [opp.c for opp in subSet]
				costV = np.vstack(costV)
				automaticRewardV = [opp.ra for opp in subSet]
				automaticRewardV = np.vstack(automaticRewardV)
				commonInterestV = [opp.commonInterest for opp in subSet]
				commonInterestV = np.vstack(commonInterestV)
				rewardedV = rewardV + automaticRewardV - costV >= 0
				if GameStructure == "RecipientAlone":
					testV = np.logical_or(typeV == 0, typeV == 2)	#Investor should "pay to see" in Selfish and Interdependent (because c - ra < 0, at least at some occasion)
				elif GameStructure == "InvestorDarwinianDemon":
					#testV = np.logical_or(typeV == 0, typeV == 2, np.logical_and(typeV == 1, rewardedV == 1))	#Investor should test in Selfish and Interdependent and [Coop if Recipient rewards]
					testV = np.logical_or.reduce((typeV == 0, typeV == 2, np.logical_and(np.logical_or(typeV == 1, typeV == 3), rewardedV == 1)))	#Investor should test in Selfish and Interdependent and [Coop or Waste if Recipient rewards]
				elif GameStructure == "CoEvolution":
					testV = self.investing(subSet)

					#TARGETS AND BACKWARD for Investor
					targetTestV = np.logical_or(rewardedV, typeV == 2)	#Investor should invest if Inter or r - c >= 0
					#Selection strength for Investor (depends on both the investor output and the game's type)
					testSelectionStrengthV = testV != targetTestV	#We don't train the Investor on opportunities for which it is already correct
					selectionCoopOrInterV = np.logical_or(typeV == 1, typeV == 2)
					selectionSelfishOrWasteful = NonCooperationCost * np.logical_or(typeV == 0, typeV == 3)
					testSelectionStrengthV = testSelectionStrengthV * (selectionCoopOrInterV + selectionSelfishOrWasteful)	#For Cooperative and Interdependent games, the selection strength is normalized to 1. For Selfish and Wasteful games, the factor "NonCooperationCost" is applied (between 0 and 1).
					testUpdateSubSetSize = testSelectionStrengthV[testSelectionStrengthV > 0].size
					if testUpdateSubSetSize > 0:
						self.Investor.backward(targetTestV, testSelectionStrengthV, testUpdateSubSetSize)

				#TARGETS AND BACKWARD for Recipient
				testedV = testV > 0
				targetRewardV = np.logical_and(testV > 0, np.logical_or(typeV == 1, np.logical_and(typeV == 2, commonInterestV == 0)))	#The Recipient should reward for Inter (when ra < c) and Coop if I tests
				targetRewardV = targetRewardV * (costV - automaticRewardV + Extra)
				#Selection strength for Recipient (depends on both the investor output and the game's type)
				rewardSelectionStrengthV = testV		#Selection strength of reward linearily increases with the probability that the Investor tests
				if GameStructure != "CoEvolution":
					selectionCoopOrInterV = np.logical_or(typeV == 1, typeV == 2)
					selectionSelfishOrWasteful = NonCooperationCost * np.logical_or(typeV == 0, typeV == 3)
				rewardSelectionStrengthV = rewardSelectionStrengthV * (selectionCoopOrInterV + selectionSelfishOrWasteful)	#For Cooperative and Interdependent games, the selection strength is normalized to 1. For Selfish and Wasteful games, the factor "NonCooperationCost" is applied (between 0 and 1).
				rewardUpdateSubSetSize = rewardSelectionStrengthV[rewardSelectionStrengthV > 0].size
				if rewardUpdateSubSetSize > 0:
					self.Recipient.backward(targetRewardV, rewardSelectionStrengthV, rewardUpdateSubSetSize)

				#Metrics
				self.update_metrics(typeV, commonInterestV, rewardedV, testedV)

			if i % PrintLag == 0:
				self.normalize_metrics()
				self.update_metrics_file(i, self.selectionMetricsName)
				self.metricsRecipient = np.zeros(10)
				self.metricsInvestor = np.zeros(12)
				if HeatMapFile: self.update_heatmap_file(i, self.selectionHeatMapName)


	def randomizeInterdependentGames(self):
		for opp in self.trainingSet:
			if opp.type == 2:
				opp.randomizeCommonInterest()


	def allCommonInterest(self, isCommonInterest):
		for opp in np.concatenate((self.trainingSet, self.testSet)):
			if opp.type == 2:
				opp.setCommonInterest(isCommonInterest)


	def rewarding(self, subSet):
		inputV = [opp.recipientInputsV for opp in subSet]
		inputMatrix = np.vstack(inputV)
		outputR = self.Recipient.forwardActivation(inputMatrix)
		outputR *= MaxValueGameParameter	#Normalization
		return outputR


	def investing(self, subSet):
		inputV = [opp.investorInputsV for opp in subSet]
		inputMatrix = np.vstack(inputV)
		outputI = self.Investor.forwardActivation(inputMatrix)

		if IsSigmoidOutput:	 #If the output of the Investor's ANN is a sigmoid, we normalize it so that output between 0 and 0.5 gives a probability = 0 of investing. Accordinglyn outputs in [0.5; 1] are normalized for a probability of investing in [0; 1]
			outputI = (outputI - 0.5) * 2
			outputI[outputI < 0] = 0

		return outputI



	def update_metrics(self, typeV, commonInterestV, rewardedV, investedV):
		self.metricsRecipient[0] += np.sum(np.logical_and(rewardedV == 0, typeV == 0))	#Selfish True Negatives
		self.metricsRecipient[1] += np.sum(np.logical_and(rewardedV == 1, typeV == 0))	#Selfish False Positives
		self.metricsRecipient[2] += np.sum(np.logical_and(rewardedV == 1, typeV == 1))	#Coop True Positives
		self.metricsRecipient[3] += np.sum(np.logical_and(rewardedV == 0, typeV == 1))	#Coop False Negatives
		self.metricsRecipient[4] += np.sum(np.logical_and.reduce((rewardedV == 1, typeV == 2, commonInterestV == 0)))	#Interdependent True Positives
		self.metricsRecipient[5] += np.sum(np.logical_and.reduce((rewardedV == 1, typeV == 2, commonInterestV == 1)))	#Interdependent False Positives
		self.metricsRecipient[6] += np.sum(np.logical_and.reduce((rewardedV == 0, typeV == 2, commonInterestV == 1)))	#Interdependent True Negatives
		self.metricsRecipient[7] += np.sum(np.logical_and.reduce((rewardedV == 0, typeV == 2, commonInterestV == 0)))	#Interdependent False Negatives
		self.metricsRecipient[8] += np.sum(np.logical_and(rewardedV == 0, typeV == 3))	#Wasteful True Negatives
		self.metricsRecipient[9] += np.sum(np.logical_and(rewardedV == 1, typeV == 3))	#Wasteful False Positives

		self.metricsInvestor[0] += np.sum(np.logical_and(investedV == 1, typeV == 0))								#Selfish True Positives
		self.metricsInvestor[1] += np.sum(np.logical_and(investedV == 0, typeV == 0))								#Selfish False Negatives
		self.metricsInvestor[2] += np.sum(np.logical_and.reduce((investedV == 0, rewardedV == 0, typeV == 1)))		#Coop True Negatives
		self.metricsInvestor[3] += np.sum(np.logical_and.reduce((investedV == 1, rewardedV == 0, typeV == 1)))		#Coop False Positives
		self.metricsInvestor[4] += np.sum(np.logical_and.reduce((investedV == 1, rewardedV == 1, typeV == 1)))		#Coop True Positives
		self.metricsInvestor[5] += np.sum(np.logical_and.reduce((investedV == 0, rewardedV == 1, typeV == 1)))		#Coop False Negatives
		self.metricsInvestor[6] += np.sum(np.logical_and(investedV == 1, typeV == 2))								#Interdependent True Positives
		self.metricsInvestor[7] += np.sum(np.logical_and(investedV == 0, typeV == 2))								#Interdependent False Negatives
		self.metricsInvestor[8] += np.sum(np.logical_and.reduce((investedV == 0, rewardedV == 0, typeV == 3)))		#Wasteful True Negatives
		self.metricsInvestor[9] += np.sum(np.logical_and.reduce((investedV == 1, rewardedV == 0, typeV == 3)))		#Wasteful False Positives
		self.metricsInvestor[10] += np.sum(np.logical_and.reduce((investedV == 1, rewardedV == 1, typeV == 3)))	#Wasteful True Positives
		self.metricsInvestor[11] += np.sum(np.logical_and.reduce((investedV == 0, rewardedV == 1, typeV == 3)))	#Wasteful False Negatives


	def normalize_metrics(self):
		nbOfSELFMetrics = self.metricsRecipient[0] + self.metricsRecipient[1]
		nbOfCOOPMetrics = self.metricsRecipient[2] + self.metricsRecipient[3]
		nbOfINTERCommonMetrics = self.metricsRecipient[5] + self.metricsRecipient[6]
		nbOfINTERDivergentMetrics = self.metricsRecipient[4] + self.metricsRecipient[7]
		nbOfINTERMetrics = nbOfINTERCommonMetrics + nbOfINTERDivergentMetrics
		nbOfWASTEMetrics = self.metricsRecipient[8] + self.metricsRecipient[9]
		self.metricsRecipient[0] /= nbOfSELFMetrics
		self.metricsRecipient[1] /= nbOfSELFMetrics
		self.metricsRecipient[2] /= nbOfCOOPMetrics
		self.metricsRecipient[3] /= nbOfCOOPMetrics
		if nbOfINTERCommonMetrics != 0:
			self.metricsRecipient[5] /= nbOfINTERCommonMetrics
			self.metricsRecipient[6] /= nbOfINTERCommonMetrics
		if nbOfINTERDivergentMetrics != 0:
			self.metricsRecipient[4] /= nbOfINTERDivergentMetrics
			self.metricsRecipient[7] /= nbOfINTERDivergentMetrics
		self.metricsRecipient[8] /= nbOfWASTEMetrics
		self.metricsRecipient[9] /= nbOfWASTEMetrics
		self.metricsInvestor[0] /= nbOfSELFMetrics
		self.metricsInvestor[1] /= nbOfSELFMetrics
		self.metricsInvestor[2] /= nbOfCOOPMetrics
		self.metricsInvestor[3] /= nbOfCOOPMetrics
		self.metricsInvestor[4] /= nbOfCOOPMetrics
		self.metricsInvestor[5] /= nbOfCOOPMetrics
		if nbOfINTERMetrics != 0:
			self.metricsInvestor[6] /= nbOfINTERMetrics
			self.metricsInvestor[7] /= nbOfINTERMetrics
		self.metricsInvestor[8] /= nbOfWASTEMetrics
		self.metricsInvestor[9] /= nbOfWASTEMetrics
		self.metricsInvestor[10] /= nbOfWASTEMetrics
		self.metricsInvestor[11] /= nbOfWASTEMetrics



	def create_metrics_file(self, name):
		with open(name, "wb") as fileName:
			fileName.write('"SelfishNb" "CooperativeNb" "InterdependentNb" "WastefulNb" "MaxValueGameParameter" "InputComplexity" "NbOfInputPerParam" "NbOfConfoundingFeatures" "NonCooperationCost" "NbOfIterationsSeparatedSelection" "NbOfIterations" "PrintLag" "NbOfHiddenNeurons" "Extra" "ClassifierThreshold" "IsSigmoidOutput" "GameStructure"')
			fileName.write('"GradientMethod" "BatchSize" "LearningRate"')
			fileName.write('\n')

			fileName.write(repr(SelfishNb) + " " + repr(CoopNb) + " " + repr(InterNb) + " " + repr(WasteNb) + " " + repr(MaxValueGameParameter) + " " + repr(InputComplexity) + " " + repr(NbOfInputPerParam) + " " + repr(NbOfConfoundingFeatures) + " " + repr(NonCooperationCost) + " " + repr(NbOfIterationsSeparatedSelection) + " " + repr(NbOfIterations) + " " + repr(PrintLag) + " " + repr(NbOfHiddenNeurons) + " " + repr(Extra) + " " + repr(ClassifierThreshold) + " " + repr(IsSigmoidOutput) + " " + repr(GameStructure))
			fileName.write(" " + repr(GradientMethod) + " " + repr(BatchSize) + " " +	repr(LearningRate))
			fileName.write('\n')
			fileName.write('"Iteration" "SELFTrueNegativesR" "SELFFalsePositivesR" "COOPTruePositivesR" "COOPFalseNegativesR" "INTERTruePositivesR" "INTERFalsePositivesR" "INTERTrueNegativesR" "INTERFalseNegativesR" "WASTETrueNegativesR" "WASTEFalsePositivesR" "SELFTruePositivesI" "SELFFalseNegativesI" "COOPTrueNegativesI" "COOPFalsePositivesI" "COOPTruePositivesI" "COOPFalseNegativesI" "INTERTruePositivesI" "INTERFalseNegativesI" "WASTETrueNegativesI" "WASTEFalsePositivesI" "WASTETruePositivesI" "WASTEFalseNegativesI"')
			fileName.write('\n')


	def create_heatmap_file(self, name):
		with open(name, "wb") as fileName:
			fileName.write('"SelfishNb" "CooperativeNb" "InterdependentNb" "WastefulNb" "MaxValueGameParameter" "InputComplexity" "NbOfInputPerParam" "NbOfConfoundingFeatures" "NonCooperationCost" "NbOfIterationsSeparatedSelection" "NbOfIterations" "PrintLag" "NbOfHiddenNeurons" "Extra" "ClassifierThreshold" "IsSigmoidOutput" "GameStructure"')
			fileName.write(' "GradientMethod" "BatchSize" "LearningRate"')
			fileName.write('\n')

			fileName.write(repr(SelfishNb) + " " + repr(CoopNb) + " " + repr(InterNb) + " " + repr(WasteNb) + " " + repr(MaxValueGameParameter) + " " + repr(InputComplexity) + " " + repr(NbOfInputPerParam) + " " + repr(NbOfConfoundingFeatures) + " " + repr(NonCooperationCost) + " " + repr(NbOfIterationsSeparatedSelection) + " " + repr(NbOfIterations) + " " + repr(PrintLag) + " " + repr(NbOfHiddenNeurons) + " " + repr(Extra) + " " + repr(ClassifierThreshold) + " " + repr(IsSigmoidOutput) + " " + repr(GameStructure))
			fileName.write(" " + repr(GradientMethod) + " " + repr(BatchSize) + " " +	repr(LearningRate))
			fileName.write('\n')
			fileName.write('"Iteration" "GameIndex" "Type" "Cost" "Benefit" "AutomaticReward" "CommonInterest" "Reward" "Rewarded" "Test" "ErrorRecipient" "ErrorInvestor"')
			fileName.write('\n')


	def update_heatmap_file(self, iteration, name):
		with open(name, "a") as fileName:
			for opp in self.trainingSet:
				typeOfOpp = opp.type

				if typeOfOpp == 2:
					listCommonInterest = [False, True]
					listWrite = range(2)
				else:
					listCommonInterest = [False, True]
					listWrite = range(1)

				for i in listWrite:
					if typeOfOpp == 2:
						opp.setCommonInterest(listCommonInterest[i])

					fileName.write(repr(iteration) + " " + repr(opp.index) + " " + repr(typeOfOpp) + " " + repr(opp.c) + " " + repr(opp.b) + " " + repr(opp.ra) + " " + repr(str(opp.commonInterest)))
					outputR = self.rewarding([opp])
					fileName.write(" " + repr(outputR[0][0]))
					rewarded = outputR + opp.ra - opp.c >= 0
					fileName.write(" " + repr(int(rewarded[0][0])))

					if GameStructure == "RecipientAlone":
						test = (typeOfOpp == 0 or typeOfOpp == 2)	#Investor should test in Selfish and Interdependent (because c - ra < 0, at least at some occasion)
						test = int(test)
					elif GameStructure == "InvestorDarwinianDemon":
						test = (typeOfOpp == 0 or typeOfOpp == 2 or ((typeOfOpp == 1 or typeOfOpp == 3) and rewarded == 1))	#Investor should test in Selfish and Interdependent and [Coop or Waste if Recipient rewards]
						test = int(test)
					elif GameStructure == "CoEvolution":
						test = self.investing([opp])
						test = test[0][0]
					fileName.write(" " + repr(test))
					targetReward = (typeOfOpp == 1 or typeOfOpp == 2)	#The Recipient should reward for Inter (when ra < c) and Coop if I tests
					targetReward = targetReward * (opp.c - opp.ra + Extra)
					errorR = math.pow(outputR - targetReward, 2)
					fileName.write(" " + repr(errorR))
					targetTest = (rewarded or typeOfOpp == 2)
					errorI = math.pow(test - targetTest, 2)
					fileName.write(" " + repr(errorI))
					fileName.write('\n')



	def update_metrics_file(self, iteration, name):
		with open(name, "a") as fileName:
			fileName.write(repr(iteration))
			for metrics in self.metricsRecipient:
				fileName.write(" " + wr(metrics))
			for metrics in self.metricsInvestor:
				fileName.write(" " + wr(metrics))
			fileName.write('\n')


	def create_totalSet_file(self, name):
		with open(name, "wb") as fileName:
			fileName.write('"TypeOfSet" "GameIndex" "Type" "Cost" "Benefit" "AutomaticReward" "CommonInterest" "Reward" "Rewarded" "Test" "ErrorRecipient" "ErrorInvestor"')
			fileName.write('\n')

			totalSet = np.concatenate((self.trainingSet, self.testSet))
			for opp in totalSet:
				typeOfOpp = opp.type

				if typeOfOpp == 2:
					listCommonInterest = [False, True]
					listWrite = range(2)
				else:
					listCommonInterest = [False, True]
					listWrite = range(1)

				for i in listWrite:
					if typeOfOpp == 2:
						opp.setCommonInterest(listCommonInterest[i])

					fileName.write(repr(opp.typeOfSet) + " " + repr(opp.index) + " " + repr(typeOfOpp) + " " + repr(opp.c) + " " + repr(opp.b) + " " + repr(opp.ra) + " " + repr(str(opp.commonInterest)))
					outputR = self.rewarding([opp])
					fileName.write(" " + repr(outputR[0][0]))
					rewarded = outputR + opp.ra - opp.c >= 0
					rewarded = rewarded[0][0]
					fileName.write(" " + repr(int(rewarded)))

					if GameStructure == "RecipientAlone":
						test = (typeOfOpp == 0 or typeOfOpp == 2)	#Investor should test in Selfish and Interdependent (because c - ra < 0, at least at some occasion)
						test = int(test)
					elif GameStructure == "InvestorDarwinianDemon":
						test = ((typeOfOpp == 0 or typeOfOpp == 2) or ((typeOfOpp == 1 or typeOfOpp == 3) and rewarded == 1))	#Investor should test in Selfish and Interdependent and [Coop or Waste if Recipient rewards]
						test = int(test)
					elif GameStructure == "CoEvolution":
						test = self.investing([opp])
						test = test[0][0]
					fileName.write(" " + repr(test))

					targetReward = (typeOfOpp == 1 or typeOfOpp == 2)	#The Recipient should reward for Interdependent (when ra < c) and Coop if I tests
					targetReward = targetReward * (opp.c - opp.ra + Extra)
					errorR = math.pow(outputR - targetReward, 2)
					fileName.write(" " + repr(errorR))

					targetTest = (rewarded or typeOfOpp == 2)
					errorI = math.pow(test - targetTest, 2)
					fileName.write(" " + repr(errorI))

					fileName.write('\n')
