import sys



NbOfInputPerParam = 1	# Parameter NP in the article
InputComplexity = 1		# Additional complexity parameter, set to 1 for all results shown in the article
NbOfConfoundingFeatures = 10	# Parameter NS
Alpha = 0	#Importance of cooperation, should be in [0, 1]
InterNb = 16	# Number of Interdependent games in the training set

NonCooperationCost = 1 - Alpha
if InputComplexity == 0 and NbOfInputPerParam != 1:
	sys.exit("InputComplexity = 0 can work only with NbOfInputPerParam = 1")

SelfishNb = 500		#500
CoopNb = SelfishNb / 2
WasteNb = SelfishNb / 2
ProbabilityOfInterdependency = 0.5

GradientMethod = "MiniBatch"	# "Batch"	"MiniBatch"	"Stochastic"
if GradientMethod == "Batch":
	BatchSize = SelfishNb + CoopNb + InterNb + WasteNb
elif GradientMethod == "Stochastic":
	BatchSize = 1
else:
	BatchSize = 50	#50

# There are two players: the Investor annd the Recipient
GameStructure = "CoEvolution"	# "RecipientAlone"	"InvestorDarwinianDemon"	"CoEvolution"

ClassifierThreshold = 0.5
NbOfHiddenNeurons = 10
LearningRate = 0.2
MaxValueGameParameter = 1
IsSigmoidOutput = False

Extra = 0.05
NbOfIterationsSeparatedSelection = 1000 #Pre-selection phase
NbOfIterations = 10000 					#Selection phase
PrintLag = 500 							#The 2 metrics files are updated every PrintLag iterations

HeatMapFile = False		# For generating figure 5 in the article
