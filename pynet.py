# TODO
# 	put in assertions that input data is numpy.array type?
# testing:
#	that it actually works
#	gradient calculations
#		tbh if these are wrong I will probably give up and switch to numerical estimate
#	

import numpy as np
import math

class NeuralNetClassifier:
	# this class just holds all the helper functions for the NeuralNetClassifier,
	# for sake of readability

# probably have:
# Z, W, a, bprop, logisticPrime, delta

class NeuralNetClassifier:
	def __init__(self, featureset, classifications, nodesInHiddenLayers, learningrate):
		# TODO Add error handling: different number of examples and classes

		# initialise data and hyperparameters
		maxes = np.amax(np.array(featureset), axis=0)
		self.data = [np.divide(row, m) for row, m in zip(featureset, maxes)]
		self.eta = learningrate
		self.numLayers = len(nodesInHiddenLayers)

		# Stuff to do with classifications
		self.classifications = classifications
		self.classes = {} # dict that maps classification to 1-of-k vector
		for i, classification in enumerate(list(set(classifications))):
			temp = []
			[temp.append(0) for j in range(len(list(set(classifications))))]
			temp[i] = 1
			self.classes[classification] = temp

		# biases and weights
		self.weights = []
		self.biases = []
		for i in nodesInHiddenLayers:
			self.weights.append(np.random.randn(len(featureset[0]), i))
			self.biases.append(np.random.randn(i))

	def update(self):
		accumulator = self.X
		for bias, weight in zip(self.biases, self.weights):
			multiplied = np.matmul(accumulator, weight)
			# since bias is only one row but accumulator may be many rows.
			multiplied = np.array([row - bias for row in multiplied])
			accumulator = self.logistic(multiplied)

		self.result = accumulator
		return accumulator

	def classifiy(self, features):
		# TODO Add error handling: feature vector wrong size
			# Python deafult exception throwing might be all good
		# maybe:
			# Take mean value and std.dev of features from training.
			# if any of features are more than 3 std.dev away present warning about "strange values"
		self.X = features
		predictvec = np.round(self.update())
		classifications = []
		for prediction in predictvec:
			noclass = True
			for classification in self.classes:
				if prediction == self.classes[classification]:
					classifications.append(self.classes[classification])
					noclass = False
					break

			if noclass:
				classifications.append(None)

		return classifications

	def bprop(self):
		dEdw = [np.matmul(self.a(n).T, self.delta(n+1)) for n in range(1, len(self.weights) + 1)]
		[wn - (self.eta*dEdwn) for wn, dEdwn in zip(self.weights, dEdw)]

	def train(self):
		# TODO automatic stop bpropping once training losses diverge from testing losses
		for i in range(2000):
			self.bprop()

	def test(self, testdata, classifications):
		predictions = self.classify(testdata)
		numcorrect = 0
		for prediction, classification in zip(predictions, classifications):
			if prediction == classification:
				numcorrect = numcorrect + 1

		self.trainingloss = 1 - (numcorrect/len(classifications))
		return float(numcorrect)/len(classifications)

	# vector difference between classification and result
	def error(classifications):
		# TODO Add error handling: classification not in dict
		self.currenterror = []
		[self.currenterror.append(np.array([self.result[i] - self.classes[classification] for i, classification in enumerate(classifications)]))]
		return self.currenterror

	# Mean squared-error over error vector
	def MSE(classifications):
		e = self.error(classifications)
		return np.array([sum(result) * (1 / len(result)) for result in e])

	# Root mean squared-error over error vector
	def RMSe(classifications):
		return np.sqrt(MSE(classifications))
		
	def logistic(x):
		return 1 / (1 + np.exp(-x))

	def logisticPrime(x):
		return np.exp(-x) / ((1 + np.exp(-x))**2)

	def delta(k):
		if k == self.numLayers:
			return np.multiply(self.currenterror, logisticPrime(self.Z(k)))

		return np.multiply(np.matmul(self.delta(k+1), self.W(k).T), logisticPrime(self.Z(k)))

	def W(k):
		return self.weights[k-1]

	def Z(k):
		if k == 1:
			return self.X

		return np.multiply(self.logistic(self.Z(k-1)), self.W(n-1))

	def a(k):
		return self.logistic(Z(k))
