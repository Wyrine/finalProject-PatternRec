#!/usr/local/bin/python3
import numpy as np
import operator as op

def minkowski(tr, te, d):
		"""
				computes the minkowski distance between the tr and te
				samples and returns the distance
		"""
		rv = 0
		for a, b in zip(tr, te):
				rv += (abs(a-b)** d)
		return rv ** (1/d)

def kNN(train, test, trainLabels, k, d = 2):
		"""
				Parameters:
						train := training data set
						test := testing data set
						trainLabels := labels of the training set
						k := number of neighbors to vote on
						d := minkowski distance to use
				Objective:
						runs kNN on the test set to try and classify
						using the training set
				returns:
						the predicted class of the test set
		"""
		rv = []
		i, tot = 0, len(test)
		for te in test:
				print("Iteration: %d of %d" %(i, tot))
				rv.append(kNN_Classify(train, te, trainLabels, k, d))
				i += 1
		return rv
				
def kNN_Classify(train, te, trainLabels, k, d):
		"""
				classifies a test sample based on the k nearest neighbors
		"""
		kNeighbs = []
		#initialize the first k samples in the training set in the
		#training set
		for tr, c in zip(train[:k, :], trainLabels[:k]):
				kNeighbs.append((minkowski(tr, te, d), c))
		kNeighbs.sort(key = op.itemgetter(0))
		
		#iterate through the remaining training samples
		for tr, c in zip(train[k:, :], trainLabels[k:]):
				#get the distance between the current tr and te
				dist = minkowski(tr, te, d)
				if dist < kNeighbs[-1][0]:
						#update the largest distance with the new, smaller, distance
						kNeighbs[-1] = (dist, c)
						#sort the array
						kNeighbs.sort(key = op.itemgetter(0))
		return neighborVoting(kNeighbs)

def neighborVoting(kNeighbs):
		"""
				counts the number of class zeros and class ones and picks
				the one with the maximum. If there is a tie then class 0 is chosen
		"""
		counts = np.zeros(2, dtype=np.int)
		for _, i in kNeighbs:
				counts[i] += 1
		return 0 if counts[0] > counts[1] else 1
