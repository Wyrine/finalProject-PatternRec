#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
import kNN as knn
import validation as vd

def kNN_Validate(dataName, grpName, folds, k = 3, d = 2):
		"""
				params: dataName := file with the data set
								grpName := file with the different groupings
								folds := number of folds
								k := number of neigbors to base the classification off of
												where the default is 3
								d := the minkowski distance to use, default is 2
				objective: performs cross validation using kNN as classifier
				returns: a list of tuples organized as (test_predicted, test_groundTruth)

		"""
		valid = vd.Validate(grpName, folds)
		data, labels = bd(dataName)
		results = [] #stores tuples: (list_predicted, list_groundTruth)
		for i in range(valid.getFoldCount()):
				#get the train and test indices of the data set
				testIndex, trainIndex = valid.getTest(i), valid.getTrain(i)
				#build the test set and test labels
				testSet, testLabels = data[testIndex, :], labels[testIndex]
				#build the train set and training labels
				trainSet, trainLabels = data[trainIndex, :], labels[trainIndex]
				#standardize the training and test set
				trainSet, testSet = standard(trainSet, testSet)
				#classify test set and add it to the results list
				results.append((knn.kNN(trainSet, testSet, trainLabels, k, d), testLabels))
		return results	

kNN_Validate("../data/EEG_dropcat.csv", "../data/folds.grp", 23)
