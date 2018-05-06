#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
from mpp import MPP
import validation as vd

def MPP_Validate(dataName, grpName, folds, case = 3):
		"""
				params: dataName := file with the data set
								grpName  := file with the different groupings
								folds		 := number of folds
								case		 := case of the discriminant function to use
														defaulted to case 3
				objective: performs cross validation using mpp as classifier
										with the discriminant function cases
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
				results.append((MPP(trainSet, testSet, trainLabels, case), testLabels))
		return results	
