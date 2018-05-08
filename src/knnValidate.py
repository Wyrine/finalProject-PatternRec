#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
import kNN as knn
import validation as vd
import evaluation as ev
from fld import fld
from pca import pca

def kNN_Validate(dataName, grpName, folds, k = 3, d = 2, trans = None):
	"""
		params: dataName := file with the data set
			grpName := file with the different groupings
			folds := number of folds
			k := number of neigbors to base the classification off of
							where the default is 3
			d := the minkowski distance to use, default is 2
			trans := transformation function to be applied on data set
		objective: performs cross validation using kNN as classifier
				eturns: a list of tuples organized as (test_predicted, test_groundTruth)

	"""
	valid = vd.Validate(grpName, folds)
	data, labels = bd(dataName)
	results = [] #stores tuples: (list_predicted, list_groundTruth)
	for i in range(valid.getFoldCount()):
		print("kNN iteration %d" % i)
		#get the train and test indices of the data set
		testIndex, trainIndex = valid.getTest(i), valid.getTrain(i)
		#build the test set and test labels
		testSet, testLabels = data[testIndex, :], labels[testIndex]
		#build the train set and training labels
		trainSet, trainLabels = data[trainIndex, :], labels[trainIndex]
		#if the data is to be transformed
		if trans is not None:
			if trans is fld:
				tmp = trans(trainSet, trainLabels)
				trainSet = np.matmul(trainSet, tmp)
				trainSet = trainSet.reshape(-1,1).astype(np.float64)
				testSet = np.matmul(testSet, tmp)
				testSet = testSet.reshape(-1,1).astype(np.float64)
			else:
				tmp = trans(trainSet).transpose()
				trainSet = np.matmul(trainSet, tmp)
				testSet = np.matmul(testSet, tmp)
		#standardize the training and test set
		trainSet, testSet = standard(trainSet, testSet)
		#classify test set and add it to the results list
		results.append((knn.kNN(trainSet, testSet, trainLabels, k, d), testLabels))
	results = ev.buildConfusionMatrices(results)	
	results = ev.normalizeConfMat(results)
	results = ev.getAvgProbMatrix(results)
	print("knn results", results)
	results = ev.rocData(results)
	print("%d-NN Accuracy: %f" % (k, results["Acc"]))
	return results	

kNN_Validate("../data/EEG_dropcat.csv", "../data/folds.grp", 23, 23, 2, fld)
