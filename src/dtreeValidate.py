#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
import dtree 
import validation as vd
import evaluation as ev

def dtree_Validate(dataName, grpName, folds, trans = None): 
	""" 
			params: 
					dataName := file with the data set
					grpName := file with the different groupings
					folds := number of folds
					trans := tranformation function to be applied to the data set
			objective: performs cross validation using neural net as classifier
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
			#if the data is to be transformed
			if trans is not None:
					tmp = trans(trainSet).transpose()
					trainSet = np.matmul(trainSet, tmp)
					testSet = np.matmul(testSet, tmp)
			#standardize the training and test set
			trainSet, testSet = standard(trainSet, testSet)
			#classify test set and add it to the results list
			results.append((dtree.dtree(trainSet, testSet, trainLabels), testLabels))
	results = ev.buildConfusionMatrices(results)    
	results = ev.normalizeConfMat(results)
	results = ev.getAvgProbMatrix(results)
	results = ev.rocData(results)
	print("dtree Accuracy: %f" % results["Acc"])
	return results  

#dtree_Validate("../data/EEG_dropcat.csv", "../data/folds.grp", 23) 
