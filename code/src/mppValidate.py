#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
from mpp import MPP
import validation as vd
import evaluation as ev
from pca import pca
from fld import fld

def MPP_Validate(dataName, grpName, folds, case = 3, priors = None, trans = None):
	"""
		params: dataName := file with the data set
				grpName  := file with the different groupings
				folds		 := number of folds
				trans := transformation function to do dimensionality reduction
				case		 := case of the discriminant function to use
							defaulted to case 3
				priors := the prior probabilities for the two classes. Defaulted to None
		objective: performs cross validation using mpp as classifier
				with the discriminant function cases
		returns: a dictionary with performance evaluation data 
	"""
	valid = vd.Validate(grpName, folds)
	data, labels = bd(dataName)
	results = [] #stores tuples: (list_predicted, list_groundTruth)
	for i in range(valid.getFoldCount()):
		print("Iteration %d" % i)
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
				trainSet = np.matmul(trainSet, tmp).reshape(-1,1).astype(np.float64)
				testSet = np.matmul(testSet, tmp).reshape(-1,1).astype(np.float64)
			else:
				tmp = trans(trainSet).transpose()
				trainSet = np.matmul(trainSet, tmp)
				testSet = np.matmul(testSet, tmp)
		#standardize the training and test set
		trainSet, testSet = standard(trainSet, testSet)
		#classify test set and add it to the results list
		results.append((MPP(trainSet, testSet, trainLabels, case, priors), testLabels))
	
	results = ev.buildConfusionMatrices(results)	
	results = ev.normalizeConfMat(results)
	results = ev.getAvgProbMatrix(results)
	print(results)
	results = ev.rocData(results)
	print("Case %i Accuracy: %f" % (case, results["Acc"]))
	return results	
		
if __name__ == "__main__":
	MPP_Validate("../data/EEG_dropcat.csv", "../data/folds.grp", 23, 1)
