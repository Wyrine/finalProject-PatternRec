#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
import bpnn
from mpp import MPP as mpp
from dtree import dtree as tree
import validation as vd
import evaluation as ev
from fld import fld
from pca import pca
from fusion import *

def bpnn_mpp_fusion(dataName, grpName, folds, trans = None): 
	""" 
		params: 
			dataName := file with the data set
			grpName := file with the different groupings
			folds := number of folds
			trans := transformation function to be applied on the data set
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

		pred0 = bpnn.nn(trainSet, testSet, trainLabels)
		pred1 = mpp(trainSet, testSet, trainLabels,2)
		pred = bind(pred0,pred1,testLabels)
		results.append((np.array(pred).astype(np.int), testLabels))
	results = ev.buildConfusionMatrices(results)    
	results = ev.normalizeConfMat(results)
	results = ev.getAvgProbMatrix(results)
	results = ev.rocData(results)
	print("bpnn_mpp2_fusion Accuracy: %f" % (results["Acc"]))
	return results  

bpnn_mpp_fusion("../data/EEG_dropcat.csv", "../data/folds.grp", 23) 
