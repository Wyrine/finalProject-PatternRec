#!/usr/local/bin/python3
import numpy as np
from standardize import standard
from buildData import buildData as bd
import bpnn 
import validation as vd

def bpnn_Validate(dataName, grpName, folds): 
                """ 
                                params: dataName := file with the data set
                                                                grpName := file with the different groupings
                                                                folds := number of folds
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
                                #standardize the training and test set
                                trainSet, testSet = standard(trainSet, testSet)
                                #classify test set and add it to the results list
                                results.append((bpnn.nn(trainSet, testSet, trainLabels), testLabels))
                                print(results[-1][0])
                return results  

bpnn_Validate("../data/EEG_dropcat.csv", "../data/folds.grp", 23) 
