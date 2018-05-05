#!/usr/local/bin/python3
from sys import argv, exit
import numpy as np
from standardize import standard
from buildData import buildData as bd
from mpp import MPP
import validation as vd

def main():
		valid = vd.Validate("../data/folds.grp", 23)
		data, labels = bd("../data/EEG_dropcat.csv")
		results = []
		for i in range(valid.getFoldCount()):
				testIndex, trainIndex = valid.getTest(i), valid.getTrain(i)
				testSet, testLabels = data[testIndex, :], labels[testIndex]
				trainSet, trainLabels = data[trainIndex], labels[trainIndex]
				trainSet, testSet = standard(trainSet, testSet)
				results.append(MPP(trainSet, testSet, trainLabels, 3))
				print(results[-1])
				exit(1)
				
if __name__ == "__main__":
		exit(main())
