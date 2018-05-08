#!/usr/local/bin/python3

import sys
import numpy as np

class Validate:
	def __init__(self, fname, m):
		"""	
			Stores all of the folds of the grouping file into a matrix
			of dimensions m x number of elements in each fold
		"""
		self.folds = None
		self.m = m
		i = 0
		with open(fname) as f:
			for line in f:
				tmp = line.replace("\n", "").split()
				if self.folds is None:
					self.folds = np.zeros( ( m, len(tmp)) )
					self.folds[i, :] = np.array(tmp).astype(np.int)
					i += 1
	def getTest(self, i):
		""" get the test fold set which is index i """
		return self.folds[i, :].astype(np.int).flatten()
	def getFoldCount(self): 
		return self.m
	def getTrain(self, i):
		""" get all of the other folds excluding index i """
		return np.delete(np.array(self.folds), i, 0).astype(np.int).flatten()

def main():
	valid = Validate("../data/folds.grp", 23)
	for i in range(valid.getFoldCount()):
		test, train = valid.getTest(i), valid.getTrain(i)
	return 0

if __name__ == "__main__":
	sys.exit(main())
