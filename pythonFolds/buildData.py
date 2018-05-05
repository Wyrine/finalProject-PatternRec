#!/usr/local/bin/python3

import numpy as np
import sys

def buildData(fName):
		""" reads the data file and return the data set and it's classes """
		with open(fName) as fin:
				fin.readline()
				rv = []
				for line in fin:
						try:
								rv.append([float(i) for i in line.replace("\n", "").split(",")])
						except:
								pass
		rv = np.array(rv)
		return rv[:, :-1], rv[:, -1]
