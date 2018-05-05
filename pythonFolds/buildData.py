#!/usr/local/bin/python3

import numpy as np
import sys

def buildData(fName):
		with open(fName) as fin:
				fin.readline()
				rv = []
				for line in fin:
						try:
								rv.append([float(i) for i in line.replace("\n", "").split(",")])
						except:
								print(line)
		return np.array(rv)
