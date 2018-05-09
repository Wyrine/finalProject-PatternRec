#!/usr/local/bin/python3
import evaluation as ev
import numpy as np
def fuse(C1, C2):
	'''
		returns the fusion choices based on confusion matrices of two classifiers
		***C1, C2 should be passed by value to prevent overwrite
	'''
	fuse = np.zeros(4).reshape([2,2])
	C1, C2 = C1.astype(np.float64), C2.astype(np.float64)
	#divide elements by row sum
	C1[0][0] = C1[0][0] / ( C1[0][0] + C1[0][1] )
	C1[0][1] = C1[0][1] / ( C1[0][0] + C1[0][1] )
	C1[1][0] = C1[1][0] / ( C1[1][0] + C1[1][1] )
	C1[1][1] = C1[1][1] / ( C1[1][0] + C1[1][1] )
	C2[0][0] = C2[0][0] / ( C2[0][0] + C2[0][1] )
	C2[0][1] = C2[0][1] / ( C2[0][0] + C2[0][1] )
	C2[1][0] = C2[1][0] / ( C2[1][0] + C2[1][1] )
	C2[1][1] = C2[1][1] / ( C2[1][0] + C2[1][1] )

	#c1 0 c2 0
	if (C1[0][0]*C2[0][0]) > (C1[1][0]*C2[1][0]):
		fuse[0][0] = 0
	else:
		fuse[0][0] = 1

	#c1 0 c2 1
	if (C1[0][0]*C2[0][1]) > (C1[1][0]*C2[1][1]):
		fuse[0][1] = 0
	else:
		fuse[0][1] = 1

	#c1 1 c2 0
	if (C1[0][1]*C2[0][0]) > (C1[1][1]*C2[1][0]):
		fuse[1][0] = 0
	else:
		fuse[1][0] = 1

	#c1 1 c2 1
	if (C1[0][1]*C2[0][1]) > (C1[1][1]*C2[1][1]):
		fuse[1][1] = 0
	else:
		fuse[1][1] = 1
	
	return fuse

def bind(pred0, pred1, test):
	'''
		params
			pred0: predictions from classifier 0
			pred1: predictions from classifier 1
			conf0: confusion matrix from classifier 0
			conf1: confusion matrix from classifier 1
		objective
			uses naive bayes fusion to fuse the results of two classifiers
		returns
			the fused predictions
	'''
	conf0 = ev.buildConfusionMatrices([(pred0,test)])[0]
	conf1 = ev.buildConfusionMatrices([(pred1,test)])[0]
	fuser = fuse(conf0,conf1)
	print(fuser)
	pred = []
	for x, y in zip(pred0, pred1):
		pred.append(fuser[x][y])
	return pred
