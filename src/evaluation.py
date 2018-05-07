#!/usr/local/bin/python3
import numpy as np

def rocData(conMat):
		"""
				returns a dictionary built from the normalized confusion matrix
				with keys as follows:
						TN, TP, FP, FN, Acc, Sens, Spec, and Prec
		"""
		rv = {}
		tp, tn, fp, fn = conMat[1,1], conMat[0,0], conMat[0,1], conMat[1,0]
		rv["TN"] =  tn
		rv["TP"] = tp
		rv["FP"] = fp 
		rv["FN"] = fn
		rv["Acc"] = (tp+tn)/(tp+tn+fp+fn)
		rv["Sens"] = tp / (tp+fn)
		rv["Spec"] = tn / (tn+fp)
		rv["Prec"] = tp / (tp+fp)
		return rv
		
def getAvgProbMatrix(normMat):
		"""
				Iterates through the normalized matrices and computes one avg
				matrix and return that average
		"""
		rv = np.zeros((2,2), dtype=np.float64)
		for mat in normMat:
				rv += mat
		return (rv / 23)

def buildConfusionMatrices(results):
		"""
				goes through the results 
		"""
		rv = []
		for pred, ground in results:
				tmp = np.zeros((2,2), dtype = np.int)
				for x, y in zip(pred, ground):
						tmp[x, y] += 1
				rv.append(tmp)
		return rv

def normalizeConfMat(conMat):
		"""
				normalizes all confusion matrices in the conMat array and
				returns the new matrix array
		"""
		for i, mat in enumerate(conMat):
				divisor = 0
				for a in mat:
						divisor += np.sum(a)
				conMat[i] = conMat[i] /divisor
		return conMat
