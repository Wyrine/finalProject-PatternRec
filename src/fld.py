#!/usr/local/bin/python3
import numpy as np

def fld(tr, tr_class):
	'''
		returns the FLD projection matrix
	'''
	mat0 = []
	mat1 = []
	for x,xClass in zip(tr, tr_class):
		if xClass == 0:
			mat0.append(np.array(x))
		else:
			mat1.append(np.array(x))

	#compute scatter
	
	sInv = ((len(mat0)-1)  * np.cov(mat0, rowvar=False)) + ((len(mat1)-1) * np.cov(mat1, rowvar=False))
	#print(sInv)
	sInv = np.linalg.inv(sInv)

	return np.matmul(sInv, (np.mean(mat0,axis=0) - np.mean(mat1,axis=0)))
