#!/usr/local/bin/python3
import numpy as np

def pca(tr, tol = .1):
	'''
		returns the number of eigenvectors to drop, given tolerance
	'''
	c = np.cov(tr, rowvar=False)
	#presumably, the eigs are sorted in descending order
	eigs,vec = np.linalg.eig(c)
	divisor = np.sum(eigs)
	drop, accum = 0, 0
	#iterate through the eigenvalues in reverse order
	for eig in eigs[::-1]:
		accum += eig
		if (accum / divisor) > tol :
			break
		drop += 1

	if drop != 0:
		#drop the eigs that are of least importance
		vec = vec[:drop, :]
	return vec
