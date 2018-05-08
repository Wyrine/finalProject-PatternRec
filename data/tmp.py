#!/usr/local/bin/python3

from buildData import buildData as bd
import numpy as np



def pca(tr, tol = .1):
		'''
				returns the number of eigenvectors to drop, given tolerance
		'''
		c = np.cov(tr, rowvar=False)
		#presumably, the eigs are sorted in descending order
		eigs,vec = np.linalg.eig(c)
		divisor = np.sum(eigs)
		vec = vec[:3, :]
		return vec


data, labels = bd("../data/EEG_dropcat.csv")
mat = np.matmul(data, pca(data).transpose())

for r, c in zip(mat, labels):
		for v in r:
				print(str(v), end=",")
		print(c)
