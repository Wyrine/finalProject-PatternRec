#!/usr/local/bin/python3
import numpy as np
import math

def MPP(tr, te, trLabels, case, priors = None):
	""" takes the training set, testing set,
		training labels, case number {1,2,3},
		and optional prior probabilities and
		predicts which class the data set should belong to
	"""
	feats = len(tr[0, :])
	classes = [ [], [] ]
	#splitting up the two class training sets
	for row, c in zip(tr, trLabels):
		classes[c].append(row)
	means, sigs = [], []
	#calculating mean and covariance matrices for each class
	for c in classes:
		means.append(np.mean(c, axis=0))
		sigs.append(np.cov(c, rowvar=False))
	#if the prior was not provided, calculate it based on training set
	if priors is None:
		priors = [len(c)/len(tr) for c in classes]

	#return value list
	rv = []
	mm = np.matmul

	#case 1 builds covariance based on the sum of traces
	#and then averages them by dividing by 2 * number of features
	#after this if statement sigs is updated to reflect the changes
	if case == 1:
		sig = np.sum([np.trace(s) for s in sigs])/(2*feats)
		sigs = [sig*np.identity(feats) for _ in sigs]
	#case 2 sums the covariance matrices and averages them and uses them
	#as the covariance for all classes
	elif case == 2:
		sig = sigs[0] + sigs[1]
		sigs = [sig/2 for _ in sigs]

	#computing inverse sigma for all covariance
	#as well as the determinants
	sigInv, dets = [], []
	for s in sigs:
		if s.shape == ():
			sigInv.append(np.array(1/s))
			dets.append(np.array(s))
		else:
			sigInv.append(np.linalg.inv(s))
			dets.append(np.linalg.det(s))

	#1/(2*pi)
	piConst = 1/(2 * math.pi)
	#iterate through the test samples
	for samp in te:
		#initializing the posterior
		post, choice = 0, 0
		#getting the mean, determinant, covInverse, and class for each loop
		for pri, mu, det, sI, i in zip(priors, means, dets, sigInv, [0,1]):
			#x - mu
			dist = samp - mu
			#tmp is posterior probability for this class
			tmp = piConst * 1 / np.sqrt(det) * pri
			if sI.shape != ():
				tmp *= math.exp(-mm(mm(dist.transpose(), sI), dist) / 2)
			else:
				tmp *= math.exp(-(dist* sI * dist) / 2)
				
			#tmp = pri * piConst * (1/ (det**0.5)) * math.exp(-0.5* mm(mm(dist.transpose(), sI),dist))
			#if this class is the maximum, update choice and post
			if tmp > post: choice, post = i, tmp
		#add the choice to the return values
		rv.append(choice)
	return rv
