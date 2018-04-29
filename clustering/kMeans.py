#!/usr/local/bin/python3
import sys
import numpy as np
from random import *
import readPPM as rpp

def euc(A, B, n = 3):
		dist = 0
		for i in range(n):
				dist += (A[i] - B[i]) ** 2
		return dist ** (1/2)

def updateClusters(ppm, mappings, means):
		means.fill(0)
		counts = np.zeros(len(means), dtype=np.int)
		
		for i in range(len(ppm)):
				for j in range(len(ppm[0])):
						cl, dist = mappings[i,j]
						counts[int(cl)] += 1
						means[int(cl)] += ppm[i,j]
		for i in range(len(counts)):
				if counts[i] != 0:
						means[i] /= counts[i]

		return means

def kMeans(ppm, k, dist = euc):
		means = np.random.uniform(0,255,k*3).reshape([k,3])

		w, h = len(ppm[0]), len(ppm)
		#h*w matrix with each element being [cluster, distance]
		mappings = np.zeros((h,w, 2), dtype=np.float)
		for i in range(h):
				for j in range(w):
						#initialize all pixels to be part of cluster 0
						mappings[i,j] = [0, dist(ppm[i,j], means[0])]
		iteration = 0
		changed = True
		while changed and iteration < 50:
				iteration += 1
				print("iteration:", iteration, file=sys.stderr)
				changed = False
				#map all the pixels to a clusters and if any single one changes then changed = True
				for i in range(h):
						for j in range(w): #each pixel
								for t in range(k): #each cluster
										#compute the distance
										newDist = dist(ppm[i,j], means[t])
										#if strictly less than
										if newDist < mappings[i,j, 1]:
												#a class changed
												changed = True
												#update weight and which cluster this pixel belongs to
												mappings[i, j] = [t, newDist]
				#compute the mean of each cluster based off of the pixels associated with it
				if changed: means = updateClusters(ppm, mappings, means)
		return mappings, means	


def main():
		ppm, mI = rpp.readImage()
		for k in [2, 8, 16, 32, 64, 128, 256]:
				mappings, clusters = kMeans(ppm, k)
				with open('kMeans_' + str(k) + ".ppm", 'w') as f:
						rpp.writeImage(mappings, mI, clusters, f)
				print("Done with k=",k, file=sys.stderr)
		return 0

if __name__ == "__main__":
		sys.exit(main())
