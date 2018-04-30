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

def winnerTakeAll(ppm, k, eps, dist = euc):
		means = np.random.uniform(0, 255, k*3).reshape([k,3])

		w, h = len(ppm[0]), len(ppm)
		#h*w matrix with each element being [cluster, distance]
		mappings = np.zeros((h,w, 2), dtype=np.float)
		mappings.fill(float("inf"))

		iteration = 0
		changed = True
		while changed and iteration < 50:
				iteration += 1
				print("Starting iteration:", iteration, file=sys.stderr)
				changed = False
				#map all the pixels to a clusters and if any single one changes then changed = True
				for i in range(h):
						for j in range(w): #each pixel
								winner = mappings[i,j,0]
								bestDist = mappings[i,j,1]
								for t in range(k): #each cluster
										#compute the distance
										newDist = dist(ppm[i,j], means[t])
										#if strictly less than
										if newDist < bestDist:
												#a class changed
												changed = True
												bestDist = newDist
												winner = t
								if winner != mappings[i,j,0]:
										winner = int(winner)
										means[winner] = means[winner] + eps * (ppm[i,j] - means[winner])
										mappings[i,j,0] = winner
										mappings[i,j,1] = bestDist
				if changed: means = updateClusters(ppm, mappings, means)
		return mappings, means	

def main():
		ppm, mI = rpp.readImage()
		for k in [2, 8, 16, 32, 64, 128, 256]:
				mappings, clusters = winnerTakeAll(ppm, k, 0.1)
				with open('wta_'+str(k)+'.ppm', 'w') as f:
						rpp.writeImage(mappings, mI, clusters, f)
				print("Done with k=",k, file=sys.stderr)
		return 0

if __name__ == "__main__":
		sys.exit(main())
