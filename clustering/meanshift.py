#!/usr/local/bin/python3

import sys
from sklearn.cluster as mean_shift
import readPPM as rpp

ppm, mI = rpp.readImage()
w, h = len(ppm[0]), len(ppm)
p2 = ppm.reshape([h*w, 3])
for bw in [0.1, 0.5, 1, 2, 3, 6, 10, 20, 30,50, 70]:
		a, b = mean_shift(p2, bandwidth=bw, n_jobs = -1)
		with open("ms_"+str(bw)+".ppm", 'w') as f:
				print("P3", file=f)
				print(w,h, file=f)
				print(mI, file = f)
				for i in b:
						for var in a[i]:
								print(int(var), file=f, end=" ")
						print(file=f)
