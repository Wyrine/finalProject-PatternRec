#!/usr/local/bin/python3
import sys
import numpy as np
from mppValidate import MPP_Validate as mppv
from knnValidate import kNN_Validate as knnv
from bpnnValidate import bpnn_Validate as bpnnv
from dtreeValidate import dtree_Validate as dtreev
#from kmValidate import km_Validate as kmv
from multiprocessing import Process as pr
from fld import fld
from pca import pca
import evaluation as ev

STEP = 0.001

def varyMPP(trans):
	for case in range(1, 4):
		prior = [STEP, 1 - STEP]
		roc = []
		while prior[0] < 1:
			roc.append(mppv(sys.argv[1], sys.argv[2], 23, case, prior, trans))
			prior[0] = STEP+prior[0]
			prior[1] -= STEP
			s = ""
			if trans is not None:
				s = "_FLD" if trans == fld else "_PCA"
			ev.toCSV("MPP" + str(case) + s, roc)

def varykNN():
	roc = []
	for k in range(1, 27, 2):
		for d in [1,2,3,10,100, 1000]:
			roc.append(knnv(sys.argv[1], sys.argv[2], 23,k,d))
	ev.toCSV("KNN", roc)


def main():
	if len(sys.argv) < 3:
		print("Usage: ./driver dataFilePath foldsFilePath")
		return 1
	#MPP
	m = pr(target=varyMPP, args=(None,))
	m.start()

	#bpnn
	b = pr(target=bpnnv, args=(sys.argv[1], sys.argv[2], 23, None))
	b.start()

	#knn
	knn = pr(target=varykNN, args=())
	knn.start()

	#dtree
	d = pr(target=dtreev, args=(sys.argv[1], sys.argv[2], 23, None))
	d.start()

	m.join()
	b.join()
	d.join()
	knn.join()

if __name__ == "__main__":
	sys.exit(main())
