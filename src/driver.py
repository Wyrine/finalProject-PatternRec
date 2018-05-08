#!/usr/local/bin/python3
import sys
import numpy as np
from mppValidate import MPP_Validate as mppv
from knnValidate import kNN_Validate as knnv
from bpnnValidate import bpnn_Validate as bpnnv
from dtreeValidate import dtree_Validate as dtreev
from multiprocessing import Process as pr
from fld import fld
from pca import pca
import evaluation as ev

STEP = 0.01

def varyMPP(case, trans):
		for i in range(1, 4):
				prior = [STEP, 1 - STEP]
				roc = []
				while prior[0] < 1:
						roc.append(mppv(sys.argv[1], sys.argv[2], 23, i, prior, trans))
						prior[0] = STEP+prior[0]
						prior[1] -= STEP
				s = ""
				if trans is not None:
						s = "_FLD" if trans == fld else "_PCA"
				ev.toCSV("MPP" + str(i) + s, roc)


def main():
		if len(sys.argv) < 3:
				print("Usage: ./driver dataFilePath foldsFilePath")
				return 1
		m = pr(target=varyMPP, args=(2, None))
		m.start()
		b = pr(target=bpnnv, args=(sys.argv[1], sys.argv[2], 23, None))
		b.start()
		knn = pr(target=knnv, args=(sys.argv[1], sys.argv[2], 23))
		knn.start()
		#TODO: dtree can be run in this thread rather than spawning a thread
		d = pr(target=dtreev, args=(sys.argv[1], sys.argv[2], 23, None))
		d.start()

		m.join()
		b.join()
		d.join()
		knn.join()

if __name__ == "__main__":
		sys.exit(main())
