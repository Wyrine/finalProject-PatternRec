#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == "__main__":
		if len(sys.argv) < 2:
				print("Usage: ./roc.py file1.csv file2.csv ...")
				sys.exit(1)
		mList = []
		for i in range(1, len(sys.argv)):
				df = pd.read_csv(sys.argv[i])
				tpr = np.array(df["Sensitivity(Recall)"])
				fp, tn = np.array(df["FP"]), np.array(df["TN"])
				ind = sys.argv[i].find("/")
				classifier = sys.argv[i][ind+1:].replace(".csv", "")

				fpr = np.zeros(len(fp), dtype=np.float64)
				for j in range(len(fp)):
						fpr[j] = fp[j] / (fp[j] + tn[j])
				mList.append(plt.plot(fpr, tpr, label=classifier))
		plt.legend(fontsize='medium')
		plt.title("ROC Curves")
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.grid()
		plt.show()
