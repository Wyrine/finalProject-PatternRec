#!/usr/local/bin/python3

from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np

def randForest(tr, te, trClass):
	model = rfc(criterion='entropy')
	model.fit(tr,trClass)
	return model.predict(te)
