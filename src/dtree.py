#!/usr/local/bin/python3
from sklearn import tree

def dtree(tr, tr_class, te):
		'''
				returns the predicted class of the test samples
		'''
		model = tree.DecisionTreeClassifier()
		model.fit(tr,tr_class)
		pred = model.predict(te)
		return pred
