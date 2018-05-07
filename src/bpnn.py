#!/usr/local/bin/python3
from sklearn.neural_network import MLPClassifier

def nn(tr, te, tr_class):
		'''
				returns the predicted class of the test samples
		'''
		model = MLPClassifier(solver = 'sgd', alpha = 1e-3, activation = 'logistic', verbose = True)
		model.fit(tr,tr_class)
		pred = model.predict(te)
		return pred
