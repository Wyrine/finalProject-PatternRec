from sklearn.neural_network import MLPClassifier

def nn(tr, te, tr_class):
		'''
				returns the predicted class of the test samples
		'''
		model = MLPClassifier(solver = 'adam', learning_rate = "adaptive", alpha = 1e-3, activation = 'relu', verbose = True)
		model.fit(tr,tr_class)
		pred = model.predict(te)
		return pred
