from sklearn import tree

'''
returns the predicted class of the test samples
'''
def dtree(tr, tr_class, te):
    model = tree.DecisionTreeClassifier()
    model.fit(tr,tr_class)
    pred = model.predict(te)
    return pred
