import sys
import numpy as np

def euc(A, B, n = 3):
		dist = 0
		for i in range(n):
				dist += (A[i] - B[i]) ** 2
		return dist ** (1/2)

def kmeans(k=2, dat, col):
    '''
    params
        k: the number of clusters
        dat: dataset
        col: the number of features
    objective
        cluster in to k seperate clusters
    returns
        the membership of each the samples passed in
    '''
    #assume standardized values
    means = np.random.uniform(-2,2,k*col) #cluster centers
    membership = np.zeros(len(dat))

    for i in range(0, 50):
	#for each point, assign to nearest cluster
	print 'iter: ', i
	for j in range(0, len(dat)):
	    mindex = -1
	    min = sys.maxint
	    #find closest mean
	    for m in range(len(means)): #each mean
		dist = euc(dat[j],means[m],col)
		if dist < min:
		    #changed = True
		    min = dist
		    mindex = m
	    membership[j] = mindex

	#find new cluster centers
	for m in range(len(means)):
	    tmp = np.zeros(col)
	    count = 0.0
	    for j in range(len(membership)):
		if membership[j] == m:
		    count += 1
		    tmp += dat[j]
	    if count != 0: #if no members in cluster, pass
		means[m] = tmp / count

    out = []
    for i in range(len(membership)):
        out.append(means[int(membership[i])])
    return out

def cluster(tr, te, tr_class):
    pred = kmeans(2,tr,len(data[0,:]))
    print pred
    print tr_class
