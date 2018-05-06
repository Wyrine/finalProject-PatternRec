def kmeans(k, dat, col):
    '''
    returns the membership of each the samples passed in
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
		dist = __euc(dat[j],means[m])
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
