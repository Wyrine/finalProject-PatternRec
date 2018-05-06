'''
returns the fusion choices based on confusion matrices of two classifiers
***C1, C2 should be passed by value to prevent overwrite
'''
def fuse(C1, C2):
    fuse = np.zeros(4).reshap([2,2])

    #divide elements by row sum
    C1[0][0] = C1[0][0] / ( C1[0][0] + C1[0][1] )
    C1[0][1] = C1[0][1] / ( C1[0][0] + C1[0][1] )
    C1[1][0] = C1[1][0] / ( C1[1][0] + C1[1][1] )
    C1[1][1] = C1[1][1] / ( C1[1][0] + C1[1][1] )
    C2[0][0] = C2[0][0] / ( C2[0][0] + C2[0][1] )
    C2[0][1] = C2[0][1] / ( C2[0][0] + C2[0][1] )
    C2[1][0] = C2[1][0] / ( C2[1][0] + C2[1][1] )
    C2[1][1] = C2[1][1] / ( C2[1][0] + C2[1][1] )

    #c1 0 c2 0
    if (C1[0][0]*C2[0][0]) > (C1[1][0]*C2[1][0]):
	fuse[0][0] = 0
    else:
	fuse[0][0] = 1

    #c1 0 c2 1
    if (C1[0][0]*C2[0][1]) > (C1[1][0]*C2[1][1]):
	fuse[0][1] = 0
    else:
	fuse[0][1] = 1

    #c1 1 c2 0
    if (C1[0][1]*C2[0][0]) > (C1[1][1]*C2[1][0]):
	fuse[1][0] = 0
    else:
	fuse[1][0] = 1

    #c1 1 c2 1
    if (C1[0][1]*C2[0][1]) > (C1[1][1]*C2[1][1]):
	fuse[1][1] = 0
    else:
	fuse[1][1] = 1

    return fuse