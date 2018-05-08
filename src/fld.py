#!/usr/local/bin/python3
import numpy as np

def fld(tr, tr_class):
    '''
				returns the FLD projection matrix
    '''
    mat0 = []
    mat1 = []
    for i in range(len(tr)):
        if tr_class[i] == 0:
            mat0.append(tr[i])
        else:
            mat1.append(tr[i])

    #compute scatter
    sInv = ((len(mat0)-1)  * np.cov(mat0,rowvar=False)) + ((len(mat1)-1) * np.cov(mat1,rowvar=False))
    sInv = np.linalg.inv(sInv)

    return np.matmul(sInv, (np.mean(mat0,axis=0) - np.mean(mat1,axis=0)))
