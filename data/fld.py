'''
returns the FLD projection matrix
'''
def fld(tr, tr_class):
    mat0 = []
    mat1 = []
    for i in range(len(tr)):
        if tr_class[i] == 0:
            mat0.append(tr[i])
        else:
            mat1.append(tr[i])

    sInv = ((len(mat0)-1)  * np.cov(mat0)) + ((len(mat1)-1) * np.cov(mat1))
    sInv = np.linalg.inv(sInv)

    return sInv * (np.mean(mat0) - np.mean(mat1))
