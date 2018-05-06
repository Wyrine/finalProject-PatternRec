import numpy as np

def pca(tr, tol):
    '''
    returns the number of eigenvectors to drop, given tolerance
    '''
    c = np.cov(tr)
    eigs,vec = np.linalg.eig(tr)
    tot = float(sum(eigs))
    accum = 0.0
    drop = 0
    for i in range(len(eigs),0,-1): #step in direction of increasing magnitude
        accum += eigs[i]
        if(accum / tot) > tol:
            return drop
            break
        drop += 1
