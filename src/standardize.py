#!/usr/local/bin/python3
import numpy as np

def standard(train, test):
    test = (test - np.mean(train)) / np.std(train)
    train = (train - np.mean(train)) / np.std(train)
    return train, test

#test
if __name__ == "__main__":
    tr = [0, 1, 2]
    te = [0, 5, 7]
    tr,te = standard(tr, te)
    print(tr)
    print(te)
