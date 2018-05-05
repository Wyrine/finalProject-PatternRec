#!/usr/local/bin/python2
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv, exit

dat = pd.read_csv("EEG_converted.csv")

if len(argv) < 3:
    print("failure.\n")
    exit()
print dat
plt.scatter(dat[argv[1]],dat[argv[2]],c=dat["user-definedlabeln"])
plt.show()
