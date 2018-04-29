import numpy as np


a = np.arange(12811)
np.random.shuffle(a)

for i in range(0, 12811, 557):
    s  = ",".join(a[i : min(i+557, 12810) ].astype(str))
    print s
