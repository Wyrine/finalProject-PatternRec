#!/usr/local/bin/python3
import pandas as pd
df = pd.read_csv("EEG_data.csv")
#df2 = df.drop("predefinedlabel", axis=1)
#print(df2.to_csv(index=False))
print(df.to_csv(index=False))
