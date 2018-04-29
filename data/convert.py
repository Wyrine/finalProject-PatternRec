#!/usr/local/bin/python3
import pandas as pd
print(pd.read_csv("EEG_data.csv").to_csv())
