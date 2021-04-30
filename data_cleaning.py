import numpy as np
import pandas as pd

df = pd.read_csv("glassdor_jobs.csv")

print(df.isnull().sum())