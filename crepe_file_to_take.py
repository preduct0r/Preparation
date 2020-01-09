import numpy as np
import pandas as pd


df = pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\aff_beh_test.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)

grouped = df.groupby(by='name')
grouped['confidence'].agg(np.mean).to_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\aff_mean_confidence_per_file.csv")