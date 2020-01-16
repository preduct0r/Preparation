import pandas as pd
import numpy as np
import os

# объединяем разметку из meta_train.csv и результат vad, пропущенный через наши данные по общему имени файла
df_crepe = pd.read_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\aff_beh_test.csv")
df_prep = pd.read_csv(os.path.join("some_path", "meta_train.csv"))                # поменяй some_path
df_prep['vad'] = np.nan


grouped_crepe = df_crepe.groupby('name')
grouped_prep = df_prep.groupby('init_name')
for curr_file in df_prep['init_name'].unique():
     curr_crepe_group = grouped_crepe.get_group(curr_file)
     curr_prep_group = grouped_prep.get_group(curr_file)
     for idx, row in curr_prep_group.iterrows():
         label, start, end = row[3], round(row[5],1), round(row[6],1)
         confidence = curr_crepe_group[start<curr_crepe_group['time']<end]['confidence'].agg(np.mean)
         df_prep.loc[idx,'vad'] = confidence
