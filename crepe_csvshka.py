import numpy as np
import pandas as pd

import os
from glob import glob
import crepe
from scipy.io import wavfile

all_files = glob(r"C:\Users\kotov-d\Documents\aff_preprocess\aff_beh\data\*")
df_name, df_time, df_confidence = [],[],[]
for file_path in all_files:
    sr, audio = wavfile.read(file_path)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    for ti, co in zip(time, confidence):
        df_name.append(os.path.basename(file_path))
        df_time.append(ti)
        df_confidence.append(co)
df = pd.DataFrame(columns=['name', 'time', 'confidence'], index=range(len(df_name)))
df['name'] = df_name
df['time'] = df_time
df['confidence'] = df_confidence

df.to_csv(r"C:\Users\kotov-d\Documents\task#7\test.csv")

