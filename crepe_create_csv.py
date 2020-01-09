import numpy as np
import pandas as pd
import tensorflow as tf


import os
from glob import glob
import crepe
from scipy.io import wavfile

all_files = glob(r"C:\Users\kotov-d\Documents\BASES\aff_beh\data\*")
df_name, df_time, df_confidence = [],[],[]

# =stackoverflow snippet =====================================
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# ============================================================

for file_path in all_files:
    try:
        sr, audio = wavfile.read(file_path)
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
        for ti, co in zip(time, confidence):
            df_name.append(os.path.basename(file_path))
            df_time.append(ti)
            df_confidence.append(co)
    except Exception as err:
        print(err)
        print('dermo')
df = pd.DataFrame(columns=['name', 'time', 'confidence'], index=range(len(df_name)))
df['name'] = df_name
df['time'] = df_time
df['confidence'] = df_confidence

df.to_csv(r"C:\Users\kotov-d\Documents\TASKS\task#7\aff_beh_test.csv")

