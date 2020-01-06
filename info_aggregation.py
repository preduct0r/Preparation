import os
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import glob

dict_counter = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}


def get_stat(file_name):
    data = pd.read_csv(file_name, skiprows=[0], header=None, names=["label"])
    t = data.reset_index()[data.label.shift() != data.label].assign(f_name=file_name)
    t['to'] = t['index'].shift(-1)
    t['frames'] = t['to'] - t['index']
    t['seconds'] = t.frames / 30.
    t.rename(columns={"index": "from"}, inplace=True)
    t.reset_index(drop=True, inplace=True)
    return t

data = pd.concat([get_stat(i) for i in glob.glob(r"C:\Users\kotov-d\Documents\Aff_Beh\annotations\EXPR_Set\Training_Set\*")])
data = data[data.label!=(-1)]
data = data[data.frames>30]


with open(r'C:\Users\kotov-d\Documents\aff_preprocess\data_about_sep.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=2)

