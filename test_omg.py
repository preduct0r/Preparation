import subprocess
import os
import pickle

import pandas as pd
import numpy as np
from scipy.io import wavfile


def calc_feature(wav_path):
    global opensmile_config_path
    global opensmile_root_dir

    sr, wav_data = wavfile.read(wav_path)
    n_samples = len(wav_data)
    w_step = int(0.5 * sr)
    w_len = int(1.0 * sr)

    features = []

    features_file = os.path.join(os.path.dirname(wav_path), 'temp.csv')
    tmp_wav = os.path.join(os.path.dirname(wav_path), 'tmp.wav')
    for i in range(0, n_samples - w_len, w_step):
        wavfile.write(tmp_wav, sr, wav_data[i: i + w_len])
        command = "{opensmile_dir}/bin/Win32/SMILExtract_Release -I {input_file} -C {conf_file} --csvoutput {output_file}".format(
            opensmile_dir=opensmile_root_dir,
            input_file=tmp_wav,
            conf_file=opensmile_config_path,
            output_file=features_file)
        command = command.replace('\\', '/')
        res = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

        ##----------------------------------------------------------
        ## merge metadata and features
        ##----------------------------------------------------------
        feature = pd.read_csv(features_file, sep=';', index_col=None)
        ## get rid of useless column
        feature.drop('name', axis=1, inplace=True)
        ## force the indexes to be equal so they will concat into 1 row. WTF, pandas?
        feature = np.transpose(feature.as_matrix()[0])

        if os.path.exists(features_file):
            os.remove(features_file)
        features.append(feature)
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    # print("processing complete!")
    return np.asarray(features)


def get_x_y_file(txt_path, path_to_wavs):
    global opensmile_config_path
    global opensmile_root_dir
    col_names = ['f', 'c1', 'c2', 'num_label'] + [str(i) for i in range(136)]

    test_data = pd.read_csv(txt_path, sep=' ', names=col_names)
    # lambda transform 04899849f_1\utterance_6.mp4\93.jpg into 04899849f_1\utterance_6.mp4.wav
    # that's what we need
    f = np.array(test_data['f'].apply(lambda arg: os.path.dirname(arg) + '.wav'))
    # only one occurrence of each file we need
    f, ind = np.unique(f, return_index=True)
    y = np.array(test_data['num_label'])[ind]
    x = []
    for f_name in f:
        print('Process file: {}...'.format(f_name), end='')
        f1 = os.path.join(path_to_wavs, f_name)
        x.append(calc_feature(f1))
        print('done.')
    return np.asarray(x), y, f


# opensmile configuration
opensmile_root_dir = r'C:\opensmile230'
# opensmile_config_path = r'C:\opensmile230\config\avec2013.conf'
opensmile_config_path = r'C:\opensmile230\config\IS11_speaker_state.conf'

if __name__ == '__main__':
    f_train = r'C:\Users\agafonov\Desktop\OMGEmotionChallenge-master\omg_TrainVideos\train_data_with_landmarks.txt'
    train_wav_path = r'C:\Users\agafonov\Desktop\OMGEmotionChallenge-master\omg_TrainVideos\wave'
    train_pickle = r'C:\Users\agafonov\Desktop\OMGEmotionChallenge-master\full_train_opensmile_2.3.0.pkl'
    x_train, y_train, f_train = get_x_y_file(f_train, train_wav_path)
    # save results
    with open(train_pickle, 'wb') as f:
        pickle.dump([x_train, y_train, f_train], f)

    f_test = r'C:\Users\agafonov\Desktop\OMGEmotionChallenge-master\omg_ValidVideos\valid_data_with_landmarks.txt'
    test_wav_path = r'C:\Users\agafonov\Desktop\OMGEmotionChallenge-master\omg_ValidVideos\wave'
    test_pickle = r'C:\Users\agafonov\Desktop\OMGEmotionChallenge-master\full_test_opensmile_2.3.0.pkl'
    x_test, y_test, f_test = get_x_y_file(f_test, test_wav_path)
    # save results
    with open(test_pickle, 'wb') as f:
        pickle.dump([x_test, y_test, f_test], f)
