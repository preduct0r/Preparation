"""
Этот скрипт собирает базу из файлов, которые были собраны для пресета caralarm осенью 2018 года.
\\stor.stc\Science\SIV\DB_CAR_ALARM
"""

import os
from os.path import join, exists
import yaml

import pandas as pd
from scipy.io import wavfile
import numpy as np

from prepare_waveassistant import listdir_wav_seg, parse_seg


def prep_label(init_label):
    transform_dict = {
        'e': 'engine',
        'i': 'idling',
        'g': 'glass',
    }
    if init_label in transform_dict:
        return transform_dict[init_label]
    return init_label.lower().replace(' ', '-').replace('_', '-')


def prepare(remote_storage_path):
    # папки, в которых данные лежат с форматом разметки waveassistant
    waveassistant_folders = ['engine', 'glass']

    # inside base_descr_yaml
    base_descr_yaml = {
        'data_path': 'prepared_data',
        'preprocessed_path': 'preprocessed',
        'feature_path': 'feature',
        'base_name': 'safecity-1',
        'general_meta': 'meta.csv',
        'train_meta': 'meta_train.csv',
        'test_meta': 'meta_test.csv',
        'parent_bases': [],
    }

    # create new base_descr_yaml or return existing one
    remote_base_path = join(remote_storage_path, base_descr_yaml['base_name'])
    base_descr_file = join(remote_base_path, 'base_description.yml')
    if exists(base_descr_file):
        print('Base {} has been prepared before.'.format(base_descr_yaml['base_name']))
        return True

    data_path = join(remote_base_path, base_descr_yaml['data_path'])
    if not exists(data_path):
        os.mkdir(data_path)

    # прописываем пути к файлам с мета данными
    meta_file_general = join(remote_base_path, base_descr_yaml['general_meta'])
    meta_file_train = join(remote_base_path, base_descr_yaml['train_meta'])
    meta_file_test = join(remote_base_path, base_descr_yaml['test_meta'])

    init_names, cur_names, init_labels, cur_labels, bases, ends, begins, ids = [], [], [], [], [], [], [], []

    def upd_meta_lists(init_name_, cur_name_, init_label_, cur_label_, base, begin, end, id_):
        """
        Функция просто для экономии места в коде.
        """
        init_names.append(init_name_)
        cur_names.append(cur_name_)
        init_labels.append(init_label_)
        cur_labels.append(cur_label_)
        bases.append(base)
        begins.append(begin)
        ends.append(end)
        ids.append(id_)

    # и создаем папки preprocessed_path и feature_path
    if not exists(join(remote_base_path, base_descr_yaml['preprocessed_path'])):
        os.mkdir(join(remote_base_path, base_descr_yaml['preprocessed_path']))

    if not exists(join(remote_base_path, base_descr_yaml['feature_path'])):
        os.mkdir(join(remote_base_path, base_descr_yaml['feature_path']))

    meta_id = 0
    target_labels = None

    # если нужно игнорировать какой-либо класс, то нужно использовать другой словарь!
    # этот только для начала и конца файла!
    special_labels = ['End File', 'Begin File']

    # work with wave_assistan format
    for event_folder in waveassistant_folders:
        for wav_f_name_full, old_wavfile, seg_f_name_full in listdir_wav_seg(join(remote_base_path, event_folder)):
            segments = parse_seg(seg_f_name_full)
            # отдельный столбец с подготовленными метками
            segments['new_label'] = segments['label'].apply(prep_label)

            # счетчик объектов классов
            if target_labels is None:
                target_labels = {label: 0 for label in np.unique(segments.new_label)}
            else:
                uniq_labels = np.unique(segments.new_label)
                for un_lab in uniq_labels:
                    if un_lab not in target_labels.keys():
                        target_labels[un_lab] = 0


            sr, wav_data = wavfile.read(wav_f_name_full)
            for i, row in segments.iterrows():
                old_label = row['label']
                if old_label in special_labels:
                    continue

                label = row['new_label']
                start_sample = row['start_sample']
                end_sample = row['end_sample']
                f_name = '{}_{}_{}.wav'.format(label, base_descr_yaml['base_name'], target_labels[label])
                out_wav_name = os.path.join(data_path, f_name)
                target_labels[label] += 1
                wavfile.write(out_wav_name, sr, wav_data[start_sample: end_sample])
                upd_meta_lists(old_wavfile, f_name, old_label, label, base_descr_yaml['base_name'], start_sample / sr,
                               end_sample / sr, meta_id)
                meta_id += 1

    # work with raw format (crowd)
    crowd_dir = join(remote_base_path, 'crowd')
    label = 'crowd'
    target_labels[label] = 0
    for wav_f_name in os.listdir(crowd_dir):
        wav_f_name_full = join(crowd_dir, wav_f_name)

        f_name = '{}_{}_{}.wav'.format(label, base_descr_yaml['base_name'], target_labels[label])
        out_wav_name = os.path.join(data_path, f_name)
        target_labels[label] += 1

        sr, wav_data = wavfile.read(wav_f_name_full)

        upd_meta_lists(wav_f_name, f_name, label, label, base_descr_yaml['base_name'], 0,
                       len(wav_data) / sr, meta_id)
        meta_id += 1
        wavfile.write(out_wav_name, sr, wav_data)

    # make full meta
    meta_df = pd.DataFrame(list(zip(ids, init_names, cur_names, init_labels,
                                    cur_labels, bases, begins, ends)),
                           columns=['id', 'init_name', 'cur_name', 'init_label',
                                    'cur_label', 'database', 'begin', 'end'])
    meta_df.to_csv(meta_file_general, index=False, sep=';')
    print(target_labels)

    test_size = 0.3
    label = list(target_labels.keys())[0]

    count = target_labels[label]
    test_count = int(test_size * count)
    # make train/test meta
    class_df = meta_df[meta_df['cur_label'] == label]
    test_meta_df = class_df[:test_count]
    train_meta_df = class_df[test_count:]

    for label, count in list(target_labels.items())[1:]:
        class_df = meta_df[meta_df['cur_label'] == label]
        test_count = int(test_size * count)
        test_meta_df = test_meta_df.append(class_df.iloc[:test_count])
        train_meta_df = train_meta_df.append(class_df.iloc[test_count:])

    test_meta_df.to_csv(meta_file_test, index=False, sep=';')
    train_meta_df.to_csv(meta_file_train, index=False, sep=';')

    # save base meta
    base_descr_yaml['events'] = target_labels
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_descr_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_descr_yaml['base_name']))
    return True


# if __name__ == '__main__':
#     # путь к расположению базы
#     storage_path = r'C:\Users\kotov-d\Documents\safecity'
#     prepare(storage_path)
