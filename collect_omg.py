import os
from os.path import join, exists
import yaml

import pandas as pd
from scipy.io import wavfile
import numpy as np
from prepare_waveassistant import listdir_wav_seg, parse_seg
from glob import glob
import ntpath
import re
import shutil

def change_label(x):
    if x==0:
        return 'anger'
    elif x==1:
        return 'disgust'
    elif x==2:
        return 'fear'
    elif x==3:
        return 'happiness'
    elif x==4:
        return 'neutrality'
    elif x==5:
        return 'sadness'
    elif x==6:
        return 'surprise'

def prepare(input_base_path, output_base_path):
    # папки, в которых данные лежат
    inner_folders = ['omg_TrainVideos', 'omg_ValidVideos']

    # inside base_descr_yaml
    base_meta_yaml = {
        'base_name': 'omg',
        'general_meta': 'meta.csv',
        'test_meta': 'meta_test.csv',
        'train_meta': 'meta_train.csv',
        'data_path': 'data',
        'feature_path': 'feature',
        'preprocessed_path': 'prepocessed',
        # список баз, из которых будет собираться текущая база
        'parent_bases': [],
        'extra_labels': ['valence',
                         'arousal']
    }
    # создание всех папок, упомянутых в base_meta_yaml
    output_base_path = os.path.join(output_base_path, base_meta_yaml['base_name'])
    # make dirs for new base
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    data_path = os.path.join(output_base_path, base_meta_yaml['data_path'])
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    preprocessed_path = os.path.join(output_base_path, base_meta_yaml['preprocessed_path'])
    if not os.path.exists(preprocessed_path):
        os.mkdir(preprocessed_path)

    feature_path = os.path.join(output_base_path, base_meta_yaml['feature_path'])
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)

    base_descr_file = os.path.join(output_base_path, 'base_description.yml')

    # имена столбцов в будущей разметке
    col_names = ['id', 'init_name', 'cur_name', 'init_label', 'cur_label',
                 'database', 'begin', 'end'] + base_meta_yaml['extra_labels']
    # списки значений для каждого из столбцов
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, arous = [], [], [], [], [], [], [], [], [], []

    # пути к метам
    meta_file_test = os.path.join(output_base_path, base_meta_yaml['test_meta'])
    meta_file_train = os.path.join(output_base_path, base_meta_yaml['train_meta'])
    meta_file_general = os.path.join(output_base_path, base_meta_yaml['general_meta'])

    # теперь уже начинаем обрабатывать нашу базу
    file_id = 0
    # подсчет, сколько примеров каждого класса есть в базе
    labels_counter = {}
    for inner_dir in inner_folders:
        if not os.path.exists(input_base_path+"\\"+inner_dir+"\\labels_for_audio_{}.csv".format(inner_dir)):
            # избавимся от лишней инфы в данных по разметке
            with open(glob(input_base_path+"\\"+inner_dir+"\\*.txt")[0], 'r') as f:
                rows = f.readlines()
                data = []
                for idx,row in enumerate(rows):
                    [f_name, valence, arousal, label] = row.split(' ')[0:4]
                    z = re.findall('[0-9]+.jpg$', f_name)[0]
                    data.append([f_name[:-(len(z)+1)], valence, arousal, label])
                data_about_labels = pd.DataFrame(data = np.array(data), columns = ['name', 'valence', 'arousal', 'label'])
                data_about_labels.drop_duplicates(inplace=True)
                data_about_labels.to_csv(path_or_buf=input_base_path+"\\"+inner_dir+"\\labels_for_audio_{}.csv".format(inner_dir), index=False)
                # на выходе два новых файла без дупликатов, названия которых совпадают с именами wav файлов

        else:
            data_about_labels = pd.read_csv(input_base_path+"\\"+inner_dir+"\\labels_for_audio_{}.csv".format(inner_dir))

        inner_path = input_base_path+'\\'+inner_dir+'\\wave'
        for i, row in data_about_labels.iterrows():
            init_label = row['label']
            # у нас может появиться необходимость в исправлении метки, поэтому есть следующая строка
            # сейчас она бесполезна
            new_label = change_label(row['label'])
            # добавляем инфу по классу в счетчик
            if new_label not in labels_counter:
                labels_counter[new_label] = 1
            else:
                labels_counter[new_label] += 1

            full_wav_name = inner_path+'\\'+row['name']+'.wav'
            # оставлю это здесь, вдруг пригодится
            sr, wav_data = wavfile.read(full_wav_name)

            # новое имя: класс база счетчик
            new_wav_name = '{}_{}_{}.wav'.format(new_label, base_meta_yaml['base_name'], labels_counter[new_label])
            # полный путь до новой вавки
            new_wav_name_full = os.path.join(output_base_path, base_meta_yaml['data_path'], new_wav_name)

            # копируем вавку
            wavfile.write(new_wav_name_full, sr, wav_data)

            # время начала конца события (сейчас неизвестно, поэтому берем весь файл)
            start = 0
            end = len(wav_data) / sr

            valence, arousal = row['valence'], row['arousal']

            # теперь добавляем полученную инфу в каждый столбец
            ids.append(file_id)
            file_id += 1
            init_names.append(row['name'])
            cur_names.append(new_wav_name)
            init_labels.append(init_label)
            cur_labels.append(new_label)
            dbs.append(base_meta_yaml['base_name'])
            begs.append(start)
            ends.append(end)
            val.append(valence)
            arous.append(arousal)

        # создаем датафрейм
        meta_df = pd.DataFrame(
            list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, arous)),
            columns=col_names)
        # сохраняем
        if inner_dir=='omg_TrainVideos':
            meta_df.to_csv(meta_file_train, index=False, sep=';')
        elif inner_dir=='omg_ValidVideos':
            meta_df.to_csv(meta_file_test, index=False, sep=';')
        else:
            raise Exception('Some mistake occured with meta saving')


        # обнуляем значения
        ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom = [], [], [], [], [], [], [], [], [], [], []

    # сохранение общей меты
    train_meta_df = pd.read_csv(meta_file_train, sep=';')
    test_meta_df = pd.read_csv(meta_file_test, sep=';')
    # (pd.concat([train_meta_df, test_meta_df])).to_csv(meta_file_general, index=False)
    (train_meta_df.append(test_meta_df)).to_csv(meta_file_general, index=False, sep=';')



    # добавляем посчитанные события в макро описание базы и сохраняем описание
    base_meta_yaml['events'] = labels_counter
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_meta_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_meta_yaml['base_name']))



if __name__ == '__main__':
    # путь к базе, которую хотим подготовить
    input_base_path = r'C:\Users\kotov-d\Documents\task#1\OMGEmotionChallenge-master'

    # именно здесь появится папка с новой подготовленной базой
    output_base_path = r'C:\Users\kotov-d\Documents\bases'

    prepare(input_base_path, output_base_path)


