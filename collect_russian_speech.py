import os
import yaml

import pandas as pd
from scipy.io import wavfile
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

def change_label(x):
    if x==0:
        return 'sadness'
    elif x==1:
        return 'neutrality'
    elif x==2:
        return 'happiness'

def prepare(input_base_path, output_base_path):
    # папки, в которых данные лежат
    inner_folders = ['Депрессивно', 'Нейтрально', 'Радостно']

    # inside base_descr_yaml
    base_meta_yaml = {
        'base_name': 'russ_tts',
        'general_meta': 'meta.csv',
        'test_meta': 'meta_test.csv',
        'train_meta': 'meta_train.csv',
        'data_path': 'data',
        'feature_path': 'feature',
        'preprocessed_path': 'prepocessed',
        # список баз, из которых будет собираться текущая база
        'parent_bases': [],
        'extra_labels': []
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
    col_names = ['ids', 'init_name', 'cur_name', 'init_label', 'cur_label',
                 'database', 'begin', 'end'] + base_meta_yaml['extra_labels']
    # списки значений для каждого из столбцов
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends= [], [], [], [], [], [], [], []

    # пути к метам
    meta_file_test = os.path.join(output_base_path, base_meta_yaml['test_meta'])
    meta_file_train = os.path.join(output_base_path, base_meta_yaml['train_meta'])
    meta_file_general = os.path.join(output_base_path, base_meta_yaml['general_meta'])

    # теперь уже начинаем обрабатывать нашу базу
    file_id = 0
    # подсчет, сколько примеров каждого класса есть в базе
    labels_counter = {"sadness":0, "neutrality":0, "happiness":0}
    for label, inner_dir in zip(["sadness", "neutrality", "happiness"], inner_folders):
        for fi in glob(input_base_path+"\\"+inner_dir+"\\*.wav"):
            # add to counter
            labels_counter[label] += 1
            # preprocess data
            sr, wav_data = wavfile.read(fi)
            # новое имя: класс база счетчик
            new_wav_name = '{}_{}_{}.wav'.format(label, base_meta_yaml['base_name'], labels_counter[label])
            # полный путь до новой вавки
            new_wav_name_full = os.path.join(output_base_path, base_meta_yaml['data_path'], new_wav_name)
            # копируем вавку
            wavfile.write(new_wav_name_full, sr, wav_data)
            # время начала конца события (сейчас неизвестно, поэтому берем весь файл)
            start = 0
            end = len(wav_data) / sr
            ids.append(file_id)
            file_id += 1
            init_names.append(os.path.basename(fi))
            cur_names.append(new_wav_name)
            init_labels.append(label)
            cur_labels.append(label)
            dbs.append(base_meta_yaml['base_name'])
            begs.append(start)
            ends.append(end)


    # создаем датафрейм
    meta_df = pd.DataFrame(
        list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends)),
        columns=col_names)

    X_train, X_test, y_train, y_test = \
        train_test_split(meta_df['ids'], meta_df['cur_label'], stratify=meta_df['cur_label'], test_size=0.25)
    meta_file_train_df = meta_df[meta_df['ids'].isin(X_train)]
    meta_file_test_df = meta_df[meta_df['ids'].isin(X_test)]

    # сохраняем
    meta_df.to_csv(meta_file_general, index=False, sep=';')
    meta_file_train_df.to_csv(meta_file_train, index=False, sep=';')
    meta_file_test_df.to_csv(meta_file_test, index=False, sep=';')


    # обнуляем значения
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom = [], [], [], [], [], [], [], [], [], [], []



    # добавляем посчитанные события в макро описание базы и сохраняем описание
    base_meta_yaml['events'] = labels_counter
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_meta_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_meta_yaml['base_name']))



if __name__ == '__main__':
    # путь к базе, которую хотим подготовить
    input_base_path = r'C:\Users\kotov-d\Documents\task#4\Образцы'

    # именно здесь появится папка с новой подготовленной базой
    output_base_path = r'C:\Users\kotov-d\Documents\bases'

    prepare(input_base_path, output_base_path)


