"""
Пример приведения базу к общему виду
input_base_path - директория, в которой лежит исходная база.

output_base_path - директория, в которой появится подготовленная база.

Данный скрипт собирает только записи импровизированных выражений эмоций от актеров!
В базе есть так же записи эмоций, который отыгрываются всеми актерами по одному сценарию,
этот скрипт их игнорирует!
"""
import os
import yaml

from scipy.io import wavfile
import pandas as pd

# путь к базе, которую хотим подготовить
input_base_path = r'C:\Projects\EventDetectionSDK\python\experiments\emo\dummy_base_example\dummy_base'

# именно здесь появится папка с новой подготовленной базой
output_base_path = r'C:\Projects\EventDetectionSDK\python\experiments\emo\dummy_base_example'


if __name__ == '__main__':
    base_meta_yaml = {
        'base_name': 'dummy',  # имя подготовленной базы
        'general_meta': 'meta.csv',  # общая разметка по всем файлам
        'test_meta': 'meta_test.csv',  # разметка по тесту
        'train_meta': 'meta_train.csv',  # разметка по трейну
        'data_path': 'data',  # папка, в которой будут находиться все вавки
        'feature_path': 'feature',  # папка, в которой будут находиться все признаки
        'preprocessed_path': 'prepocessed',  # папка, в которой будут находиться все предобработанные вавки
        # список баз, из которых будет собираться текущая база (пока лишние, просто оставляем пустым)
        'parent_bases': [],
        # список дополнительных меток, присутствующих в базе помимо основной категориальной
        # столбцы с такими именами должны быть в каждой мете, указанной в base_meta_yaml
        'extra_labels': ['valence',
                         'activation',
                         'dominance'],
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
    #
    col_names = ['id', 'init_name', 'cur_name', 'init_label', 'cur_label',
                 'database', 'begin', 'end'] + base_meta_yaml['extra_labels']
    # списки значений для каждого из столбцов
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom = [], [], [], [], [], [], [], [], [], [], []

    # пути к метам
    meta_file_test = os.path.join(output_base_path, base_meta_yaml['test_meta'])
    meta_file_train = os.path.join(output_base_path, base_meta_yaml['train_meta'])
    meta_file_general = os.path.join(output_base_path, base_meta_yaml['general_meta'])

    # теперь уже начинаем обрабатывать нашу базу
    file_id = 0
    # подсчет, сколько примеров каждого класса есть в базе
    labels_counter = {}
    input_train_data = pd.read_csv(os.path.join(input_base_path, 'dummy_meta_train.csv'), sep=';')
    # обработка всех трейн файлов
    for i, row in input_train_data.iterrows():
        init_label = row['label']
        # у нас может появиться необходимость в исправлении метки, поэтому есть следующая строка
        # сейчас она бесполезна
        new_label = row['label']
        # добавляем инфу по классу в счетчик
        if new_label not in labels_counter:
            labels_counter[new_label] = 1
        else:
            labels_counter[new_label] += 1

        wav_name = row['f_name']
        full_wav_name = os.path.join(input_base_path, 'train_data', wav_name)
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

        # значения дополнительных меток
        valence, activation, dominance = row['v'], row['a'], row['d']

        # теперь добавляем полученную инфу в каждый столбец
        ids.append(file_id)
        file_id += 1
        init_names.append(wav_name)
        cur_names.append(new_wav_name)
        init_labels.append(init_label)
        cur_labels.append(new_label)
        dbs.append(base_meta_yaml['base_name'])
        begs.append(start)
        ends.append(end)
        val.append(valence)
        act.append(activation)
        dom.append(dominance)
    # создаем датафрейм
    train_meta_df = pd.DataFrame(
        list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom)),
        columns=col_names)
    # сохраняем
    train_meta_df.to_csv(meta_file_train, index=False, sep=';')
    # обнуляем значения
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom = [], [], [], [], [], [], [], [], [], [], []

    # теперь повторяем то же самое для теста
    input_test_data = pd.read_csv(os.path.join(input_base_path, 'dummy_meta_test.csv'), sep=';')
    # обработка всех трейн файлов
    for i, row in input_test_data.iterrows():
        init_label = row['label']
        # у нас может появиться необходимость в исправлении метки, поэтому есть следующая строка
        # сейчас она бесполезна
        new_label = row['label']
        # добавляем инфу по классу в счетчик
        if new_label not in labels_counter:
            labels_counter[new_label] = 1
        else:
            labels_counter[new_label] += 1

        wav_name = row['f_name']
        full_wav_name = os.path.join(input_base_path, 'test', wav_name)
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

        # значения дополнительных меток
        valence, activation, dominance = row['v'], row['a'], row['d']

        # теперь добавляем полученную инфу в каждый столбец
        ids.append(file_id)
        file_id += 1
        init_names.append(wav_name)
        cur_names.append(new_wav_name)
        init_labels.append(init_label)
        cur_labels.append(new_label)
        dbs.append(base_meta_yaml['base_name'])
        begs.append(start)
        ends.append(end)
        val.append(valence)
        act.append(activation)
        dom.append(dominance)

    test_meta_df = pd.DataFrame(
        list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom)),
        columns=col_names)
    test_meta_df.to_csv(meta_file_test, index=False, sep=';')
    # сохранение общей меты
    (train_meta_df.append(test_meta_df)).to_csv(meta_file_general, index=False, sep=';')

    # добавляем посчитанные события в макро описание базы и сохраняем описание
    base_meta_yaml['events'] = labels_counter
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_meta_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_meta_yaml['base_name']))
