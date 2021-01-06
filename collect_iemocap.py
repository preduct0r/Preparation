import os
from fnmatch import fnmatch
from shutil import copyfile
import yaml

import pandas as pd


if __name__ == '__main__':
    base_meta_yaml = {
        'base_name': 'iemocap',
        'general_meta': 'meta.csv',
        'test_meta': 'meta_test.csv',
        'train_meta': 'meta_train.csv',
        'data_path': 'data',
        'feature_path': 'feature',
        'preprocessed_path': 'prepocessed',
        # список баз, из которых будет собираться текущая база
        'parent_bases': [],
    }
    # make dirs for new base
    prepared_base_path = '/home/den/datasets/iemocap'
    if not os.path.exists(prepared_base_path):
        os.mkdir(prepared_base_path)

    data_path = os.path.join(prepared_base_path, base_meta_yaml['data_path'])
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    preprocessed_path = os.path.join(prepared_base_path, base_meta_yaml['preprocessed_path'])
    if not os.path.exists(preprocessed_path):
        os.mkdir(preprocessed_path)

    feature_path = os.path.join(prepared_base_path, base_meta_yaml['feature_path'])
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)

    base_descr_file = os.path.join(prepared_base_path, 'base_description.yml')

    # for csv
    col_names = ['id', 'init_name', 'cur_name', 'init_label', 'cur_label',
                 'database', 'begin', 'end', 'valence', 'activation', 'dominance']
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom = [], [], [], [], [], [], [], [], [], [], []

    meta_file_test = os.path.join(prepared_base_path, base_meta_yaml['test_meta'])
    meta_file_train = os.path.join(prepared_base_path, base_meta_yaml['train_meta'])
    meta_file_general = os.path.join(prepared_base_path, base_meta_yaml['general_meta'])

    # тестовую сессию обязательно ставим в конец
    test_sess_id = 4
    sessions = [1, 2, 3, 4, 5]
    for i in range(len(sessions)):
        if sessions[i] == test_sess_id:
            sessions[i], sessions[-1] = sessions[-1], sessions[i]
            break
    # iterate through sessions
    path_to_base = '/media/den/0080AF5B80AF55C6/grive/Ulma/IEMOCAP/IEMOCAP_full_release'
    file_id = 0
    labels_counter = {}

    for sess in sessions:
        session_path = os.path.join(path_to_base, 'Session{}'.format(sess))
        label_path = os.path.join(session_path, 'dialog', 'EmoEvaluation')
        for f_name in os.listdir(label_path):
            if not fnmatch(f_name, '*impro??.txt') or f_name.startswith('._'):
                continue
            with open(os.path.join(label_path, f_name), 'r') as f:
                for line in f:
                    line = line.strip()
                    if fnmatch(line, '[[]*[]]'):
                        parts = line.split('	')
                        cat_label = parts[2]  # categorical
                        if cat_label not in labels_counter.keys():
                            labels_counter[cat_label] = 1
                        else:
                            labels_counter[cat_label] += 1

                        # reading [2.5000, 2.5000, 2.5000]
                        valence, activation, dominance = parts[-1][1:-1].split(', ')
                        start, _, end = parts[0][1: -1].split(' ')
                        # from Ses01F_impro01_F001 to Ses01F_impro01
                        wav_path = '_'.join(parts[1].split('_')[:2])

                        wav_name = parts[1] + '.wav'
                        wav_path = os.path.join(session_path, 'sentences', 'wav', wav_path, wav_name)
                        copyfile(wav_path, os.path.join(data_path, wav_name))

                        # update meta info
                        ids.append(file_id)
                        file_id += 1
                        init_names.append(wav_name)
                        cur_names.append(wav_name)
                        init_labels.append(cat_label)
                        cur_labels.append(cat_label)
                        dbs.append('iemocap')
                        begs.append(start)
                        ends.append(end)
                        val.append(valence)
                        act.append(activation)
                        dom.append(dominance)
        # обработан весь train
        if sess == sessions[-2]:
            train_meta_df = pd.DataFrame(
                list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom)),
                columns=col_names)
            train_meta_df.to_csv(meta_file_train, index=False, sep=';')
            ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom = [], [], [], [], [], [], [], [], [], [], []
        # это будет тест
        if sess == sessions[-1]:
            test_meta_df = pd.DataFrame(
                list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends, val, act, dom)),
                columns=col_names)
            test_meta_df.to_csv(meta_file_test, index=False, sep=';')
    (train_meta_df.append(test_meta_df)).to_csv(meta_file_general, index=False, sep=';')
    # добавляем посчитанные события
    base_meta_yaml['events'] = labels_counter
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_meta_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_meta_yaml['base_name']))
