"""
Обрабатывает указанную базу, размеченную в ЦРТ с помощью WaveAssistant.
Использует разметку из *.seg файлов.

Нужно явно указывать имена файлов, события из которых будут отнесены в тестовое множество.
Сделано так, потому что изначально на несколько микрофонов писались одни и те же события,
и чтобы не было неявного пересечения трейна и теста (когда файлы не пересекаются, но события записаны те же).
"""

import os
from os.path import join, exists
from fnmatch import fnmatch
import yaml

import pandas as pd
from scipy.io import wavfile
import numpy as np

# from soundbaseprocessor import storage_path


def prep_label(label: str) -> str:
    """
    Приводит метку в исходной базе к нужному формату.
    Я пока взял за правило использовать "-" вместо пробелов и нижних подчеркиваний, а еще название в нижнем реестре.
    Это надо для того, чтобы одинаковые классы назывались одинаково в разных базах.
    Еще это удобно, когда делаем имена новых файлов в формате event-name_base-name_id.wav,
    можно быстро получить необходимую информацию по имени файла.
    :param label: метка, которую нужно отформатировать.
    :return: отформатированная метка.
    """
    _label = str(label)
    # исправление опечаток в базе
    if _label == 'windiw opening':
        _label = 'window opening'

    if _label == 'backgroung':
        _label = 'background'

    if _label == 'futniture':
        _label = 'furniture'

    return _label.lower().replace(' ', '-').replace('_', '-')


def listdir_wav_seg(path: str):
    """
    Вспомогательный генератор путей к wav-файлам, а также соответствующих сегментаций.
    :param path: директория, в которой хранятя *.wav и *.seg файлы
    :return: путь к *.wav файлу, имя *.wav файла отдельно, полный путь к *.seg файлу
    """
    for wav_file_name in os.listdir(path):
        if fnmatch(wav_file_name, '*.wav'):
            seg_file_name = wav_file_name[:-4] + '.seg'

            yield join(path, wav_file_name), wav_file_name, join(path, seg_file_name)


def parse_seg(seg_file: str) -> pd.DataFrame:
    """
    Извлекает информацию из указанного *.seg файлаю
    :param seg_file: путь к *.seg file
    :return: Извлеченная разметка.
             Имеет следующие поля:
                - start_sample: отсчет, с которого начинается событие в *.wav файле
                - end_sample: отсчет, которым заканчивается событие в *.wav файле
                - label: метка события (здесь ее не исправляем с помощью prep_label)
    """
    start_samples = []
    end_samples = []
    labels = []
    with open(seg_file, 'r') as f:
        line = ''
        while 'Begin File' not in line:
            line = f.readline()
            if 'SAMPLING_FREQ' in line:
                sample_rate = int(line.strip().split('=')[-1])
            if 'BYTE_PER_SAMPLE' in line:
                byte_depth = int(line.strip().split('=')[-1])
            if 'CODE' in line:
                code = int(line.strip().split('=')[-1])
            if 'N_CHANNEL' in line:
                n_channel = int(line.strip().split('=')[-1])

        class_name = 'stub'
        open_ids = []
        open_classes = []
        start_times = []
        while class_name != 'End File':
            byte, id_, class_name = f.readline().strip().split(',')
            # don't ask me why
            if class_name == 'Begin File':
                print(seg_file)
                break

            current_sample = int(int(byte) / (n_channel * byte_depth))

            if id_ not in open_ids:
                open_ids.append(id_)
                open_classes.append(class_name)
                if class_name == '':
                    print(seg_file)
                start_times.append(current_sample)
            else:
                target_id = open_ids.index(id_)

                start_samples.append(start_times[target_id])
                end_samples.append(current_sample)
                labels.append(open_classes[target_id])

                # конец открытого класса с заданным id
                if class_name == '':
                    del open_ids[target_id]
                    del open_classes[target_id]
                    del start_times[target_id]
                # конец открытого класса с заданным id и одновременно начало нового класса с тем же id
                else:
                    open_classes[target_id] = class_name
                    start_times[target_id] = current_sample

    return pd.DataFrame(list(zip(start_samples, end_samples, labels)), columns=['start_sample', 'end_sample', 'label'])


def prepare(remote_storage_path):
    # Тестовые файлы выбраны так, чтобы записи одних и тех же событий на разные микрофоны не попали одновременно
    # и в трейн и в тест
    test_files = [
        '2019-07-11_16-59-58_A-01-03_9374231091DB8AF4F2E32898862EB041_3.wav',
        '2019-07-11_16-59-58_A-01-01_BA94E858C1B1B722F9C27E7C71A5C5E3_3.wav',
        '2019-07-11_17-00-04_ReSpeaker_1562853604-01_3.wav',
        '2019-07-11_17-00-01_miniDSP_1562853601-01_3.wav',
    ]
    base_descr_yaml = {
        'data_path': 'prepared_data',
        'preprocessed_path': 'preprocessed',
        'feature_path': 'feature',
        'base_name': 'office-stc-1',
        'general_meta': 'meta.csv',
        'train_meta': 'meta_train.csv',
        'test_meta': 'meta_test.csv',
        'parent_bases': [],
    }
    # временное расположение тестовых баз
    remote_base_path = join(remote_storage_path, base_descr_yaml['base_name'])
    base_descr_file = join(remote_base_path, 'base_description.yml')
    if exists(base_descr_file):
        print('Base {} has been prepared.'.format(base_descr_yaml['base_name']))
        return True

    init_data_path = join(remote_base_path, 'initial_data')
    data_path = join(remote_base_path, base_descr_yaml['data_path'])

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

    os.mkdir(data_path)
    if not exists(join(remote_base_path, base_descr_yaml['preprocessed_path'])):
        os.mkdir(join(remote_base_path, base_descr_yaml['preprocessed_path']))

    if not exists(join(remote_base_path, base_descr_yaml['feature_path'])):
        os.mkdir(join(remote_base_path, base_descr_yaml['feature_path']))

    meta_id = 0
    target_labels = None

    # если нужно игнорировать какой-либо класс, то нужно использовать другой словарь!
    # этот только для начала и конца файла!
    special_labels = ['End File', 'Begin File']

    # эти классы игнорируем, так как они мало пердставлены
    ignore_list = ['sing', 'background', 'coffee-machine', 'laugh', 'human', 'furniture-moving', 'furniture',
                   'farfield-speech', 'non-standard']

    for wav_f_name_full, old_wavfile, seg_f_name_full in listdir_wav_seg(init_data_path):
        segments = parse_seg(seg_f_name_full)
        # отдельный столбец с подготовленными метками
        segments['new_label'] = segments['label'].apply(prep_label)
        for ign_file in ignore_list:
            segments = segments[segments.new_label != ign_file]

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
            wavfile.write(out_wav_name, sr, wav_data[start_sample: end_sample, :])
            upd_meta_lists(old_wavfile, f_name, old_label, label, base_descr_yaml['base_name'], start_sample / sr,
                           end_sample / sr, meta_id)
            meta_id += 1

    # make full meta
    meta_df = pd.DataFrame(list(zip(ids, init_names, cur_names, init_labels,
                                    cur_labels, bases, begins, ends)),
                           columns=['id', 'init_name', 'cur_name', 'init_label',
                                    'cur_label', 'database', 'begin', 'end'])
    meta_df.to_csv(meta_file_general, index=False, sep=';')

    # make train/test meta
    test_meta_df = meta_df[meta_df['init_name'] == test_files[0]]
    train_meta_df = meta_df[meta_df['init_name'] != test_files[0]]

    if len(test_files) > 1:
        for test_file in test_files[1:]:
            test_meta_df = test_meta_df.append(meta_df[meta_df['init_name'] == test_file])
            train_meta_df = train_meta_df[train_meta_df['init_name'] != test_file]

    test_meta_df.to_csv(meta_file_test, index=False, sep=';')
    train_meta_df.to_csv(meta_file_train, index=False, sep=';')

    # save base meta
    base_descr_yaml['events'] = target_labels
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_descr_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_descr_yaml['base_name']))
    return True


if __name__ == '__main__':
    print(1)
    # prepare(storage_path)
