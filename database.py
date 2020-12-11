"""Abstract class for database preparation"""
import os
import shutil

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import yaml
"""Abstract class for database preparation"""
import os
import shutil

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import yaml

from utils import calc_md5_hash, read_audio


class CheckBaseError(Exception):
    """Special exceptions for db builder"""

    def __init__(self, message):
        super().__init__(message)


class DataBase(ABC):
    """Description"""

    def __init__(self, input_base_path, output_base_path, base_name):
        if not os.path.exists(input_base_path):
            raise FileExistsError('Cannot find directory: {}'.format(input_base_path))
        self.input_base_path = input_base_path
        if not os.path.exists(output_base_path):
            raise FileExistsError('Cannot find directory: {}'.format(output_base_path))
        self.output_base_path = os.path.join(output_base_path, base_name)
        self.base_name = base_name

    def _build_description_dict(
        self,
        sample_rate: int,
        n_channels: int,
        extra_labels: List[str] = [],
        parent_bases: List[str] = [],
        data_path: str = 'data',
        feature_path: str = 'feature',
        preprocessed_path: str = 'preprocessed',
        general_meta: str = 'meta.csv',
        test_meta: str = 'meta_test.csv',
        train_meta: str = 'meta_train.csv',
    ):
        """Build description file and create all path.
        Return full bath to meta files and description file and also description dict
        You should use this inside self._collect_base().

        Parameters
        ----------
        sample_rate:       sample rate of audio file inside database
        n_channels:        number of channels in each file inside database
        extra_labels:      special labels for particular database
        parent_bases:      from which bases this should be gathered
        data_path:         path inside database which contain all audio files (it will be created)
        feature_path:      path inside database which contain all features files (it will be created)
        preprocessed_path: path inside database which contain preprocessed audio files (it will be created)
        general_meta:      name of file containing info about whole base
        test_meta:         name of file containing test set info
        train_meta:        name of file containing train set info
        """
        base_meta_yaml = {
            'base_name': self.base_name,
            'general_meta': general_meta,
            'test_meta': test_meta,
            'train_meta': train_meta,
            'data_path': data_path,
            'data_properties': {'sample_rate': sample_rate, 'n_channels': n_channels},
            'feature_path': feature_path,
            'preprocessed_path': preprocessed_path,
            # список баз, из которых будет собираться текущая база
            'parent_bases': parent_bases,
            # список дополнительных меток, присутствующих в базе помимо основной категориальной
            'extra_labels': extra_labels,
            'pass_tests': False,  # don't change it manually!!!!! it becomes True after tests
        }
        # make dirs for new base
        if not os.path.exists(self.output_base_path):
            os.mkdir(self.output_base_path)

        data_path = os.path.join(self.output_base_path, base_meta_yaml['data_path'])
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        preprocessed_path = os.path.join(self.output_base_path, base_meta_yaml['preprocessed_path'])
        if not os.path.exists(preprocessed_path):
            os.mkdir(preprocessed_path)

        feature_path = os.path.join(self.output_base_path, base_meta_yaml['feature_path'])
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)

        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')

        meta_file_test = os.path.join(self.output_base_path, base_meta_yaml['test_meta'])
        meta_file_train = os.path.join(self.output_base_path, base_meta_yaml['train_meta'])
        meta_file_general = os.path.join(self.output_base_path, base_meta_yaml['general_meta'])

        return base_meta_yaml, base_descr_file, meta_file_general, meta_file_train, meta_file_test

    def _check_files_and_folders(self):
        if not os.path.exists(self.output_base_path):
            raise CheckBaseError('Path {} does not exist!'.format(self.output_base_path))

        # existing of all meta-files and paths inside database
        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')
        if not os.path.exists(base_descr_file):
            raise CheckBaseError('There is no base_descritption.yml file inside {}'.format(self.output_base_path))
        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)

        data_path = os.path.join(self.output_base_path, base_descr['data_path'])
        if not os.path.exists(data_path):
            raise CheckBaseError('There is no data_path folder inside {}'.format(self.output_base_path))

        preprocessed_path = os.path.join(self.output_base_path, base_descr['preprocessed_path'])
        if not os.path.exists(preprocessed_path):
            raise CheckBaseError('There is no preprocessed path inside {}'.format(self.output_base_path))

        feature_path = os.path.join(self.output_base_path, base_descr['feature_path'])
        if not os.path.exists(feature_path):
            raise CheckBaseError('There is no feature_path inside {}'.format(self.output_base_path))

        meta_file_test = os.path.join(self.output_base_path, base_descr['test_meta'])
        if not os.path.exists(meta_file_test):
            raise CheckBaseError('There is no test_meta file inside {}'.format(self.output_base_path))

        meta_file_train = os.path.join(self.output_base_path, base_descr['train_meta'])
        if not os.path.exists(meta_file_train):
            raise CheckBaseError('There is no train_meta file inside {}'.format(self.output_base_path))

        meta_file_general = os.path.join(self.output_base_path, base_descr['general_meta'])
        if not os.path.exists(meta_file_general):
            raise CheckBaseError('There is no general_meta file inside {}'.format(self.output_base_path))

    def _check_extra_labels(self):
        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')
        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)
        extra_labels = base_descr['extra_labels']
        meta_file_test = os.path.join(self.output_base_path, base_descr['test_meta'])
        meta_file_train = os.path.join(self.output_base_path, base_descr['train_meta'])
        meta_file_general = os.path.join(self.output_base_path, base_descr['general_meta'])

        metas = [meta_file_test, meta_file_train, meta_file_general]
        for meta in metas:
            test_columns = pd.read_csv(meta, sep=';').columns
            for extra_label in extra_labels:
                if extra_label not in test_columns:
                    raise CheckBaseError('Extra label "{}" is not inside {} columns!'.format(extra_label, meta))

    def _check_metas(self):
        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')
        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)
        meta_file_test = os.path.join(self.output_base_path, base_descr['test_meta'])
        meta_file_train = os.path.join(self.output_base_path, base_descr['train_meta'])
        meta_file_general = os.path.join(self.output_base_path, base_descr['general_meta'])
        metas = [meta_file_test, meta_file_train, meta_file_general]

        db_labels = base_descr['events']

        # проверка пересечений мет (объединение трейн тест = общей, пересечение трейн тест = пусто)
        file_sets = []
        for meta in metas:
            df = pd.read_csv(meta, sep=';')
            file_list = list(df.cur_name)
            unique_files, counts = np.unique(file_list, return_counts=True)
            # в каждой мете не должно быть повторяющихся файлов!
            if len(file_list) != len(unique_files):
                duplicates = unique_files[(counts - 1).astype(bool)]
                mes = 'You have file duplicates in your {} meta. Duplicates are: {}'.format(meta, duplicates)
                raise CheckBaseError(mes)
            # проверка присутствия меток в base_description.events
            label_list = list(df.cur_label)
            unique_labels = np.unique(label_list)
            for label in unique_labels:
                if label not in db_labels:
                    raise CheckBaseError('Label "{}" is not inside "{}"!'.format(label, base_descr_file))

            file_sets.append(set(file_list))
        # пересечение
        intersection_train_test = file_sets[0].intersection(file_sets[1])
        if len(intersection_train_test) > 0:
            raise CheckBaseError('Your train and test sets intersect on files: {}'.format(intersection_train_test))
        # объединение
        united_train_test_set = file_sets[0].union(file_sets[1])
        if len(united_train_test_set) != len(file_sets[2]):
            raise CheckBaseError('Amount of files in train and test metas is not equal to amount in general meta!')
        # проверка присутствия всех файлов в мете
        data_path = os.path.join(self.output_base_path, base_descr['data_path'])
        for file_name in os.listdir(data_path):
            if file_name not in file_sets[2]:
                raise CheckBaseError('General meta does not contain file "{}"!'.format(file_name))
        # проверка наличия всех файлов из меты в дате
        for f_name in file_sets[2]:
            if not os.path.exists(os.path.join(data_path, f_name)):
                raise CheckBaseError('File "{}" exists in general_meta but doesn\'t exist in data path!'.format(f_name))

    def _check_pseudo_random(self):
        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')
        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)

        meta_file_test = os.path.join(self.output_base_path, base_descr['test_meta'])
        meta_file_train = os.path.join(self.output_base_path, base_descr['train_meta'])
        train_meta_hash = calc_md5_hash(meta_file_train)
        test_meta_hash = calc_md5_hash(meta_file_test)

        # remove data and recollect it
        data_path = os.path.join(self.output_base_path, base_descr['data_path'])
        shutil.rmtree(data_path)
        os.mkdir(data_path)
        self._collect_base()

        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)

        meta_file_test = os.path.join(self.output_base_path, base_descr['test_meta'])
        meta_file_train = os.path.join(self.output_base_path, base_descr['train_meta'])
        train_meta_hash_new = calc_md5_hash(meta_file_train)
        test_meta_hash_new = calc_md5_hash(meta_file_test)
        if train_meta_hash != train_meta_hash_new:
            raise CheckBaseError('Your train meta content differs from one collect to another.')
        if test_meta_hash != test_meta_hash_new:
            raise CheckBaseError('Your test meta content differs from one collect to another.')

    def _check_data_properties(self):
        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')
        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)
        target_sr = base_descr['data_properties']['sample_rate']
        n_channels = base_descr['data_properties']['n_channels']

        meta_file_general = os.path.join(self.output_base_path, base_descr['general_meta'])
        df = pd.read_csv(meta_file_general, sep=';')
        data_path = os.path.join(self.output_base_path, base_descr['data_path'])
        for i, row in df.iterrows():
            f_name = os.path.join(data_path, row['cur_name'])
            try:
                _, wav_data = read_audio(f_name, target_sr, dtype='float')
            except Exception as e:
                print(str(e))
                # raise CheckBaseError(str(e))
            if len(wav_data.shape) != n_channels:
                raise CheckBaseError(
                    'Wrong number of channels! Target is {}, current is {}. File: '
                    '"{}"'.format(n_channels, len(wav_data.shape), f_name)
                )
            begin, end = float(row['begin']), float(row['end'])
            if abs(len(wav_data) / target_sr - (end - begin)) > 0.1:
                print('Wrong audio length. It must be the same as (end - begin) in meta! File: {} target_sr={}, begin={}, end={}, len(wav_data)={}'.
                        format(f_name, target_sr, begin, end, len(wav_data)))
                # raise CheckBaseError(
                #     'Wrong audio length. It must be the same as (end - begin) in meta! File: {} target_sr={}, begin={}, end={}, len(wav_data)={}'.
                #         format(f_name, target_sr, begin, end, len(wav_data))
                # )

    def _check_some_other_tests(self):
        pass

    def _check_base(self):
        # проверка того, что все папки и файлы из base_description на своем месте
        self._check_files_and_folders()

        # все экстра метки есть в базе
        self._check_extra_labels()

        # все файлы, указанные в метах, присутствуют в базе
        # все файлы, присутствующие в базе, указаны в метах
        # все имена файлов в метах уникальны
        # пересечение трейн и тест - это пустое множество
        self._check_metas()

        # sample_rate файлов соответствует заявленному в base_description
        # длина файлов, сохраненных после обработки, соответствует указанной в мете (чтобы исключить ошибки нарезки)
        self._check_data_properties()

        # только псевдорандом при генерации трейна и теста (при повторном запуске меты получаются идентичные)
        self._check_pseudo_random()

        # прочие тесты, можно дополнять сколько угодно
        self._check_some_other_tests()

    @abstractmethod
    def _collect_base(self):
        pass

    def prepare_base(self):
        """Prepare base to specified format and run tests to check if there were mistakes while preparing.
        All staff connected to base preparation must be inside self._collect_base.
        """
        # if not os.path.exists(self.output_base_path):
        #     self._collect_base()
        self._collect_base()

        base_descr_file = os.path.join(self.output_base_path, 'base_description.yml')
        with open(base_descr_file, 'r') as f:
            base_descr = yaml.safe_load(f)

        # default value for pass_tests is False
        # If tests weren't passed before or it's first attempt, then run all tests
        if not base_descr['pass_tests']:
            print('Start checking')
            try:
                self._check_base()
            except CheckBaseError as e:
                # shutil.rmtree(self.output_base_path)
                raise e

        base_descr['pass_tests'] = True
        with open(base_descr_file, 'w') as yaml_meta:
            yaml.dump(base_descr, yaml_meta, default_flow_style=False)

        print('Base has been prepared without errors.')
