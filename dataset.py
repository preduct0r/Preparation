"""
Класс для работы с csv-файлами.
Имплементирует именованный numpy.array
"""


import numpy
import os
import yaml
import pandas as pd

# разделитель в мета файле
DELIMETER = ';'
ALIAS_FOR_META_FILE_IN_CONFIG = 'data_file'
ALIAS_FOR_TEST_META_FILE_IN_CONFIG = 'test_data_file'


class DatasetNew(dict):
    def __init__(self, config):
        self.base_meta_file = os.path.join(config['base_path'], 'base_description.yml')
        with open(self.base_meta_file, 'r') as f:
            base_meta = yaml.safe_load(f)
        # делаем директории абсолютными
        base_meta['data_path'] = os.path.join(config['base_path'], base_meta['data_path'])
        base_meta['feature_path'] = os.path.join(config['base_path'], base_meta['feature_path'])
        base_meta['preprocessed_path'] = os.path.join(config['base_path'], base_meta['preprocessed_path'])

        # загрузили инфу о датасете, можем пользоваться как словарем теперь
        super(DatasetNew, self).__init__(**base_meta)

        # в датасете прогружена вся дата
        self.train_data = pd.read_csv(os.path.join(config['base_path'], base_meta['train_meta']), sep=';')
        self.test_data = pd.read_csv(os.path.join(config['base_path'], base_meta['test_meta']), sep=';')
        self.general_data = pd.read_csv(os.path.join(config['base_path'], base_meta['general_meta']), sep=';')


class Dataset(numpy.ndarray):

    def __new__(cls, config):
        """
        :param config: объект класса `Config`
        :return Numpy Structured Array
        """
        assert isinstance(config, dict), \
            'Config is not a dictionary or Config: {}'.format(config)
        path_ = config[ALIAS_FOR_META_FILE_IN_CONFIG]
        assert os.path.isfile(path_), \
            'No meta-file for dataset found: {}'.format(path_)
        # Dataset здесь становится `numpy.array`
        dataset = numpy.genfromtxt(path_, delimiter=DELIMETER, dtype=None,
                                   names=True, encoding='utf-8', autostrip=True)
        return dataset
