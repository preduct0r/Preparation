"""
Класс для работы с файлом конфигурации
Конфигурация должна храниться в формате YAML.
В процессе работы объкет данного класса (с загруженной конфигурацией) передаётся на вход другим этапам пайплайна.
По-умолчанию yaml-файл хранится в директории config.
Config имплементирует класс dictionary питона
"""


import os
import yaml


class Config(dict):

    def __init__(self, path_to_configfile):
        """
        :param path_to_configfile: строка, путь до файла с конфигом
        :return None
        """
        assert os.path.isfile(path_to_configfile), \
            'No configuration file found: {}'.format(path_to_configfile)
        with open(path_to_configfile, "r") as file:
            self.config_ = yaml.safe_load(file)
        # на этом этапе сам объект конфигурации становится словарем
        super(Config, self).__init__(**self.config_)
