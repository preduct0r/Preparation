from pipeline.pipe_modules.config import Config
from pipeline.pipe_modules.dataset import DatasetNew
from pipeline.pipe_modules.featureextractor import FeatureExtractorPickle


def main():
    # прогрузить конфигурационный файл
    config = Config('experiments/emo/pre_sff_config.yml')

    dataset = DatasetNew(config)

    # запускаем генератор фич
    ext = FeatureExtractorPickle(dataset, config)

    # примеры доступных фич указаны в функции extract файла featureextractor.py
    ext.extract()


if __name__ == "__main__":
    main()
