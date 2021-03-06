"""Describes class to collect RAMAS database"""
import os
import subprocess

from glob import glob

import pandas as pd
import yaml

from database import DataBase
from soundfile import read


class Ramas(DataBase):
    """Ramas collector"""

    def _collect_base(self):
        (
            base_meta_yaml,
            base_descr_file,
            meta_file_general,
            meta_file_train,
            meta_file_test,
        ) = self._build_description_dict(sample_rate=8000, n_channels=1,)
        # так нельзя!
        # нужно, чтобы объект класса создавался для любого пользователя!!!!
        #
        audio_path = os.path.join(self.input_base_path, 'Data', 'Audio')
        annotations_path = os.path.join(self.input_base_path, 'Annotations_by_emotions')
        target_sample_rate = base_meta_yaml['data_properties']['sample_rate']
        target_n_channels = base_meta_yaml['data_properties']['n_channels']

        # имена столбцов в будущей разметке
        col_names = [
            'ids',
            'init_name',
            'cur_name',
            'init_label',
            'cur_label',
            'database',
            'begin',
            'end',
        ] + base_meta_yaml['extra_labels']
        # списки значений для каждого из столбцов
        ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends = [], [], [], [], [], [], [], []

        # теперь уже начинаем обрабатывать нашу базу
        file_id = 0
        # подсчет, сколько примеров каждого класса есть в базе
        train_ids, test_ids = [], []

        dict_emo = {
            "Angry": "anger",
            "Disgusted": "disgust",
            "Happy": "happiness",
            "Neutral": "neutrality",
            "Sad": "sadness",
            "Scared": "fear",
            "Surprised": "surprise",
        }
        labels_counter = {
            "anger": 0,
            "neutrality": 0,
            "sadness": 0,
            "happiness": 0,
            "disgust": 0,
            "surprise": 0,
            "fear": 0,
        }
        for emo_file in glob(os.path.join(annotations_path, "*")):
            label = os.path.basename(emo_file)[5:-4]
            if label in dict_emo.keys():
                emo_info = pd.read_csv(emo_file)
                for idx, row in emo_info.iterrows():
                    raw_file_name, start, end = row[1], row[2], row[3]

                    file_name = raw_file_name + "_mic.wav"
                    tmp = os.path.join(audio_path, file_name)
                    wav_data, sr = read(tmp)
                    if len(wav_data[int(sr * float(start)) : int(sr * float(end))]) < 1.5 * sr:
                        continue

                    # новое имя: класс база счетчик
                    new_wav_name = '{}_{}_{}.wav'.format(
                        dict_emo[label], base_meta_yaml['base_name'], labels_counter[dict_emo[label]]
                    )
                    # полный путь до новой вавки
                    new_wav_name_full = os.path.join(self.output_base_path, base_meta_yaml['data_path'], new_wav_name)

                    ffmpeg_command = (
                        r'ffmpeg -ss {} -t {} -i {} -c:a '
                        r'pcm_s16le -ar {} -ac {} {} -loglevel panic'.format(
                            start,
                            end - start,
                            os.path.join(audio_path, file_name),
                            target_sample_rate,
                            target_n_channels,
                            new_wav_name_full,
                        )
                    )
                    subprocess.call(ffmpeg_command, shell=True)

                    if file_name.startswith('22dec'):
                        test_ids.append(file_id)
                    else:
                        train_ids.append(file_id)

                    ids.append(file_id)
                    file_id += 1
                    labels_counter[dict_emo[label]] += 1
                    init_names.append(file_name)
                    cur_names.append(new_wav_name)
                    init_labels.append(label)
                    cur_labels.append(dict_emo[label])
                    dbs.append(base_meta_yaml['base_name'])
                    begs.append(start)
                    ends.append(end)

        # создаем датафрейм
        meta_df = pd.DataFrame(
            list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends)), columns=col_names
        )

        # сохраняем
        meta_df.to_csv(meta_file_general, index=False, sep=';')
        meta_df[meta_df['ids'].isin(train_ids)].to_csv(meta_file_train, index=False, sep=';')
        meta_df[meta_df['ids'].isin(test_ids)].to_csv(meta_file_test, index=False, sep=';')

        # добавляем посчитанные события в макро описание базы и сохраняем описание
        base_meta_yaml['events'] = labels_counter
        with open(base_descr_file, 'w') as yaml_meta:
            yaml.dump(base_meta_yaml, yaml_meta, default_flow_style=False)

        print('Base {} has been prepared.'.format(base_meta_yaml['base_name']))


if __name__ == '__main__':
    # путь к базе!!!!
    input_base_path = '/home/den/datasets/ramaz_audio_data'

    # именно здесь появится папка с новой подготовленной базой
    output_base_path = '/home/den/datasets'

    db = Ramas(input_base_path, output_base_path, 'ramas')

    db.prepare_base()
