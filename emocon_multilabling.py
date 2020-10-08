from database import DataBase
import pandas as pd
import os
import pickle
import fnmatch
from scipy.io import wavfile
import subprocess
import yaml
from sklearn.model_selection import train_test_split
from glob import glob
import soundfile
import numpy as np
from math import modf


class Emocon_mult(DataBase):
    def __init__(self, input_base_path, output_base_path, window_size=5):
        super().__init__(input_base_path, output_base_path, 'emocon_mult')
        self.window_size = window_size

    def _collect_base(self):
        (
            base_meta_yaml,
            base_descr_file,
            meta_file_general,
            meta_file_train,
            meta_file_test,
        ) = self._build_description_dict(sample_rate=8000, n_channels=1,
                    extra_labels = ['arousal','valence',"happiness","anger","sadness", "surprise", "disgust"])


        audio_path = os.path.join(self.input_base_path, 'debate_audios')
        annotations_path = os.path.join(self.input_base_path, 'additional', 'new_annotations')
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


        labels_counter = {
            "anger": 0,
            "neutrality": 0,
            "sadness": 0,
            "happiness": 0,
            "disgust": 0,
            "surprise": 0,
        }
        # теперь уже начинаем обрабатывать нашу базу
        file_id = 0
        # подсчет, сколько примеров каждого класса есть в базе
        train_ids, test_ids = [], []
        raw_meta_data = []


        for emo_file in glob(os.path.join(annotations_path, "*")):
            file_name = os.path.basename(emo_file)[:-4]
            wavka = ''
            for wav in os.listdir(audio_path):
                if file_name in wav:
                    wavka = wav
                    break


            emo_info = pd.read_csv(emo_file, sep=',')                       # csvка с мультилейблингом файла

            df = pd.DataFrame(data=0, index=emo_info['seconds'], columns=base_meta_yaml['extra_labels'])

            for idx, row in emo_info.iterrows(): #'arousal','valence',"happiness","anger","sadness", "surprise", "disgust"
                df.loc[row[0],:] = [row[1], row[2], row[4], row[5], row[7], row[13], row[18]]
            # ============================================================================================


            # определим лейбл, который попадет в cur_label и init_label
            df['curr_label'] = np.nan
            for idx, row in df.iterrows():
                idx_max = row.iloc[3:8].idxmax()
                val_counts = row.iloc[:-1].value_counts()
                if val_counts.loc[row.iloc[:-1].max()]>1:
                    curr_label = 'neutrality'
                else:
                    curr_label = idx_max
                df.loc[idx, 'curr_label'] = curr_label
            #=============================================================================================


            file_path = os.path.join(audio_path, wavka)
            wav_data, sr = soundfile.read(file_path)


            # нарезка и формирование разметки
            def cut_file(start,end, file_id):
                # новое имя: класс база счетчик
                new_wav_name = '{}_{}_{}.wav'.format(
                    curr_label, base_meta_yaml['base_name'], labels_counter[curr_label]
                )
                # полный путь до новой вавки
                new_wav_name_path = os.path.join(self.output_base_path, base_meta_yaml['data_path'], new_wav_name)

                ffmpeg_command = (
                    r'ffmpeg -ss {} -t {} -i {} -c:a '
                    r'pcm_s16le -ar {} -ac {} {} -loglevel panic'.format(
                        start,
                        (end - start),
                        file_path,
                        target_sample_rate,
                        target_n_channels,
                        new_wav_name_path,
                    )
                )
                raw_meta_data.append([file_id, file_name, new_wav_name, curr_label, curr_label, self.base_name, start, end] +
                                list(np.mean(df.loc[start:end, base_meta_yaml['extra_labels']], axis=0)))
                subprocess.call(ffmpeg_command)


                # пригодится, чтобы создать meta_train, meta_test
                if file_name.startswith('p29') or file_name.startswith('p31'):
                    test_ids.append(file_id)
                else:
                    train_ids.append(file_id)


                # апгрейдим инфу
                file_id += 1
                labels_counter[curr_label] += 1

                return file_id

            # =========================================================================================


            # исполняем предыдущую функцию
            for i in np.arange(0, len(df.index), self.window_size):
                try:
                    file_id = cut_file(df.index[i], df.index[i] + self.window_size, file_id)
                except Exception as e:
                    pass

            # =========================================================================================

        # создаем датафрейм
        meta_df = pd.DataFrame(data=raw_meta_data, columns=col_names)

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
    # путь к Audio данным из папки Data базы Ramaz
    input_base_path = r'C:\Users\preductor\Google_Drive\Ulma'

    # именно здесь появится папка с новой подготовленной базой
    output_base_path = r'C:\Users\preductor\Documents'

    db = Emocon_mult(input_base_path=input_base_path, output_base_path=output_base_path, window_size=5)

    db.prepare_base()
