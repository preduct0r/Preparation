from database import DataBase
import pandas as pd
import os
import math
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
                    extra_labels = ["arousal","valence","cheerful","happy","angry","nervous","sad","arvalmix"])


        audio_path = os.path.join(self.input_base_path, 'debate_audios')
        annotations_path = os.path.join(self.input_base_path, 'emotion_annotations', 'self_annotations')
        target_sample_rate = base_meta_yaml['data_properties']['sample_rate']
        target_n_channels = base_meta_yaml['data_properties']['n_channels']
        dictors_time = os.path.join(self.input_base_path, 'additional', 'dictors_time_sec')

        # имена столбцов в будущей разметке
        col_names = [
                        'ids',
                        'init_name',
                        'cur_name',
                        'speaker',
                        'cur_label',
                        'database',
                        'begin',
                        'end',
                    ] + base_meta_yaml['extra_labels']


        labels_counter = {
            "angry": 0,
            "sad": 0,
            "happy": 0,
            "cheerful": 0,
            "nervous": 0,
            "neutral": 0
        }
        # теперь уже начинаем обрабатывать нашу базу
        file_id = 0
        # подсчет, сколько примеров каждого класса есть в базе
        train_ids, test_ids = [], []
        raw_meta_data = []

        def class_define(arousal, valence):
            dict_change = {1:-1, 2:-1, 3:0, 4:1, 5:1}
            ar, val = dict_change[int(arousal)], dict_change[int(valence)]
            dict_class = {(-1,-1):0, (-1,0):1, (0,-1):3, (-1,1):2, (0,0):4, (0,1):5, (1,1):8, (1,0):7, (1,-1):6}
            return dict_class[(ar, val)]


        for emo_file in glob(os.path.join(annotations_path, "*")):
            name = os.path.basename(emo_file)[:-9].lower() + '.'
            wavka = ''
            for wav in os.listdir(audio_path):
                if name in wav:
                    wavka = wav
                    break

            time_df = pd.read_csv(os.path.join(dictors_time, name+'csv'), sep=',')
            if int(name[1:-1])%2 == 1:
                time_df = time_df.iloc[:,:2]
            else:
                time_df = time_df.iloc[:,2:]

            possible_values = []

            for idx, row in time_df.iterrows():
                if math.isnan(row[0])==False:
                    for i in range(int(row[0]), int(row[1])):
                        if i%5==0:
                            possible_values.append(i)



            emo_info = pd.read_csv(emo_file, sep=',')                       # csvка с мультилейблингом файла

            df = pd.DataFrame(data=0, index=emo_info['seconds'], columns=base_meta_yaml['extra_labels'])

            for idx, row in emo_info.iterrows():
                df.loc[row[0],:] = [row[1], row[2], row[3], row[4], row[5], row[6], row[7], class_define(row[1], row[2])]

            possible_values = list(set(possible_values) & set(df.index))
            df = df.loc[possible_values, :]

            # ============================================================================================


            # определим лейбл, который попадет в cur_label и init_label
            df['curr_label'] = np.nan
            for idx, row in df.loc[:,["cheerful","happy","angry","nervous","sad"]].iterrows():
                idx_max = row.idxmax()
                val_counts = row.value_counts()
                if val_counts.loc[row.max()]>1:
                    curr_label = 'neutral'
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
                # print()
                raw_meta_data.append([file_id, wavka, new_wav_name, name[:-1], curr_label, self.base_name, start, end] +
                                list(df.loc[start, base_meta_yaml['extra_labels']]))

                subprocess.call(ffmpeg_command)


                # пригодится, чтобы создать meta_train, meta_test
                if name.startswith('p29') or name.startswith('p31'):
                    test_ids.append(file_id)
                else:
                    train_ids.append(file_id)


                # апгрейдим инфу
                file_id += 1
                labels_counter[curr_label] += 1

                return file_id

            # =========================================================================================


            # исполняем предыдущую функцию
            for i in np.arange(0, len(df.index)-2):
                # try:

                if len(wav_data)/float(sr) > df.index[i] + self.window_size:
                    file_id = cut_file(df.index[i], df.index[i] + self.window_size, file_id)
                else:
                    break
                # except Exception as e:
                #     print('Exception from {} to {}'.format(df.index[i], df.index[i] + self.window_size))

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
