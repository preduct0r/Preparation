# import os
# import yaml
# import fnmatch
#
# import numpy as np
# import pandas as pd
# from scipy.io import wavfile
# import subprocess
# from sklearn.model_selection import train_test_split
# import pickle
# import glob
# import tempfile
# import shutil
# import re
# from skvideo.io import FFmpegReader, ffprobe


def get_fps(video_path):
    vinfo = ffprobe(video_path)['video']
    fps = vinfo['@avg_frame_rate']
    fps = fps.split('/')
    fps = int(fps[0]) / int(fps[1])

    return fps

def prepare(input_base_path, base_path, data_about_sep, train_annotation_path, test_annotation_path):
    target_sample_rate = 8000
    videos_path = input_base_path + r'\video'


    # inside base_descr_yaml
    base_meta_yaml = {
        'base_name': 'aff_beh',
        'general_meta': 'meta.csv',
        'test_meta': 'meta_test.csv',
        'train_meta': 'meta_train.csv',
        'data_path': 'data',
        'feature_path': 'feature',
        'preprocessed_path': 'prepocessed',
        # список баз, из которых будет собираться текущая база
        'parent_bases': [],
        'extra_labels': []
    }
    # создание всех папок, упомянутых в base_meta_yaml
    output_base_path = os.path.join(base_path, base_meta_yaml['base_name'])

    # make dirs for new base
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    output_data_path = os.path.join(output_base_path, base_meta_yaml['data_path'])
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    output_audio_path = os.path.join(output_base_path, 'audio')
    if not os.path.exists(output_audio_path):
        os.mkdir(output_audio_path)

    preprocessed_path = os.path.join(output_base_path, base_meta_yaml['preprocessed_path'])
    if not os.path.exists(preprocessed_path):
        os.mkdir(preprocessed_path)

    feature_path = os.path.join(output_base_path, base_meta_yaml['feature_path'])
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)

    base_descr_file = os.path.join(output_base_path, 'base_description.yml')

    # имена столбцов в будущей разметке
    col_names = ['ids', 'init_name', 'cur_name', 'init_label', 'cur_label',
                 'database', 'begin', 'end'] + base_meta_yaml['extra_labels']
    # списки значений для каждого из столбцов
    ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends= [], [], [], [], [], [], [], []

    # пути к метам
    meta_file_test = os.path.join(output_base_path, base_meta_yaml['test_meta'])
    meta_file_train = os.path.join(output_base_path, base_meta_yaml['train_meta'])
    meta_file_general = os.path.join(output_base_path, base_meta_yaml['general_meta'])

    # теперь уже начинаем обрабатывать нашу базу
    file_id = 0
    # подсчет, сколько примеров каждого класса есть в базе
    labels_counter = {"anger":0, "neutrality":0, "sadness":0, "happiness":0, "fear": 0, 'surprise':0, "disgust":0}
    dict_emo = {0: "neutrality", 1: 'anger', 2: "disgust", 3: "fear", 4: 'happiness', 5: 'sadness', 6: 'surprise'}
    for annotation_path, meta_name in zip([test_annotation_path, train_annotation_path],['test','train']):
        for annotation_name in os.listdir(annotation_path):

            if (re.findall('_right', annotation_name)):
                video_path = os.path.join(videos_path, annotation_name[:-10] +'.mp4')
                audio_path = os.path.join(output_audio_path, annotation_name[:-10] +'.wav')
            elif (re.findall('_left', annotation_name)):
                video_path = os.path.join(videos_path, annotation_name[:-9] + '.mp4')
                audio_path = os.path.join(output_audio_path, annotation_name[:-9] + '.wav')
            else:
                video_path = os.path.join(videos_path, annotation_name[:-4] + '.mp4')
                audio_path = os.path.join(output_audio_path, annotation_name[:-4] + '.wav')

            if not os.path.exists(video_path):
                video_path = video_path[:-4]+'.avi'

            fps = round(get_fps(video_path))

            if not os.path.exists(audio_path):
                ffmpeg_command = r'ffmpeg -i {} -c:a pcm_s16le -ar {} -ac 1 {} -loglevel panic'.format(
                    video_path,
                    target_sample_rate,
                    audio_path
                )
                subprocess.call(ffmpeg_command)

            sr, wav_data = wavfile.read(audio_path)
            assert target_sample_rate==sr

            for idx, row in data_about_sep[data_about_sep['f_name']==(os.path.join(annotation_path, annotation_name))].iterrows():
            # Index(['label', 'from', 'to', 'frames', 'seconds', 'f_name'], dtype='object')
                label = row[0]
                labels_counter[dict_emo[label]] += 1

                # новое имя: класс база счетчик
                new_wav_name = '{}_{}_{}_{}.wav'.format(dict_emo[label], meta_name, base_meta_yaml['base_name'], labels_counter[dict_emo[label]])
                # полный путь до новой вавки
                new_wav_name_full = os.path.join(output_base_path, base_meta_yaml['data_path'], new_wav_name)

                from_ = int(row[1]/fps*sr)
                to_ = int(row[2]/fps*sr)

                wavfile.write(new_wav_name_full, sr, wav_data[from_: to_])


                start = from_
                end = to_
                ids.append(file_id)
                file_id += 1
                init_names.append(os.path.basename(video_path))
                cur_names.append(new_wav_name)
                init_labels.append(label)
                cur_labels.append(dict_emo[label])
                dbs.append(base_meta_yaml['base_name'])
                begs.append(start)
                ends.append(end)

        # создаем датафрейм
        meta_df = pd.DataFrame(
            list(zip(ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends)),
            columns=col_names)

        # сохраняем
        if meta_name == 'train':
            meta_df.to_csv(meta_file_train, index=False, sep=';')
        elif meta_name == 'test':
            meta_df.to_csv(meta_file_test, index=False, sep=';')

        else:
            raise Exception('Some mistake occured with meta saving')

        # обнуляем значения
        ids, init_names, cur_names, init_labels, cur_labels, dbs, begs, ends = [], [], [], [], [], [], [], []

    # сохранение общей меты
    train_meta_df = pd.read_csv(meta_file_train, sep=';')
    test_meta_df = pd.read_csv(meta_file_test, sep=';')
    # (pd.concat([train_meta_df, test_meta_df])).to_csv(meta_file_general, index=False)
    (train_meta_df.append(test_meta_df)).to_csv(meta_file_general, index=False, sep=';')

    # добавляем посчитанные события в макро описание базы и сохраняем описание
    base_meta_yaml['events'] = labels_counter
    with open(base_descr_file, 'w') as yaml_meta:
        yaml.dump(base_meta_yaml, yaml_meta, default_flow_style=False)

    print('Base {} has been prepared.'.format(base_meta_yaml['base_name']))



if __name__ == '__main__':
    # путь к базе, которую хотим подготовить
    input_base_path = r'C:\Users\kotov-d\Documents\Aff_Beh'

    # именно здесь появится папка с новой подготовленной базой
    output_base_path = r'C:\Users\kotov-d\Documents\aff_preprocess'

    # train_annotation_path = os.path.join(input_base_path,  'annotations', 'EXPR_Set', 'Training_Set')
    train_annotation_path = input_base_path + r'\annotations\EXPR_Set\Training_Set'
    test_annotation_path = input_base_path + r'\annotations\EXPR_Set\Validation_Set'

    with open(r'C:\Users\kotov-d\Documents\aff_preprocess\data_about_sep.pkl', 'rb') as f:
        data_about_sep = pickle.load(f)

    # print(data_about_sep.columns)



    prepare(input_base_path, output_base_path, data_about_sep, train_annotation_path, test_annotation_path)


