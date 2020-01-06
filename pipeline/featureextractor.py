"""
Экстрактор признаков (фич) из аудиофайлов.
Поддерживает препроцессинг – выравнивание по длине, выделение каналов (левый, правый, middle и side),
а также выделение признаков – raw (не извлекать признаки, вернуть набор амплитуд), fft, mel, mfcc.
"""


import h5py
import librosa
from librosa.core import stft
from librosa.util import buf_to_float
import numpy as np
import os
from scipy import signal
from scipy.io import wavfile
# from tqdm import tqdm
import sys
import pickle
import soundfile
sys.path.append('..')

from experiments.emo.sff_m import get_sff


META_FILE_NAME = 'meta.csv'
HDF_FILE_EXTENTION = '.hdf5'


"""
TODO
- название мета файла в config.yml
- брать параметры экстракторов из config.yml 
- не экстрактить фичи, если они есть
"""


class FeatureExtractorPickle:
	def __init__(self, dataset, config):
		"""
		:param dataset: объект класса `Dataset` с загруженным csv-файлом датасета
		:param config: объект класса `Config` с загруженным файлом конфигурации
		"""
		self.dataset = dataset
		self.config = config

	def _build_multilabel(self, row):
		# если есть дополнительные метки, то возвращаем основную вместе с ними списком
		if 'extra_labels' in self.dataset:
			multi_label = [row['cur_label']]
			for extra_label in self.dataset['extra_labels']:
				multi_label.append(row[extra_label])
			return multi_label
		# если их нет, то просто список из одной основной метки
		return [row['cur_label']]

	@staticmethod
	def _save_pickles(feature_path, x_train, y_train, f_train, x_test, y_test, f_test):
		with open(os.path.join(feature_path, 'x_train.pkl'), 'wb') as f:
			pickle.dump(x_train, f)
		with open(os.path.join(feature_path, 'y_train.pkl'), 'wb') as f:
			pickle.dump(y_train, f)
		with open(os.path.join(feature_path, 'f_train.pkl'), 'wb') as f:
			pickle.dump(f_train, f)

		with open(os.path.join(feature_path, 'x_test.pkl'), 'wb') as f:
			pickle.dump(x_test, f)
		with open(os.path.join(feature_path, 'y_test.pkl'), 'wb') as f:
			pickle.dump(y_test, f)
		with open(os.path.join(feature_path, 'f_test.pkl'), 'wb') as f:
			pickle.dump(f_test, f)

	def _vggish1(self):
		from models.vggish import VGGISH_CHECKPOINT_PATH, VGGISH_PCA_PARAMS_PATH
		import tensorflow as tf
		from models.vggish import vggish_input
		from models.vggish import vggish_params
		from models.vggish import vggish_postprocess
		from models.vggish import vggish_slim

		x_train, y_train, f_train, x_test, y_test, f_test = [], [], [], [], [], []

		feature_path = os.path.join(self.dataset['feature_path'], 'vggish1')
		if not os.path.exists(feature_path):
			os.mkdir(feature_path)

		# firstly calculate mels
		train_mels = []
		train_mels_labels = []
		train_mels_f = []
		for i, row in self.dataset.train_data.iterrows():
			print('[Train] {}) Getting mels from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)
			train_mels.append(vggish_input.wavdata_to_examples(wav_data))
			train_mels_labels.append(self._build_multilabel(row))
			train_mels_f.append(row['cur_name'])
			# если размер вавки позволяет, то строим эмбединг начиная с 0.5 секунды
			if len(wav_data) > 1.5 * sr:
				train_mels.append(vggish_input.wavdata_to_examples(wav_data[sr // 2:]))
				train_mels_labels.append(self._build_multilabel(row))
				train_mels_f.append(row['cur_name'])
			print('done.')

		test_mels = []
		test_mels_labels = []
		test_mels_f = []
		for i, row in self.dataset.test_data.iterrows():
			print('[Test] {}) Getting mels from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)
			test_mels.append(vggish_input.wavdata_to_examples(wav_data))
			test_mels_labels.append(self._build_multilabel(row))
			test_mels_f.append(row['cur_name'])
			if len(wav_data) > 1.5 * sr:
				test_mels.append(vggish_input.wavdata_to_examples(wav_data[sr // 2:]))
				test_mels_labels.append(self._build_multilabel(row))
				test_mels_f.append(row['cur_name'])

			print('done')

		# now calc embeddings in one single grapth and session
		# Prepare a postprocessor to munge the model embeddings.
		pproc = vggish_postprocess.Postprocessor(VGGISH_PCA_PARAMS_PATH)

		with tf.Graph().as_default(), tf.Session() as sess:
			# Define the model in inference mode, load the checkpoint, and
			# locate input and output tensors.
			vggish_slim.define_vggish_slim(training=False)
			vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_CHECKPOINT_PATH)
			features_tensor = sess.graph.get_tensor_by_name(
				vggish_params.INPUT_TENSOR_NAME)
			embedding_tensor = sess.graph.get_tensor_by_name(
				vggish_params.OUTPUT_TENSOR_NAME)

			for i, mels in enumerate(train_mels):
				print('[Train] {}) Getting embeddings...'.format(i), end='')
				# Run inference and postprocessing.
				try:
					[embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: mels})
				except tf.errors.InvalidArgumentError:
					raise ValueError("Audio duration should be equal or more than 1")

				ready_embs = pproc.postprocess(embedding_batch)
				for emb in ready_embs:
					x_train.append(emb)
					y_train.append(train_mels_labels[i])
					f_train.append(train_mels_f[i])
				print('done.')
			for i, mels in enumerate(test_mels):
				print('[Test] {}) Getting embeddings...'.format(i), end='')
				# Run inference and postprocessing.
				try:
					[embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: mels})
				except tf.errors.InvalidArgumentError:
					raise ValueError("Audio duration should be equal or more than 1")

				ready_embs = pproc.postprocess(embedding_batch)
				for emb in ready_embs:
					# pickle
					x_test.append(emb)
					y_test.append(test_mels_labels[i])
					f_test.append(test_mels_f[i])
				print('done')

		self._save_pickles(feature_path, x_train, y_train, f_train, x_test, y_test, f_test)

	def _vggish(self):
		from models.vggish import VGGISH_CHECKPOINT_PATH, VGGISH_PCA_PARAMS_PATH
		import tensorflow as tf
		from models.vggish import vggish_input
		from models.vggish import vggish_params
		from models.vggish import vggish_postprocess
		from models.vggish import vggish_slim

		x_train, y_train, f_train, x_test, y_test, f_test = [], [], [], [], [], []

		feature_path = os.path.join(self.dataset['feature_path'], 'vggish')
		if not os.path.exists(feature_path):
			os.mkdir(feature_path)

		# firstly calculate mels
		train_mels = []
		train_mels_labels = []
		train_mels_f = []
		for i, row in self.dataset.train_data.iterrows():
			print('[Train] {}) Getting mels from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)
			train_mels.append(vggish_input.wavdata_to_examples(wav_data))
			train_mels_labels.append(self._build_multilabel(row))
			train_mels_f.append(row['cur_name)'])
			print('done.')

		test_mels = []
		test_mels_labels = []
		test_mels_f = []
		for i, row in self.dataset.test_data.iterrows():
			print('[Test] {}) Getting mels from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)
			test_mels.append(vggish_input.wavdata_to_examples(wav_data))
			test_mels_labels.append(self._build_multilabel(row))
			test_mels_f.append(row['cur_name'])
			print('done')

		# now calc embeddings in one single grapth and session
		# Prepare a postprocessor to munge the model embeddings.
		pproc = vggish_postprocess.Postprocessor(VGGISH_PCA_PARAMS_PATH)

		with tf.Graph().as_default(), tf.Session() as sess:
			# Define the model in inference mode, load the checkpoint, and
			# locate input and output tensors.
			vggish_slim.define_vggish_slim(training=False)
			vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_CHECKPOINT_PATH)
			features_tensor = sess.graph.get_tensor_by_name(
				vggish_params.INPUT_TENSOR_NAME)
			embedding_tensor = sess.graph.get_tensor_by_name(
				vggish_params.OUTPUT_TENSOR_NAME)

			for i, mels in enumerate(train_mels):
				print('[Train] {}) Getting embeddings...'.format(i), end='')
				# Run inference and postprocessing.
				try:
					[embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: mels})
				except tf.errors.InvalidArgumentError:
					raise ValueError("Audio duration should be equal or more than 1")

				ready_embs = pproc.postprocess(embedding_batch)
				for emb in ready_embs:
					x_train.append(emb)
					y_train.append(train_mels_labels[i])
					f_train.append(train_mels_f[i])
				print('done.')
			for i, mels in enumerate(test_mels):
				print('[Test] {}) Getting embeddings...'.format(i), end='')
				# Run inference and postprocessing.
				try:
					[embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: mels})
				except tf.errors.InvalidArgumentError:
					raise ValueError("Audio duration should be equal or more than 1")

				ready_embs = pproc.postprocess(embedding_batch)
				for emb in ready_embs:
					# pickle
					x_test.append(emb)
					y_test.append(test_mels_labels[i])
					f_test.append(test_mels_f[i])
				print('done')

		self._save_pickles(feature_path, x_train, y_train, f_train, x_test, y_test, f_test)

	def logmel(self):
		from librosa.feature import melspectrogram
		from librosa.core import load

		logmel_params = self.config['logmel_params']
		sr = logmel_params['sr']
		n_fft = logmel_params['n_fft']
		hop_length = logmel_params['hop_length']
		n_mels = logmel_params['n_mels']

		feature_path = os.path.join(self.dataset['feature_path'],
									'logmel_{}_{}_{}_{}'.format(sr, n_fft, hop_length, n_mels))
		if not os.path.exists(feature_path):
			os.mkdir(feature_path)

		x_train = []
		y_train = []
		f_train = []
		for i, row in self.dataset.train_data.iterrows():
			print('[Train] {}) Getting logmels from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			wav_data, sr = load(wav_name, sr=sr)
			x_train.append(melspectrogram(wav_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
			y_train.append(self._build_multilabel(row))
			f_train.append(row['cur_name'])
			print('done.')

		x_test = []
		y_test = []
		f_test = []
		for i, row in self.dataset.test_data.iterrows():
			print('[Test] {}) Getting mels from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			wav_data, sr = load(wav_name, sr=sr)
			x_test.append(melspectrogram(wav_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
			y_test.append(self._build_multilabel(row))
			f_test.append(row['cur_name'])
			print('done')

		self._save_pickles(feature_path, x_train, y_train, f_train, x_test, y_test, f_test)

	def sff(self):
		feature_path = os.path.join(self.dataset['feature_path'], 'sff')
		if not os.path.exists(feature_path):
			os.mkdir(feature_path)

		x_train = []
		y_train = []
		f_train = []
		for i, row in self.dataset.train_data.iterrows():
			print('[Train] {}) Getting sff from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)
			# wav_data, sr = load(wav_name, sr=sr)
			x_train.append(get_sff(wav_data, sr=sr))
			y_train.append(self._build_multilabel(row))
			f_train.append(row['cur_name'])
			print('done.')

		x_test = []
		y_test = []
		f_test = []
		for i, row in self.dataset.test_data.iterrows():
			print('[Test] {}) Getting sff from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)
			x_test.append(get_sff(wav_data))
			y_test.append(self._build_multilabel(row))
			f_test.append(row['cur_name'])
			print('done.')

		# не забываем следить за порядком аргументов!
		self._save_pickles(feature_path, x_train, y_train, f_train, x_test, y_test, f_test)

	def pre_sff(self):
		feature_path = os.path.join(self.dataset['feature_path'], 'pre_sff')
		if not os.path.exists(feature_path):
			os.mkdir(feature_path)

		x_train = []
		y_train = []
		f_train = []
		for i, row in self.dataset.train_data.iterrows():
			print('[Train] {}) Getting pre_sff from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)

			spec = stft(buf_to_float(wav_data), n_fft=800, hop_length=160, win_length=320)[:200, :]  # до 4К KHz
			spec = np.log(np.abs(spec) + 1e-10)
			spec -= np.min(spec)

			x_train.append(spec)
			y_train.append(self._build_multilabel(row))
			f_train.append(row['cur_name'])
			print('done.')

		x_test = []
		y_test = []
		f_test = []
		for i, row in self.dataset.test_data.iterrows():
			print('[Test] {}) Getting sff from {}...'.format(i, row['cur_name']), end='')
			wav_name = os.path.join(self.dataset['data_path'], row['cur_name'])
			sr, wav_data = wavfile.read(wav_name)

			spec = stft(buf_to_float(wav_data), n_fft=800, hop_length=160, win_length=320)[:200, :]  # до 4К KHz
			spec = np.log(np.abs(spec) + 1e-10)
			spec -= np.min(spec)

			x_test.append(spec)
			y_test.append(self._build_multilabel(row))
			f_test.append(row['cur_name'])
			print('done.')

		self._save_pickles(feature_path, x_train, y_train, f_train, x_test, y_test, f_test)

	def extract(self):
		feature_types = self.config['feature_type']
		for feature_type in feature_types:
			if feature_type == 'vggish':
				self._vggish()
			elif feature_type == 'vggish1':
				self._vggish1()
			elif feature_type == 'logmel':
				self.logmel()
			elif feature_type == 'sff':
				self.sff()
			elif feature_type == 'pre_sff':
				self.pre_sff()
			else:
				pass


class FeatureExtractor:

	def __init__(self, dataset, config):
		"""
		:param dataset: объект класса `Dataset` с загруженным csv-файлом датасета
		:param config: объект класса `Config` с загруженным файлом конфигурации
		"""
		self.dataset = dataset
		self.config = config


	def preprocessor(self, wav, preprocess_type):
		"""
		Определение препроцессоров (и аугментаторов) файлов ДО извлечения фич.
		Добавление нового препроцессора выглядит как добавление нового `elif`
		:param wav: объект класса `numpy.ndarray`, считанный wav-файл
		:param preprocess_type: объект класса `str`, название препроцессинга
		"""
		if wav.ndim < 2:
			# если вавка одноканальная, то с ней ничего не сделать
			features = wav
		else:
			if preprocess_type == "raw":
				dtype = wav.dtype
				features = np.mean(wav, axis=1).astype(dtype)
			elif preprocess_type == "left":
				features = wav[0]
			elif preprocess_type == "right":
				features = wav[1]
			elif preprocess_type == "middle":
				features = wav[0] + wav[1]
			elif preprocess_type == "side":
				features = wav[0] - wav[1]
			else:
				pass  # for another preprocessing type
		return features


	def extractor(self, wav, sr, feature_type, full_file_path):
		"""
		TODO
		- экстрактор принимает файлик
		- `librosa` принимает wav-файлы в `float`, а не в `int` - нужна доп. обработка этого
		Определение экстракторов фичей (и постаугментаторов) файлов.
		Добавление нового экстрактора выглядит как добавление нового `elif`.
		:param wav: объект класса `numpy.ndarray`, считанный wav-файл
		:param sr: sample rate wav-файла
		:param feature_type: объект класса `str`, название экстрактора
		:param full_file_path: объект класса `str`, полный путь до wav-файла
		"""
		# сырые данные
		if feature_type == 'raw':
			features = wav

		# фурье коэффициенты
		elif feature_type == 'fft':
			_, _, spectr = signal.spectrogram(wav, nperseg=200, nfft=200, fs=8000, noverlap=128)
			features = spectr

		elif feature_type == 'mel':
			features = librosa.feature.melspectrogram(wav, n_mels=128, sr=sr, n_fft=2048, hop_length=1024)
		
		elif feature_type == 'mfcc':
			features = librosa.feature.mfcc(wav, n_mfcc=40, sr=sr)
		
		elif feature_type == 'percep_spec':
			n_fft = 512
			stft = librosa.stft(wav, n_fft=n_fft, hop_length=n_fft // 4, win_length=None, window='hann',
								center=True, pad_mode='reflect')
			stft = np.abs(stft)
			freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
			stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=99.0)

			features = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=128, fmax=sr//2)
		
		# эмбеддинги VGGish модели 
		# подробнее: 
		elif feature_type == 'vggish':
			from models.vggish import vggish_gen_embeddings, VGGISH_CHECKPOINT_PATH, \
				VGGISH_PCA_PARAMS_PATH
			embeddings = vggish_gen_embeddings.get_embeddings(VGGISH_CHECKPOINT_PATH,
																VGGISH_PCA_PARAMS_PATH,
																full_file_path)
			features = embeddings
		
		else:
			pass  # for another types of features
		
		return features


	def align(self, wav, sr):
		"""
		Выравнивает длину считанного wav-файла. При 0 возвращает файл, 
		как есть.
		:param wav: `np.array` считанного wav-файла
		:param sr: sample rate wav-файла
		"""
		wav_length = self.config['wav_length']
		if not wav_length:
			return wav
		elif wav.ndim > 1:
			if len(wav[0]) > sr * wav_length:
				wav[0] = wav[0][:sr * wav_length]
				wav[1] = wav[1][:sr * wav_length]
			elif len(wav[0]) < sr * wav_length:
				wav[0] = np.pad(wav[0], (0, wav_length * sr - len(wav[0])), "constant", constant_values=0)
				wav[1] = np.pad(wav[1], (0, wav_length * sr - len(wav[1])), "constant", constant_values=0)
		else:
			if len(wav) > sr * wav_length:
				wav = wav[:sr * wav_length]
			elif len(wav) < sr * wav_length:
				wav = np.pad(wav, (0, wav_length * sr - len(wav)), "constant", constant_values=0)
		return wav


	def preprocess(self):
		"""
		Метод для препроцессинга. Параметры задаются через yaml-файл конфигурации.
		Поддерживаются следующие типы препроцессинга:
		raw – оставить набор амплитуд аудиофайла как есть;
		left – левый канал;
		right – правый канал;
		middle – сумма левого и правого каналов;
		side – разность левого и правого каналов.
		Преобразованные файлы сохраняются в директорию, указанную в конфигурации вместе с сформированным csv-файлом
		с описанием кажого сохранённого аудиофайла.
		"""
		preprocess_types = self.config["preprocess_types"]
		data_folder = self.config["data_folder"]
		data_column = self.config["data_column"]
		processed_data_folder = self.config["processed_data_folder"]
		for preprocess_type in preprocess_types:
			for file in self.dataset:
				file_name = file[data_column]

				sr, wav  = wavfile.read(os.path.join(data_folder, file_name))
				# выравнивание файла до заданной длины
				wav = self.align(wav, sr)

				features = self.preprocessor(wav, preprocess_type)

				# Проверка на существование и создание директории для сохранения обработанных файлов.
				if not os.path.isdir(os.path.join(processed_data_folder, preprocess_type)):
					os.makedirs(os.path.join(processed_data_folder, preprocess_type))
				wavfile.write(os.path.join(processed_data_folder, preprocess_type, file_name), sr, features)

				# Добавление в датафрейм информации о преобразованном аудиофайле.
				print("Process", file[data_column], "done.")
			np.savetxt(os.path.join(processed_data_folder, preprocess_type, META_FILE_NAME), self.dataset,
			header=', '.join(self.dataset.dtype.names), delimiter=',', fmt='%s')


	def extract(self):
		"""
		Экстрактор признаков. Поддерижваются следующие признаки:
		raw – амплитуды оставлены как есть (т.е. без преобразования);
		fft – спектрограммы быстрого преобразования Фурье;
		mel – Mel-спектрограммы;
		mfcc – MFCC-спектрограммы.
		В блоке выбора признаков можно легко добавлять свои обработчики, просто добавляя новый "elif".
		"""
		preprocess_types = self.config["preprocess_types"]
		feature_types = self.config["feature_type"]
		features_folder = self.config["features_folder"]
		processed_data_folder = self.config["processed_data_folder"]
		data_column = self.config["data_column"]

		if not os.path.isdir(features_folder):
			os.makedirs(features_folder)

		for preprocess_type in preprocess_types:

			if not os.path.isdir(os.path.join(features_folder, preprocess_type)):
				os.makedirs(os.path.join(features_folder, preprocess_type))

			for feature_type in tqdm(feature_types):
				with h5py.File(os.path.join(features_folder, 
					preprocess_type, feature_type+HDF_FILE_EXTENTION), 'w') as hdf5_file:

					for file in self.dataset:
						full_file_path = os.path.join(processed_data_folder, preprocess_type, file[data_column])
						sr, wav  = wavfile.read(full_file_path)

						features = self.extractor(wav, sr, feature_type, full_file_path)
						# print(features.shape, end='\r')

						# print("Calc features from", file[data_column], "done.\r")

						hdf5_file[file[data_column]] = features
