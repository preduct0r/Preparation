"""Utils for audio processing"""
import hashlib
import warnings

from fnmatch import fnmatch
from typing import Tuple

import soundfile as sf
import torch

from scipy.io import wavfile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def read_audio(path_to_file: str, target_sr, dtype: str = 'short'):
    """Read data from audio.

    Parameters
    ----------
    path_to_file: full path to wav.
    target_sr: target sample rate for file (only check sameness, not apply resampling. To apply resampling, use ffmpeg.)
    dtype: short or float

    Returns
    -------
    sr: int, sample rate of given *.wav file
    wav_data: np.ndarray, data from *.wav file
              shape = (n, m)
              n - number of channels
              m - number of samples
    """
    if not fnmatch(path_to_file, '*.wav'):
        raise ValueError('{} is not a wawfile!'.format(path_to_file))
    if dtype == 'float':
        wav_data, sr = sf.read(path_to_file)
    else:
        try:
            sr, wav_data = wavfile.read(path_to_file)
        except Exception as e:
            warnings.warn(str(e))
            warnings.warn('Error while trying to read audio as short array! Trying to read as floating-point array.')
            wav_data, sr = sf.read(path_to_file)
            warnings.warn('Successfully read floating point data. Be carefull, now it\'s not a short array!')
    if sr != target_sr:
        raise ValueError('File {} has sample rate {} but target is {}'.format(path_to_file, sr, target_sr))

    return sr, wav_data


def calc_metrics(y_true, y_pred, labels: list):
    """
    Calculate metrics for given labels, predictions and ground true labels.

    Parameters
    ----------
    y_true: iterable type (list, np.ndarray) containing categorical true labels
    y_pred: iterable type (list, np.ndarray) containing categorical predicted labels
    labels: all possible labels in correct order

    Returns
    -------
    acc: accuracy_score
    uar: UAR (Unweighted Average Recall), Weighted accuracy (in some articles has that name)
    uap: UAP (Unweighted Average Precision)
    """
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average='macro')
    uap = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels)

    return acc, uar, uap, f1, cm


def calc_md5_hash(path_to_file):
    """Calculate md5 hash sum for given file."""
    hash_md5 = hashlib.md5()
    with open(path_to_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_model(model_object: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load weights to an initialized model object

    Parameters
    ----------
    model_class : torch.nn.Module
        initialized model object
    path : str
        path to model weights file

    Returns
    -------
    torch.nn.Module
        model object with loaded weights from file
    """
    print('model init')
    model_object.load_state_dict(torch.load(path))
    return model_object


def train(
    X_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
) -> float:
    """Teach model the entire epoch on data

    Parameters
    ----------
    X_loader : torch.utils.data.DataLoader
        Torch dataset loader object
    optimizer : torch.optim.Optimizer
        current optimizer object
    model : torch.nn.Module
        model to train object
    loss_fn : torch.nn.Module
        loss function object

    Returns
    -------
    float
        last loss of epoch
    """
    model.train()
    for embds, labs in X_loader:
        embds, labs = embds.cuda(), labs.cuda()

        optimizer.zero_grad()
        out = model(embds)
        loss = loss_fn(out, labs)
        loss.backward()
        optimizer.step()
    return loss.item()


def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """Test a model on test set

    Parameters
    ----------
    model : torch.nn.Module
        model object to test
    loader : torch.utils.data.DataLoader
        test dataset loader

    Returns
    -------
    Tuple[float, float]
        batch-wice f1-score and f1-score on the whole dataset
    """
    model.eval()
    f1 = 0
    outs_list, labs_list = [], []
    for embds, labs in loader:
        embds = embds.cuda()
        with torch.no_grad():
            out = model.inference(embds)
            # out = torch.softmax(out, dim=1)

        out = (torch.max(out, 1))[1]
        out = out.detach().cpu().numpy()
        outs_list += list(out)
        labs = labs.numpy()
        labs_list += list(labs)

        f1 += f1_score(out, labs, average='macro')

    f1 /= len(loader)

    f1_all = f1_score(list(outs_list), list(labs_list), average='macro')
    print(classification_report(list(outs_list), list(labs_list)))
    print(confusion_matrix(list(labs_list), list(outs_list), normalize='true'))

    return f1, f1_all
