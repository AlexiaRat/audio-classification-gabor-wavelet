import numpy as np
from gabor_filter import create_gabor_filterbank
from mexican_hat import create_mexican_hat_filterbank

M = 12       # Number of filters in the bank
SIZE = 1102  # Window size in samples
HOP_MS = 12  # Hop between windows in milliseconds


# Segment audio into overlapping frames
def create_windows(audio, window_size, hop_time_ms, fs):
    hop_samples = int((hop_time_ms / 1000.0) * fs)
    num_windows = (len(audio) - window_size) // hop_samples + 1
    if num_windows < 1:
        num_windows = 1
    windows = np.zeros((num_windows, window_size))
    for i in range(num_windows):
        start = i * hop_samples
        end = start + window_size
        if end <= len(audio):
            windows[i] = audio[start:end]
        else:
            remaining = len(audio) - start
            windows[i, :remaining] = audio[start:]
    return windows


def filter_windows_efficient(windows, filters_cos, filters_sin):
    filters_cos_flipped = np.flip(filters_cos, axis=1)
    filters_sin_flipped = np.flip(filters_sin, axis=1)
    responses_cos = windows @ filters_cos_flipped.T
    responses_sin = windows @ filters_sin_flipped.T
    magnitude = np.sqrt(responses_cos**2 + responses_sin**2)
    mean_features = np.mean(magnitude, axis=0)
    std_features = np.std(magnitude, axis=0)
    features = np.concatenate([mean_features, std_features])
    return features


def get_features(audio_train, fs):
    filters_cos, filters_sin, _, _ = create_gabor_filterbank(M, SIZE, fs)
    num_files = audio_train.shape[0]
    feat_train = np.zeros((num_files, 2 * M))
    for i in range(num_files):
        audio = audio_train[i, :]
        windows = create_windows(audio, SIZE, HOP_MS, fs)
        feat_train[i] = filter_windows_efficient(windows, filters_cos, filters_sin)
    return feat_train


def get_features_mexican(audio_train, fs):
    filters_cos, filters_sin, _, _ = create_mexican_hat_filterbank(M, SIZE, fs)
    num_files = audio_train.shape[0]
    feat_train = np.zeros((num_files, 2 * M))
    for i in range(num_files):
        audio = audio_train[i, :]
        windows = create_windows(audio, SIZE, HOP_MS, fs)
        feat_train[i] = filter_windows_efficient(windows, filters_cos, filters_sin)
    return feat_train


