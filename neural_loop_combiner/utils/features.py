import warnings
warnings.filterwarnings('ignore')
import os
import librosa

def get_melspectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S = librosa.util.normalize(S)
    return S