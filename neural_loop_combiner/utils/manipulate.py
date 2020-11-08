import random
import librosa
import numpy as np
import pyrubberband as pyrb
from pysndfx import AudioEffectsChain

def time_stretch(src_audio, tgt_dur, sr):
    src_dur = librosa.get_duration(src_audio, sr)
    return pyrb.time_stretch(src_audio, sr, src_dur / tgt_dur)


def loops_stretch(loops, dur, sr):
    return {key: time_stretch(loops[key], dur, sr) for key in loops}

def split_audio(audio, beats, sr):
    beats_sample = librosa.time_to_samples(beats, sr=sr)
    audio_split = [audio[beats_sample[i]:beats_sample[i+1]]for i in range(len(beats_sample)-1)]
    return audio_split

def reverse_audio(audio):
    fx = AudioEffectsChain().reverse()
    return fx(audio)

def shift_audio(audio, beats, sr):
    step = random.randint(1, len(beats)-2) * random.choice([-1, 1])
    audio_split = split_audio(audio, beats, sr)
    audio_shift = np.array([])
    for i in range(len(audio_split)):
        audio_shift = np.concatenate((audio_shift, audio_split[(i + step)%len(audio_split)]))
    return audio_shift


def rearrange_audio(audio, beats, sr):
    audio_split = split_audio(audio, beats, sr)
    audio_rearrange = np.array([])
    order     = [i for i in range(0, len(audio_split))]
    order_rag = [i for i in range(0, len(audio_split))]
    while(1):
        random.shuffle(order_rag)
        if order != order_rag:
            break
    for index in order_rag:
        audio_rearrange = np.concatenate((audio_rearrange, audio_split[index]))
    return audio_rearrange




    