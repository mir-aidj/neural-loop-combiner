import json
import random
import filetype
import os, librosa
import numpy as np
import soundfile as sf
from neural_loop_combiner.config import settings

def save_np(file_path, file):
    existed = os.path.exists(file_path)
    if not existed:
        np.save(file_path, file)
    return existed

def load_np(file_path, cache):
    return np.load(file_path, allow_pickle=True) if cache and os.path.exists(file_path) else None
    
def save_audio(audio_path, audio, sr):
    existed = os.path.exists(audio_path)
    sf.write(audio_path, audio, sr)
    return existed

def check_exist(output_path):
    existed = os.path.exists(output_path)
    if not existed:
        os.makedirs(output_path)
    return existed

def check_files_exist(files):
    exists = [os.path.exists(file) for file in files]
    return sum(exists) == len(files)
    
def str_replace(string):
    return string.replace('/', '_').replace(' ', '_').replace('__', '_').replace('__', '_')

def get_save_dir(out_dir, cat_dirs):
    for cat_dir in cat_dirs:
        out_dir = os.path.join(out_dir, cat_dir)
        check_exist(out_dir)
    return out_dir

def get_file_name(file_path):
    media_type = filetype.guess(file_path).extension 
    file_name  = os.path.split(file_path)[-1].split(f'.{media_type}')[0]
    return file_name, media_type

def remove_prefix_dir(path, first_dir):
    return os.path.relpath(path, first_dir)


def log_message(message, log_info=[]):
    if len(log_info) <= 0:
        print(f'{message} ...')
    else:
        junction = '/'
        print(f'[{junction.join([str(log) for log in log_info])}] {message}...')
        
def discrete_matrix(matrix, thre):
    return matrix > thre


def pair_iteration(lst1, lst2):
    return [list(lst) for lst in np.unique([np.array(sorted([x, y])) for x in lst1 for y in lst2 if x != y], axis=0)]


def pair_table_creation(template):
    pair_template = [v for v in template if sum(v) >= 2]
    if len(pair_template) != 0:
        unique_bar = np.unique(pair_template, axis=0)
        pair_bar   = [sorted(y[0]) for y in (map(lambda x: np.where(x == True), unique_bar))]
        pair_table = np.unique([np.array(y) for x in list(map(lambda bar: pair_iteration(bar, bar), pair_bar)) for y in x], axis=0)
        pair_table = [list(x) for x in pair_table]
    else:
        pair_table = []
    return pair_table

def data_shuffle(array):
    random.shuffle(array)
    return array

def data_exclude(array, excl_array):
    return [element for element in array if element not in excl_array]

def data_include(array, incd_array):
    return [element for element in array if element in incd_array]

def data_deduplicate(array):
    deduplicate_array = []
    for element in array:
        if element not in deduplicate_array:
            deduplicate_array.append(element)
    return deduplicate_array

def save_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
        
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data