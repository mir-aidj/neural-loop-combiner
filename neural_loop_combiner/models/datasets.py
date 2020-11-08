import os
import librosa
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

from neural_loop_combiner.config         import settings
from neural_loop_combiner.utils.features import get_melspectrogram
from neural_loop_combiner.utils.utils    import data_shuffle

class MixedDataset(Dataset):
    def __init__(self, pos_datas, neg_datas):
        self.sr         = settings.SR
        self.out_dir    = settings.OUT_DIR
        self.datas_path = data_shuffle(self.attach_label(pos_datas, 1) + self.attach_label(neg_datas, 0))
                
    def attach_label(self, datas, label):    
        return [[data, label] for data in datas]

    def __getitem__(self, index):
        sr      = self.sr
        out_dir = self.out_dir 
        
        datas_path, label = self.datas_path[index]
        datas_path   = [os.path.join(out_dir, data_path) for data_path in datas_path ] 
        datas_audio  = [librosa.load(data_path, sr=sr)[0] for data_path in datas_path]
        mixed_audio  = datas_audio[0] * 0.5 + datas_audio[1] * 0.5
        mixed_spec   = get_melspectrogram(mixed_audio, sr)
        mixed_spec   = mixed_spec.reshape(1, mixed_spec.shape[0], mixed_spec.shape[1])
        
                
        return (mixed_spec, mixed_audio, datas_audio, datas_path, label)
    
    
    def __len__(self):
        return len(self.datas_path)
    
class PairDataset(Dataset):
    def __init__(self, pos_datas, neg_datas):
        
        self.sr         = settings.SR
        self.out_dir    = settings.OUT_DIR
        self.datas_path = data_shuffle(self.attach_label(pos_datas, 1) + self.attach_label(neg_datas, 0))
        
    def attach_label(self, datas, label):
        return [[data, label] for data in datas]
    
    def __getitem__(self, index):
        sr      = self.sr
        out_dir = self.out_dir 
        
        datas_path, label = self.datas_path[index]
        datas_path   = [os.path.join(out_dir, data_path) for data_path in datas_path] 
        datas_audio  = [librosa.load(data_path, sr=sr)[0] for data_path in datas_path]
        datas_spec   = [get_melspectrogram(data_audio, sr) for data_audio in datas_audio]
        datas_spec   = [data_spec.reshape(1, data_spec.shape[0], data_spec.shape[1])for data_spec in datas_spec]        

        return (datas_spec, datas_audio, datas_path, label)    
    
    def __len__(self):
        return len(self.datas_path)
    
    
    
class SingleDataset(Dataset):
    def __init__(self, datas_path):
        
        self.sr         = settings.SR
        self.out_dir    = settings.OUT_DIR
        self.data_paths = data_shuffle(datas_path)
        
    def __getitem__(self, index):
        
        sr      = self.sr
        out_dir = self.out_dir
        
        data_path  = os.path.join(out_dir, self.data_paths[index])
        data_audio = librosa.load(data_path, sr=self.sr)[0]
        data_spec  = get_melspectrogram(data_audio, self.sr)
        data_spec  = data_spec.reshape(1, data_spec.shape[0], data_spec.shape[1])
                
        return (data_spec, data_audio, data_path)
        
    
    def __len__(self):
        return len(self.data_paths)