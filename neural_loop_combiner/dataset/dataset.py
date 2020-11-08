import warnings
warnings.filterwarnings("ignore")

import os
import math
import random
import librosa
import datetime
import numpy as np

from neural_loop_combiner.config              import settings
from neural_loop_combiner.utils.utils         import log_message, data_shuffle, data_deduplicate
from neural_loop_combiner.utils.utils         import save_json, get_save_dir, remove_prefix_dir
from neural_loop_combiner.dataset.dataSampler import DataSampler

class Dataset:
    
    def __init__(self, tracks_dict, harm_datas, log_info=[]):
        
        self.sr           = settings.SR
        self.dur          = settings.DUR
        self.log          = settings.LOG
        self.out_dir      = settings.OUT_DIR
        self.test_size    = settings.TEST_SIZE
        self.ng_types     = [neg_type for neg_type in settings.NG_TYPES.keys() if settings.NG_TYPES[neg_type] == 1]
        self.split_ratio  = settings.SPLIT_RATIO
        self.harm_datas   = harm_datas
        self.tracks_dict  = tracks_dict
        self.tracks_key   = self.data_split() 
        
        self.pos_datas    = self.pos_datas_retrieve()
        self.idv_datas    = self.idv_datas_retrieve()
        self.neg_datas    = self.neg_datas_retrieve()
    
    def idv_datas_retrieve(self):
        pos_datas = self.pos_datas
        return {data_type: data_deduplicate([idv_loop for pair in pos_datas[data_type] for idv_loop in pair]) for data_type in pos_datas}
        
    def data_split(self):
        test_size         = self.test_size
        tracks_dict       = self.tracks_dict
        split_ratio       = self.split_ratio
        tracks_key        = data_shuffle(list(tracks_dict.keys())) 
        
        test_tracks_key   = [track_key for track_key in tracks_key if len(tracks_dict[track_key]['pairs_path']) == 1][:test_size]
        other_tracks_key  = [track_key for track_key in tracks_key if track_key not in test_tracks_key]
        
        pairs_num         = [len(tracks_dict[key]['pairs_path']) for key in other_tracks_key]
        pairs_acc         = [sum(pairs_num[0:i]) for i in range(1, len(pairs_num))]
        val_pairs_num     = math.floor(pairs_acc[-1] * (1 - split_ratio))
        split_index       = [i for i in range(len(pairs_acc)) if pairs_acc[i] > val_pairs_num][0]
        
        val_tracks_key   = other_tracks_key[0:split_index]
        train_tracks_key = other_tracks_key[split_index:]
        
        return {
            'val'  : val_tracks_key,
            'test' : test_tracks_key,
            'train': train_tracks_key,
        }
    
    def pos_datas_retrieve(self):
        log         = self.log
        tracks_dict = self.tracks_dict
        tracks_key  = self.tracks_key
        pos_datas   = {}
        
        for data_type in tracks_key:
            tmp = [tracks_dict[tracks_key]['pairs_path'] for tracks_key in tracks_key[data_type]]
            pos_datas[data_type] = [pair for tracks in tmp for pair in tracks]
        if log: log_message('Positive Retrieve completed')
        return pos_datas        

    
    def neg_datas_retrieve(self):
        log         = self.log
        harm_datas  = self.harm_datas
        tracks_dict = self.tracks_dict
        tracks_key  = self.tracks_key
        idv_datas   = self.idv_datas
        neg_datas   = {}
        
        for data_type in tracks_key:
            if data_type != 'test':
                neg_datas[data_type] = DataSampler(tracks_key[data_type], tracks_dict, idv_datas[data_type], harm_datas, data_type).sampling()
        if log: log_message('Negative Retrieve completed')
        return neg_datas
                
    
    def datas_retrieve(self):
        out_dir     = self.out_dir
        ng_types    = self.ng_types
        pos_datas   = self.pos_datas
        neg_datas   = self.neg_datas
        idv_datas   = self.idv_datas
        
        date        = datetime.datetime.utcnow()
        loops_count = {data_type: len(idv_datas[data_type]) for data_type in idv_datas}
        pairs_count = {data_type: len(pos_datas[data_type]) for data_type in pos_datas}
        
        loops_count['total'] = sum([loops_count[data_type] for data_type in loops_count])
        pairs_count['total'] = sum([pairs_count[data_type] for data_type in pairs_count])
        
        datas  = {
            'pos' : pos_datas, 
            'neg' : neg_datas,
            'idv' : idv_datas, 
        } 
        
        data_dir  = get_save_dir(out_dir, ['datasets'])
        data_path = os.path.join(data_dir, f'{date}.json')
        save_json(datas, data_path)
        
        return { 
            'date'       : date,
            'data_path'  : remove_prefix_dir(data_path, out_dir),
            'neg_types'  : ng_types,
            'loops_count': loops_count,
            'pairs_count': pairs_count
        }
                
    
        