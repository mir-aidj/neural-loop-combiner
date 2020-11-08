import warnings
warnings.filterwarnings("ignore")

import os
import random
import librosa
import numpy as np
from spleeter.separator     import Separator
from spleeter.audio.adapter import get_default_audio_adapter
from neural_loop_combiner.utils.utils     import log_message, data_exclude
from neural_loop_combiner.config          import settings
from neural_loop_combiner.dataset.sampler import Sampler

class DataSampler:
    
    def __init__(self, tracks_key, tracks_dict, idv_datas, harm_datas, data_type, log_info=[]):
         
        self.sr          = settings.SR
        self.cache       = settings.CACHE
        self.dur         = settings.DUR
        self.log         = settings.LOG
        self.out_dir     = settings.OUT_DIR
        self.ng_types    = [neg_type for neg_type in settings.NG_TYPES.keys() if settings.NG_TYPES[neg_type] == 1]
        
        self.data_type   = data_type
        self.idv_datas   = idv_datas
        self.harm_datas  = harm_datas
        
        self.tracks_key  = tracks_key
        self.tracks_dict = tracks_dict
        
    
    def sampling(self):
        tracks_key  = self.tracks_key
        tracks_dict = self.tracks_dict
        ng_types    = self.ng_types
        idv_datas   = self.idv_datas
        harm_datas  = self.harm_datas
        data_type   = self.data_type
        log         = self.log  
        neg_datas   = {ng_type:[] for ng_type in ng_types} 
        total       = len(tracks_key)
        
        for i, track_key in enumerate(tracks_key):
            excl_datas  = tracks_dict[track_key]['loops_path']
            pair_datas  = tracks_dict[track_key]['pairs_path']
            other_datas = data_exclude(idv_datas, excl_datas)
            for pair_data in pair_datas:
                neg_dict = Sampler(pair_data, other_datas, harm_datas).sampling()
                for neg_type in neg_dict:
                    neg_datas[neg_type].append(neg_dict[neg_type])
            if log: log_message(f'Negative Sampling processing ({data_type})', [i+1, total])
        if log: log_message(f'Negative Sampling completed ({data_type})')
        return neg_datas
         
            
        
        
        
    
