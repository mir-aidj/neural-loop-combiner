import warnings
warnings.filterwarnings("ignore")

import os
import random
import librosa
import numpy as np

from neural_loop_combiner.config           import settings
from neural_loop_combiner.utils.seperate   import tag_loop_type
from neural_loop_combiner.utils.utils      import data_include
from neural_loop_combiner.dataset.sampler.manipuler import Manipuler

class Sampler:
    
    def __init__(self, pair_path, others_path, harm_datas, log_info=[]):
         
        self.sr          = settings.SR
        self.cache       = settings.CACHE
        self.dur         = settings.DUR
        self.log         = settings.LOG
        self.out_dir     = settings.OUT_DIR
        
        self.log_info    = log_info
        self.pair_path   = pair_path
        self.others_path = others_path
        self.harm_datas  = harm_datas
        self.map_dict    = {}
        self.ng_types    = [neg_type for neg_type in settings.NG_TYPES.keys() if settings.NG_TYPES[neg_type] == 1]        
        
    def shuffle_pair_path(self):
        pair_path = self.pair_path
        mnp_index = random.randint(0, 1)
        stc_index = 1 - mnp_index
        return pair_path[stc_index], pair_path[mnp_index]
        
    def pair_manipulation(self, mnp_type): 
        stc_path, mnp_path = self.shuffle_pair_path()
        new_mnp_path = Manipuler(mnp_path, mnp_type).save_outputs()
        return [stc_path, new_mnp_path]
    
    def random_paring(self):
        others_path = self.others_path
        stc_path, _ = self.shuffle_pair_path()
        mnp_path    = random.choice(others_path)
        return [stc_path, mnp_path]
    
    def selected_paring(self):
        others_path = data_include(self.others_path, self.harm_datas)
        stc_path, _ = self.shuffle_pair_path()
        mnp_path    = random.choice(others_path)
        
        return [stc_path, mnp_path]
    
    def choose_sampling_method(self, method):
        if method in ['shift', 'reverse', 'rearrange']:
            return self.pair_manipulation(method)
        elif method == 'random':
            return self.random_paring()
        elif method == 'selected':
            return self.selected_paring()
        
    def sampling(self):
        ng_types = self.ng_types
        ng_dict  = {} 
        
        for ng_type in ng_types:
            ng_dict[ng_type] = self.choose_sampling_method(ng_type)
        
        return ng_dict
            
        
        
        
    
