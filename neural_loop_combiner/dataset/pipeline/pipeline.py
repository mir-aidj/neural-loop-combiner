import warnings
warnings.filterwarnings("ignore")

import os, librosa
import numpy as np

from neural_loop_combiner.utils.utils      import get_save_dir, save_audio, save_np
from neural_loop_combiner.utils.utils      import pair_table_creation, remove_prefix_dir
from neural_loop_combiner.config           import settings
from neural_loop_combiner.utils.manipulate import time_stretch

from neural_loop_combiner.dataset.pipeline.extractor import Extractor
from neural_loop_combiner.dataset.pipeline.refintor  import Refintor


class Pipeline:
    
    def __init__(self, file_name, media_type, gpu_num, log_info=[]):
        
        self.sr          = settings.SR
        self.log         = settings.LOG
        self.cache       = settings.CACHE
        self.int_dir     = settings.INT_DIR
        self.out_dir     = settings.OUT_DIR
        
        self.log_info    = log_info
        self.file_name   = file_name
        self.file_path   = os.path.join(self.int_dir, f'{file_name}.{media_type}')
        
        self.extractor   = Extractor(file_name, media_type, gpu_num, self.log_info)
        self.refinter    = Refintor(self.extractor, self.log_info)
        
        
    def start(self):  
        
        file_name = self.file_name
        loops     = self.refinter.loops
        layout    = self.refinter.layout
        template  = self.refinter.template        

        loops_path = self.save_outputs(file_name, 'refined', loops, layout, template)
        pairs_path = self.pair_paths_creation(template, loops_path)
        
        return {
            'file_name' : file_name, 
            'loops_path': loops_path,
            'pairs_path': pairs_path
        }
        
        
        
    def pair_paths_creation(self, template, loops_path):
        
        pair_table = pair_table_creation(template)
        pair_path  = [[loops_path[index] for index in pair] for pair in pair_table]
        
        return pair_path
        
        
    
    def save_outputs(self, file_name, sub_dir, loops, layout, template):
        sr         = self.sr
        cache      = self.cache
        out_dir    = self.out_dir
        
        
        loops_dir  = get_save_dir(out_dir, [sub_dir, 'loops'])
        loops_path = [os.path.join(loops_dir, f'{file_name}_{index}.wav') for index in loops]
        
        for ith_loop in loops:
            save_audio(loops_path[ith_loop], loops[ith_loop], sr)
        
        if cache:
            layout_path   = os.path.join(get_save_dir(out_dir, [sub_dir, 'layout'])  , f'{file_name}.npy')
            template_path = os.path.join(get_save_dir(out_dir, [sub_dir, 'template']), f'{file_name}.npy')
            
            save_np(layout_path  , layout)
            save_np(template_path, template)
            
            
        return [remove_prefix_dir(path , out_dir) for path in loops_path]
    

    

        