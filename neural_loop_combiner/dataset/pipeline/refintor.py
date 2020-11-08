import warnings
warnings.filterwarnings("ignore")

import os, librosa
import numpy as np
from functools import reduce
from sklearn.preprocessing import normalize
from neural_loop_combiner.config               import settings
from neural_loop_combiner.utils.utils          import log_message, discrete_matrix, get_save_dir
from neural_loop_combiner.utils.comparison     import spec_similarity
from neural_loop_combiner.utils.manipulate     import time_stretch


class Refintor:
    
    def __init__(self, extractor, log_info=[]):
        
        self.sr           = settings.SR
        self.dur          = settings.DUR
        self.log          = settings.LOG

        self.exist_thre   = settings.EXISTED_THRESHOLD
        self.dupl_thre    = settings.HASH_THRESHOLD
        
        self.log_info     = log_info
        self.extractor    = extractor
        
        if self.log: log_message('Refintor started', self.log_info)
        self.refinement()
        self.template  = discrete_matrix(self.layout, self.exist_thre)
        if self.log: log_message('Refintor completed', self.log_info)
    
    def loops_refinement(self, dupl_table):
        
        sr         = self.sr
        dur        = self.dur
        
        ext_layout = self.extractor.layout
        ext_loops  = self.extractor.loops
        
        loops      = {}
        layout     = np.zeros((ext_layout.shape[0], len(dupl_table.keys())))
        
        for i, max_snr_ith in enumerate(dupl_table):
            sim_iths     = [max_snr_ith]+ dupl_table[max_snr_ith]
            actv_val     = list(map(lambda ith: ext_layout[:, ith], sim_iths))
            layout[:, i] = reduce(lambda x, y: x + y, actv_val)
            loops[i]     = time_stretch(ext_loops[max_snr_ith], dur, sr)
            
            
        return loops, normalize(layout, norm='l2') 
    
    def refinement(self):
        
        dupl_thre      = self.dupl_thre
        ext_layout     = self.extractor.layout
        ext_loops      = self.extractor.loops
        ext_specs      = self.extractor.specs
        ext_snr_scores = self.extractor.snr_scores
        
        
        dupl_table, used_iths = {}, []

        for ith in ext_loops:
            
            ith_loop  = ext_loops[ith]
            ith_spec  = ext_specs[ith]
            
            if ith not in used_iths:
                similarity    = list(map(lambda tgt_ith: spec_similarity(ith_spec, ext_specs[tgt_ith]), ext_specs))
                
                dupl_ith      = [i for i, s in enumerate(similarity) if s < dupl_thre and i not in used_iths]
                used_iths    += dupl_ith
                
                max_snr_score = max(map(lambda i: ext_snr_scores[i], dupl_ith))             
                max_snr_ith   = [i for i in dupl_ith if ext_snr_scores[i] == max_snr_score][0]
                repl_iths     = list(filter(lambda i: i != max_snr_ith, dupl_ith))

                dupl_table[max_snr_ith] = repl_iths
                
                if len(used_iths) == len(ext_loops.keys()):
                    break
                    
        self.loops, self.layout = self.loops_refinement(dupl_table)
        

        
        
    def get_features(self):
        return {
            'loops'   : self.loops,
            'layout'  : self.layout, 
            'template': self.template,
        }