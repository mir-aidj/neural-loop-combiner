import warnings
warnings.filterwarnings("ignore")

import os
import random
import librosa
import numpy as np
from neural_loop_combiner.config          import settings
from neural_loop_combiner.utils.seperate  import tag_loop_type
from neural_loop_combiner.utils.utils     import log_message, data_exclude

class Tagger:
    
    def __init__(self, loop_path):
        self.sr        = settings.SR
        self.cache     = settings.CACHE
        self.out_dir   = settings.OUT_DIR
        self.loop_path = loop_path
        
    
    def tag(self):
        return tag_loop_type(self.loop_path, self.sr, self.cache, self.out_dir)