import warnings
warnings.filterwarnings("ignore")

import filetype
import os, librosa
import numpy as np

from neural_loop_combiner.config           import settings
from neural_loop_combiner.utils.utils      import log_message, get_save_dir, save_audio, remove_prefix_dir
from neural_loop_combiner.utils.manipulate import reverse_audio, shift_audio, rearrange_audio

class Manipuler:
    
    def __init__(self, file_path, mnp_type, log_info=[]):
        
        self.sr           = settings.SR
        self.dur          = settings.DUR
        self.log          = settings.LOG
        self.out_dir      = settings.OUT_DIR
        self.file_path    = os.path.join(self.out_dir, file_path)
        self.beats        = [i / self.dur for i in range(0, 5)]
        self.log_info     = log_info
        self.mnp_type     = mnp_type
        self.audio        = librosa.load(self.file_path, self.sr)[0]
        
        self.audio_manipulation()
        
    def audio_manipulation(self):
        
        sr    = self.sr
        audio = self.audio
        beats = self.beats
        mnp_type = self.mnp_type
        
        if mnp_type == 'shift':
            self.mnp_audio = shift_audio(audio, beats, sr)
        elif mnp_type == 'reverse':
            self.mnp_audio = reverse_audio(audio)
        elif mnp_type == 'rearrange':
            self.mnp_audio =  rearrange_audio(audio, beats, sr)
        

    def save_outputs(self):
        sr          = self.sr
        out_dir     = self.out_dir
        file_path   = self.file_path
        mnp_type    = self.mnp_type
        mnp_audio   = self.mnp_audio
        saved_name  = os.path.split(file_path)[-1] 
        saved_dir   = get_save_dir(out_dir, ['mnp', mnp_type])
        saved_path  = os.path.join(saved_dir, saved_name)
        save_audio(saved_path, mnp_audio, sr)
    
            
        return remove_prefix_dir(saved_path, out_dir)