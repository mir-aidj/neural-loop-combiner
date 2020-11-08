import warnings
warnings.filterwarnings("ignore")

import os, librosa
import numpy as np
import tensorly as tl

from loopextractor.loopextractor.loopextractor import get_downbeats, make_spectral_cube, validate_template_sizes
from loopextractor.loopextractor.loopextractor import get_loop_signal, create_loop_spectrum, choose_bar_to_reconstruct

from neural_loop_combiner.utils.utils          import get_save_dir, save_audio, save_np, load_np, log_message, discrete_matrix
from neural_loop_combiner.utils.comparison     import spec_similarity, snr_cal
from neural_loop_combiner.config               import settings

tl.set_backend('pytorch')

class Extractor:
    
    def __init__(self, file_name, media_type, gpu_num, log_info=[]):
        
        self.sr          = settings.SR
        self.cache       = settings.CACHE
        self.log         = settings.LOG
        self.int_dir     = settings.INT_DIR
        self.out_dir     = settings.OUT_DIR
        self.exist_thre  = settings.EXISTED_THRESHOLD
        
        self.log_info    = log_info
        self.file_name   = file_name
        self.media_type  = media_type
        self.gpu_num     = gpu_num
        self.file_path   = os.path.join(self.int_dir, f'{file_name}.{media_type}')
        self.track_audio = librosa.load(self.file_path, self.sr)[0]
        
        if self.log: log_message('Extractor started', self.log_info)
        self.decompose()
        if self.log: log_message('Decomposition completed', self.log_info)
        self.layout_retrieve()
        self.template = discrete_matrix(self.layout, self.exist_thre)
        if self.log: log_message('Layout & Template retrive completed', self.log_info)
        self.loop_extract()
        if self.log: log_message('Extractor completed', self.log_info)
            
            
    def decompose(self):
        sr          = self.sr
        out_dir     = self.out_dir
        cache       = self.cache
        log         = self.log
        log_info    = self.log_info
        gpu_num     = self.gpu_num
        file_name   = self.file_name
        track_audio = self.track_audio
        
        downbeat_path   = os.path.join(get_save_dir(out_dir, ['extracted', 'downbeat']), f'{file_name}.npy')
        core_path       = os.path.join(get_save_dir(out_dir, ['extracted', 'core'])    , f'{file_name}.npy')
        factors_path    = os.path.join(get_save_dir(out_dir, ['extracted', 'factors']) , f'{file_name}.npy')
    
        downbeat_times  = load_np(downbeat_path, cache)
        core            = load_np(core_path    , cache)
        factors         = load_np(factors_path , cache)
        
        
        downbeat_times  = downbeat_times if type(downbeat_times) != type(None) else get_downbeats(track_audio)
        downbeat_frames = librosa.time_to_samples(downbeat_times, sr=sr)
        
        if log: log_message('Downbeat completed', log_info)
        spectral_cube   = make_spectral_cube(track_audio, downbeat_frames)
        n_sounds, n_rhythms, n_loops = validate_template_sizes(spectral_cube, n_templates=[0,0,0])
        if type(core) == type(None) or type(factors) == type(None):
            core, factors = tl.decomposition.non_negative_tucker(tl.tensor(np.abs(spectral_cube), device=f'cuda:{gpu_num}'), 
                                                                [n_sounds, n_rhythms, n_loops], n_iter_max=500, verbose=True)
            core       = np.array(core.detach().to('cpu').numpy())
            factors    = [np.array(factor.detach().to('cpu').numpy()) for factor in factors]
        
        if cache:
            save_np(downbeat_path, downbeat_times)
            save_np(core_path    , core)
            save_np(factors_path , factors)
            
        self.n_loops = n_loops
        self.spectral_cube = spectral_cube
        self.core    = core
        self.factors = factors
        self.downbeat_frames = downbeat_frames
        
        
    def get_ith_loop_info(self, index):
        return self.bar_audios[index], self.loops[index], self.snr_scores[index]
        
    def layout_retrieve(self):
        self.layout = self.factors[2]
    
    def loop_reconstruct(self, ith_loop):
        n_loops = self.n_loops
        core    = self.core
        factors = self.factors
        track_audio     = self.track_audio
        spectral_cube   = self.spectral_cube
        downbeat_frames = self.downbeat_frames
        
        if ith_loop >= n_loops:
            return None, None
        loop_spec = create_loop_spectrum(factors[0], factors[1], core[:,:,ith_loop])
        bar_index     = choose_bar_to_reconstruct(factors[2], ith_loop)
        ith_loop_signal = get_loop_signal(loop_spec, spectral_cube[:,:,bar_index])
        bar_audio       = track_audio[downbeat_frames[bar_index]: downbeat_frames[bar_index+1]]
        snr_score       = snr_cal(bar_audio, ith_loop_signal)
        
        return bar_audio, ith_loop_signal, loop_spec, snr_score

    def loop_extract(self):
        n_loops = self.n_loops
        
        bar_audios, loops, stretch_loops, specs, snr_scores = {}, {}, {}, {}, {}
        
        for ith_loop in range(n_loops):
            bar_audio, loop, spec, snr_score  = self.loop_reconstruct(ith_loop)
            loops[ith_loop]         = loop
            specs[ith_loop]         = spec
            bar_audios[ith_loop]    = bar_audio
            snr_scores[ith_loop]    = snr_score
            
        self.bar_audios = bar_audios
        self.loops      = loops
        self.specs      = specs
        self.snr_scores = snr_scores
        
    def get_features(self):
        return {
            'layout': self.layout, 
            'loops' : self.loops ,
            'specs' : self.specs ,
            'template': self.template,
            'snr_scores': self.snr_scores,
            'track_audio' : self.track_audio , 
        }

        