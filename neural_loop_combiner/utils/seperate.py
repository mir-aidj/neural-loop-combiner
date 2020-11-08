import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os
import librosa
import filetype

from spleeter.separator     import Separator
from spleeter.audio.adapter import get_default_audio_adapter
from neural_loop_combiner.utils.comparison import ssim_similarity
from neural_loop_combiner.utils.utils      import check_files_exist
from neural_loop_combiner.utils.utils      import save_audio, get_save_dir

def ssp(file_path, sr, cache, out_dir):
    separator    = Separator('spleeter:5stems', multiprocess=False)
    audio_loader = get_default_audio_adapter()
    file_path    = os.path.join(out_dir, file_path)
    waveform, _  = audio_loader.load(file_path, sample_rate=sr)
    prediction   = separator.separate(waveform)

    file_name    = os.path.split(file_path)[-1].split(f'.{filetype.guess(file_path).extension}')[0]
    cache_dir    = get_save_dir(out_dir, ['ssp', file_name])
    cache_paths  = [os.path.join(cache_dir, f'{path}.wav') for path in ['track', 'perc', 'bass', 'harm']] 
    paths_exist  = check_files_exist(cache_paths)
        
    if cache and paths_exist:
        ssp_audios = map(lambda cache_path: librosa.load(cache_path, sr)[0], cache_paths)
    else:
        audio  = waveform[:, 0]    
        perc   = prediction['drums'][:, 0]
        bass   = prediction['bass'][:, 0]
        harm   = prediction['piano'][:, 0] + prediction['vocals'][:, 0] + prediction['other'][:, 0]
        ssp_audios = [audio, perc, bass, harm]
            
    if cache and not paths_exist:
        map(lambda index: save_audio(cache_paths[index], ssp_audios[index], sr), range(len(ssp_audios)))
    
    return ssp_audios


def tag_loop_type(file_path, sr, cache, out_dir):  
    ssp_audios = ssp(file_path, sr, cache, out_dir)
    audio, perc, bass, harm = ssp_audios
    
    perc_score = ssim_similarity(audio, perc)
    harm_score = ssim_similarity(audio, harm)
    bass_score = ssim_similarity(audio, bass)
    
    if perc_score < 0.5 and bass_score < 0.5:
        return 'harm'
    else:
        if perc_score > harm_score:
            if perc_score > bass_score:
                return 'perc'
            else:
                return 'bass'
        else:
            return 'harm'
        
