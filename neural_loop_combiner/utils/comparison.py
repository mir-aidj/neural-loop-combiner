import imagehash
from PIL import Image
from skimage.measure import compare_ssim
from imagehash import average_hash, phash, dhash, whash
from mir_eval.separation import bss_eval_sources_framewise

from neural_loop_combiner.config import settings


def ssim_similarity(array_1, array_2):
    if len(array_1) > len(array_2):
        return compare_ssim(array_1[:len(array_2)], array_2)
    else:
        return compare_ssim(array_1, array_2[:len(array_1)])


def spec_similarity(spec1, spec2, hash_type=settings.HASH_TYPE):
    img1, img2  = Image.fromarray(spec1), Image.fromarray(spec2)
    
    if hash_type == 'ahash':
        hash1, hash2 = average_hash(img1), average_hash(img2)
    elif hash_type == 'phash':
        hash1, hash2 = phash(img1), phash(img2)
    elif hash_type == 'dhash':
        hash1, hash2 = dhash(img1), dhash(img2)
    elif hash_type == 'whash':
        hash1, hash2 = whash(img1), whash(img2)
    
    return hash1 - hash2

def snr_cal(ref_audio, estm_audio):
    
    if len(ref_audio) > len(estm_audio):
        ref_audio = ref_audio[:len(estm_audio)]
    elif len(ref_audio) < len(estm_audio):
        estm_audio = estm_audio[:len(ref_audio)]
    
    return bss_eval_sources_framewise(ref_audio, estm_audio)[0][0][0]