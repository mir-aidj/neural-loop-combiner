import warnings
warnings.filterwarnings("ignore")

import argparse, os
import numpy as np

from neural_loop_combiner.config          import settings
from neural_loop_combiner.dataset         import Dataset
from neural_loop_combiner.dataset.tagger  import Tagger
from neural_loop_combiner.utils.seperate  import tag_loop_type
from neural_loop_combiner.config.database import initialize_database
from neural_loop_combiner.utils.utils     import log_message



def loops_tag(col_tags, tracks, tag):
    if settings.NG_TYPES['selected']: 
        if tag == 1:
            log_message('Tag started')
            loops_path = [loop_path for track in tracks for loop_path in track['loops_path']] 
            for i, loop_path in enumerate(loops_path):
                log_info = [i + 1, len(loops_path)]
                find_item = col_tags.find_one({'loop_path': loop_path})
                if find_item:
                    log_message(f'{loop_path} existed', log_info)
                else:
                    loop_tag = Tagger(loop_path).tag()
                    log_message(f'{loop_path} tagged', log_info)
                    col_tags.save({'loop_path': loop_path, 'tag': loop_tag})
            log_message('Tag completed')
        harm_datas = [loop['loop_path'] for loop in col_tags.find({'tag': 'harm'})]
    else:
        harm_datas = []   
        
    return harm_datas

def dataset_creation(col_datasets, tracks, harm_datas):
    tracks_dict  = {track['file_name']: track for track in tracks}
    log_message('Create dataset started')
    dataset      = Dataset(tracks_dict, harm_datas).datas_retrieve()
    col_datasets.save(dataset)
    log_message('Create dataset completed')

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Dataset Creation')
    parser.add_argument('--tag', help='tag or not', default=1)
    
    tag          = parser.parse_args().tag
    col_loops    = initialize_database(settings.MONGODB_LOOP_COL)
    col_datasets = initialize_database(settings.MONGODB_DATASET_COL)
    col_tags     = initialize_database(settings.MONGODB_TAG_COL)
    tracks       = col_loops.find({'$where':'this.pairs_path.length >= 1'})
    
    # Loops Tag
    harm_datas   = loops_tag(col_tags, tracks, tag)
    # Dataset Creation
    dataset_creation(col_datasets, tracks, harm_datas)
    
    
    
            
            
            
            
        
    
            
            