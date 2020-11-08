import warnings
warnings.filterwarnings("ignore")

import argparse, os
import numpy as np

from neural_loop_combiner.config           import settings
from neural_loop_combiner.utils.utils      import log_message, get_file_name
from neural_loop_combiner.config.database  import initialize_database
from neural_loop_combiner.dataset.pipeline import Pipeline




def load_tracks(col_tracks):
    
    int_dir    = settings.INT_DIR
    int_files  = [int_file for int_file in os.listdir(int_dir) if int_file != '.ipynb_checkpoints']
    
    save_count, exist_count, total = 0, 0, len(int_files)
    log_message('Load tracks start')
   
    for int_file in int_files:
        int_path   = os.path.join(int_dir, int_file)
        file_name, media_type = get_file_name(int_path)        
        find_item = col_tracks.find_one({'file_name': file_name})

        if find_item:
            exist_count += 1
            log_message('Exist', [save_count, exist_count, total])
        else:
            save_count  += 1
            col_tracks.save({
                'extracted' : False, 
                'file_name' : file_name,
                'media_type': media_type,
            })
            log_message('Save', [save_count, exist_count, total])
            
    log_message('Load tracks completed')
    
    
def data_generation(col_tracks, col_loops):
    
    failed_count = 0
    break_loop   = False
    count        = failed_count
    
    failed_count = 0
    break_loop   = False
    count        = failed_count
    
    
    while 1:
        try:
            tracks   = col_tracks.find({'extracted': False})
            total    = tracks.count() 
            count    = failed_count
            
            if total == 0 or count + 1 == total:
                print('Finished.....')
                break

            for track in tracks[count:]:
                
                track_info      = track.copy() 
                file_name       = track['file_name']
                media_type      = track['media_type']
                log_info = [failed_count, count, total]
                track_loop      = Pipeline(file_name, media_type, gpu_num, log_info)
                track_loop_info = track_loop.start()
                
                track_info['extracted'] = True
                col_tracks.save(track_info)
                
                find_item = col_loops.find_one({'file_name': file_name})
                if find_item:
                    track_loop_info['_id'] = find_item['_id']
                col_loops.save(track_loop_info)

                count += 1
                log_message('Save Success', log_info)
                
        except Exception as e:
            if type(e).__name__ == 'CursorNotFound':
                log_message('Restart')
            else:
                log_message(f'Save Failed: {e}')
                failed_count += 1 

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Data Generation Pipline')
    parser.add_argument('--load'   , help='load tracks'  , default=1)
    parser.add_argument('--extract', help='extract loops', default=1)
    parser.add_argument('--gpu_num', help='gpu num'      , default=0)
    
    gpu_num = parser.parse_args().gpu_num
    load    = parser.parse_args().load
    extract = parser.parse_args().extract
    
    # Loading Tracks to MongoDB
    
    col_tracks = initialize_database(settings.MONGODB_TRACK_COL)
    if load == 1: load_tracks(col_tracks)
    
    # Execute Loops Extraction
    
    col_loops  = initialize_database(settings.MONGODB_LOOP_COL)
    if extract == 1: data_generation(col_tracks, col_loops)
    
            
            
            
            
            
        
    
            
            