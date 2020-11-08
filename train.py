import warnings
warnings.filterwarnings("ignore")

import os
import pymongo
import torch
import librosa
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.optim import lr_scheduler

from neural_loop_combiner.config          import settings
from neural_loop_combiner.utils.utils     import log_message, load_json
from neural_loop_combiner.config.database import initialize_database
from neural_loop_combiner.models.datasets import MixedDataset, PairDataset, SingleDataset
from neural_loop_combiner.trainer.trainer import Trainer
from neural_loop_combiner.models.models   import Skeleton, CNN, SNN
from neural_loop_combiner.models.losses   import ContrastiveLoss


def load_dataset(out_dir, data_path):
    return load_json(os.path.join(out_dir, data_path))


def load_dataloader(model_type, data_type, neg_type, datasets):
    pos_datas      = datasets['pos'][data_type]
    neg_datas      = datasets['neg'][data_type][neg_type]
    torch_datasets = MixedDataset(pos_datas, neg_datas) if model_type == 'cnn' else PairDataset(pos_datas, neg_datas)
    return Data.DataLoader(dataset=torch_datasets, batch_size=batch_size, shuffle= True if data_type else False == 'train', num_workers=1)

def load_parameters(model_type, lr, batch_size, gpu_num):
    model   = CNN(Skeleton()) if model_type == 'cnn' else SNN(Skeleton())
    loss_fn = nn.BCELoss() if model_type == 'cnn' else ContrastiveLoss()
    device  = torch.device("cuda:{}".format(gpu_num))
    optimizer    = optim.Adam(model.parameters(), lr=lr)
    scheduler    = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    
    return model, loss_fn, optimizer, scheduler, device




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--gpu_num'     , help='gpu num'     , default=2)
    parser.add_argument('--lr'          , help='lr'          , default=settings.LR)
    parser.add_argument('--epochs'      , help='epochs'      , default=settings.EPOCHS)
    parser.add_argument('--batch_size'  , help='batch size'  , default=settings.BATCH_SIZE)
    parser.add_argument('--log_interval', help='log interval', default=settings.LOG_INTERVAL)
    
    parser.add_argument('--neg_type'    , help='neg type'   , default='random')
    parser.add_argument('--model_type'  , help='model type' , default='cnn')
    
    
    col_datasets = initialize_database(settings.MONGODB_DATASET_COL)
    col_models   = initialize_database(settings.MONGODB_MODEL_COL)
    datas        = col_datasets.find({}).sort('date',pymongo.DESCENDING)[0]
    datasets     = load_dataset(settings.OUT_DIR, datas['data_path'])
    
    gpu_num      = parser.parse_args().gpu_num
    log_interval = parser.parse_args().log_interval
    neg_type     = parser.parse_args().neg_type
    
    if neg_type not in datas['neg_types']:
        log_message(f'{neg_type} not exists')
    else: 
        model_type   = 'snn' if parser.parse_args().model_type != 'cnn' else parser.parse_args().model_type
        lr           = parser.parse_args().lr
        epochs       = parser.parse_args().epochs
        batch_size   = parser.parse_args().batch_size
        model, loss_fn, optimizer, scheduler, device = load_parameters(model_type, lr, batch_size, gpu_num)

        train_loader = load_dataloader(model_type, 'train', neg_type, datasets)
        val_loader   = load_dataloader(model_type, 'val'  , neg_type, datasets)

        trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, device, log_interval)
        log_message(f'Start train {model_type} with {neg_type}')

        losses_avg, losses_history = trainer.fit()    
        model_id, model_name = trainer.save_model()

        col_models.save({
            'model_id'  : model_id,
            'model_name': model_name, 
            'dataset_id': datas['date'],
            'neg_type'  : neg_type,
            'model_type': model_type, 
            'parameters': {
                'lr': lr, 
                'epochs': epochs,
                'batch_size': batch_size
            }, 
            'losses': {
                'avg'    : losses_avg, 
                'history': losses_history
            }
        })

        log_message(f'Finish train {model_type} with {neg_type}')
    
    
    
    
    
    
    

    