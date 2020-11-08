import warnings
warnings.filterwarnings("ignore")

import os
import torch
import librosa
import datetime
import numpy as np

from neural_loop_combiner.config          import settings
from neural_loop_combiner.utils.utils     import get_save_dir, save_audio, save_np
from neural_loop_combiner.utils.utils     import log_message
from neural_loop_combiner.models.models   import Skeleton, CNN, SNN

class Trainer:
    
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, device, log_interval):
        
        self.sr           = settings.SR
        self.log          = settings.LOG
        self.out_dir      = settings.OUT_DIR
        
        self.model        = model.to(device)
        self.loss_fn      = loss_fn
        self.epochs       = epochs
        self.device       = device
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.log_interval = log_interval
        
    def fit(self):
        device       = self.device
        scheduler    = self.scheduler  
        epochs       = self.epochs
        model        = self.model
        
        losses_history = { 'train': [], 'val'  : [] }
        for epoch in range(0, epochs):
            scheduler.step()
            train_loss = self.train()
            val_loss   = self.validate()
            log_message(f'Train Loss: {train_loss}, Val Loss: {val_loss}', [epoch + 1, epochs])
            
            losses_history['val'].append(val_loss)
            losses_history['train'].append(train_loss)
        losses_avg = {data_type: np.mean(np.array(losses_history[data_type])) for data_type in losses_history}
        
        return losses_avg, losses_history
            
        
        
        
    def train(self):
        log          = self.log
        device       = self.device
        model        = self.model
        loss_fn      = self.loss_fn
        scheduler    = self.scheduler
        optimizer    = self.optimizer
        train_loader = self.train_loader
        log_interval = self.log_interval
        
        model.train()
        train_loss = 0
        
        for batch_idx, (*datas, targets) in enumerate(train_loader):
            inputs, *others = datas
            if not type(inputs) in (tuple, list):
                inputs = (inputs,)
                    
            if device:
                inputs = tuple(i.to(device) for i in inputs)
                if targets is not None:
                    targets = targets.to(device)
                
            targets = torch.tensor(targets, dtype=torch.float, device=device)
            optimizer.zero_grad()
            outputs = model(*inputs)
                
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
                    
            loss_inputs = outputs
            if targets is not None:
                targets = (targets,)
                loss_inputs += targets
                    
            loss = loss_fn(*loss_inputs)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0 & log: log_message(f'Loss: {loss.item()}', [batch_idx, len(train_loader)])
                
        self.model = model
        return train_loss / (batch_idx + 1)
    
    
    def validate(self):
        device      = self.device
        model       = self.model
        loss_fn     = self.loss_fn
        scheduler   = self.scheduler
        optimizer   = self.optimizer
        val_loader  = self.val_loader
        val_loss    = 0
        with torch.no_grad():
            model.eval()
            for batch_idx, (*datas, targets) in enumerate(val_loader):
                inputs, *others = datas
                if not type(inputs) in (tuple, list):
                    inputs = (inputs,)

                if device:
                    inputs = tuple(i.to(device) for i in inputs)
                    if targets is not None:
                        targets = targets.to(device)

                targets = torch.tensor(targets, dtype=torch.float, device=device)
                outputs = model(*inputs)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)

                loss_inputs = outputs
                if targets is not None:
                    targets = (targets,)
                    loss_inputs += targets

                loss      = loss_fn(*loss_inputs)
                val_loss += loss.item()
        return val_loss / (batch_idx + 1)
            
    
    
    
    def save_model(self):
    
        log        = self.log
        model      = self.model
        out_dir    = self.out_dir
            
        model_dir  = get_save_dir(out_dir, ['models'])
        model_id   = datetime.datetime.utcnow()
        model_name = f'{model_id}.pkl'
        model_path = os.path.join(model_dir, model_name)
            
        torch.save(model.state_dict(), model_path)
        if log: log_message(f'Saved modle {model_name}')
            
        return model_id, model_name

    
    
        