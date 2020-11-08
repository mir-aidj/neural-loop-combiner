import torch
import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

class Skeleton(nn.Module):
    def __init__(self):
        super(Skeleton, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), 
                                  nn.Dropout(0.1),
                                  nn.BatchNorm2d(16),
                                  nn.PReLU(),
                                  nn.Conv2d(16, 4, 3, padding=1),
                                  nn.Dropout(0.1),
                                  nn.BatchNorm2d(4),
                                  nn.PReLU())

        self.fc = nn.Sequential(nn.Linear(4 * 128 * 173, 256),
                                nn.BatchNorm1d(256),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                nn.BatchNorm1d(128),
                                nn.PReLU(),
                                nn.Linear(128, 16))
        
        self.apply(init_weights)

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    
class CNN(nn.Module):
    def __init__(self, skeleton):
        super(CNN, self).__init__()
        self.conv = skeleton.conv
        self.fc   = nn.Sequential(skeleton.fc, 
                                  nn.BatchNorm1d(16),
                                  nn.PReLU(),
                                  nn.Linear(16, 1),
                                  nn.Sigmoid())
        self.apply(init_weights)
                
        
    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    
class SNN(nn.Module):
    def __init__(self, skeleton):
        super(SNN, self).__init__()
        self.conv = skeleton.conv
        self.fc   = skeleton.fc
        self.apply(init_weights)
                
        
    def forward_once(self, x):
        output = self.conv(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
    
    
