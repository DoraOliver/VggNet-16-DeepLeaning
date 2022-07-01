import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Modele):
    def __init__(self, features, num_classes: int = 1000, init_weights=False):
        super().__init__()
        self.features = features
        #nn.Sequential定义最后三层全连接层
        self.classfier = nn.Sequential(
            #Conv.512*7*7 --> FC.4096
            nn.Linear(512*7*7, 4096) 
        )
