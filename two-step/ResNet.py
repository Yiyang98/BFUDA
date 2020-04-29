import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import numpy as np
import random
import torch

seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class NUMNet(nn.Module):
    def __init__(self):
        super(NUMNet, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 10)
        self.__in_features = 500


    def forward(self, source, target):
        loss = 0
        source = self.conv_params(source)
        source = source.view(source.size(0), -1)
        source = self.fc_params(source)
        if self.training == True:
            target = self.conv_params(target)
            target = target.view(target.size(0), -1)
            target = self.fc_params(target)
            loss += mmd.mmd_rbf_accelerate(source, target)
            #loss += mmd.mmd_rbf_noaccelerate(source, target)

        source = self.classifier(source)
        return source, loss

