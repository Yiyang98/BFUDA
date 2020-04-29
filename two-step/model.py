import torch.nn as nn
import numpy as np
import random
import torch
###这里放的是cL训练模型

seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class UMNet(nn.Module):
    def __init__(self):
        super(UMNet, self).__init__()
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


    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return y

    def output_num(self):
        return self.__in_features

