from builtins import super
from collections import OrderedDict

import torch.nn as nn


class ModelCNN(nn.Module):

    def __init__(self,labels):
        super(ModelCNN, self).__init__()

        self.convnet=nn.Sequential(OrderedDict([
            ('c1',nn.Conv2d(1,128,kernel_size=(3,3))),#128
            ('relu1', nn.ReLU()),
            ('c2', nn.Conv2d(128, 128, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=(2,2))),
            ('c3', nn.Conv2d(128, 256, 5)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2,2)))
        ]))
        # self.dropout=nn.Dropout(p=0.2) # For Dropout regularization
        self.fc = nn.Sequential(OrderedDict([
            ('f4',nn.Linear(2304,128)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(128, labels)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        out=self.convnet(x)
        out=out.view(x.size(0),-1)
        # self.fc.float() #Only use in CUDA
        out=self.fc(out)
        # out = self.dropout(self.fc(out))
        return out

