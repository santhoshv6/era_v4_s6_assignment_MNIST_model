# model1.py

'''
For Model 1:

Targets: Establish baseline lightweight CNN with minimal parameters, basic architecture to understand MNIST patterns, ensure adequate receptive field coverage
Results: Got 61.50% test accuracy with 5,976 parameters
Analysis: Simple CNN architecture without regularization struggles with generalization. Requires batch normalization and better feature extraction

File Link: model1.py
'''

import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Input: 28x28x1
        
        # Input Block - Consistent with Model 3 but simpler
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False)   # 28x28x1 -> 26x26x16, params: 1*16*3*3 = 144
        self.conv2 = nn.Conv2d(16, 18, 3, padding=0, bias=False)  # 26x26x16 -> 24x24x18, params: 16*18*3*3 = 2592
        
        # Transition Block 1 - Efficiency bottleneck  
        self.conv3 = nn.Conv2d(18, 10, 1, padding=0, bias=False)  # 24x24x18 -> 24x24x10, params: 18*10*1*1 = 180
        self.pool1 = nn.MaxPool2d(2, 2)                           # 24x24x10 -> 12x12x10

        # Convolution Block 2 - Efficient progression
        self.conv4 = nn.Conv2d(10, 12, 3, padding=0, bias=False)  # 12x12x10 -> 10x10x12, params: 10*12*3*3 = 1080
        self.pool2 = nn.MaxPool2d(2, 2)                           # 10x10x12 -> 5x5x12
        
        self.conv5 = nn.Conv2d(12, 10, 3, padding=0, bias=False)  # 5x5x12 -> 3x3x10, params: 12*10*3*3 = 1080
        
        # OUTPUT BLOCK - Direct classification
        self.conv6 = nn.Conv2d(10, 10, 3, padding=0, bias=False)  # 3x3x10 -> 1x1x10, params: 10*10*3*3 = 900
        
        # Total params: 144+2592+180+1080+1080+900 = 5976 (under 8K!) ✅

    def forward(self, x):
        # Input Block - Initial feature extraction
        x = F.relu(self.conv1(x))    # 28x28x1 -> 26x26x16
        x = F.relu(self.conv2(x))    # 26x26x16 -> 24x24x18
        
        # Transition Block 1 - Bottleneck + spatial reduction
        x = self.conv3(x)            # 24x24x18 -> 24x24x10 (no activation on 1x1)
        x = self.pool1(x)            # 24x24x10 -> 12x12x10
        
        # Convolution Block 2 - Feature learning
        x = F.relu(self.conv4(x))    # 12x12x10 -> 10x10x12
        x = self.pool2(x)            # 10x10x12 -> 5x5x12
        
        x = F.relu(self.conv5(x))    # 5x5x12 -> 3x3x10
        
        # OUTPUT BLOCK - Direct classification
        x = F.relu(self.conv6(x))    # 3x3x10 -> 1x1x10
        
        # Classifier: 1x1x10 -> 10
        x = x.view(x.size(0), -1)             # 1x1x10 -> 10 (flatten)
        return x  # Return raw logits for CrossEntropyLoss
