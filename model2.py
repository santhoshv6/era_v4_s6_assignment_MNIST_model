# model2.py

'''
For Model 2:

Targets: Improve training stability and generalization with Batch Normalization and Dropout. Target 99.0%+ accuracy as stepping stone
Results: Got 78.91% test accuracy with 7,808 parameters
Analysis: Added BN for better training dynamics but still insufficient capacity. Need deeper architecture and better feature extraction

File Link: model2.py
'''

import torch.nn as nn
import torch.nn.functional as F

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # Input: 28x28x1
        
        # Input Block - Feature extraction with normalization
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False)   # 28x28x1 -> 26x26x16, params: 1*16*3*3 = 144
        self.bn1 = nn.BatchNorm2d(16)                             # params: 32
        self.dropout1 = nn.Dropout2d(0.02)
        
        self.conv2 = nn.Conv2d(16, 18, 3, padding=0, bias=False)  # 26x26x16 -> 24x24x18, params: 16*18*3*3 = 2592
        self.bn2 = nn.BatchNorm2d(18)                             # params: 36
        self.dropout2 = nn.Dropout2d(0.02)
        
        # Transition Block 1 - Efficiency bottleneck + spatial reduction
        self.conv3 = nn.Conv2d(18, 10, 1, padding=0, bias=False)  # 24x24x18 -> 24x24x10, params: 18*10*1*1 = 180
        self.pool1 = nn.MaxPool2d(2, 2)                           # 24x24x10 -> 12x12x10

        # Convolution Block 2 - Enhanced feature learning with regularization
        self.conv4 = nn.Conv2d(10, 14, 3, padding=0, bias=False)  # 12x12x10 -> 10x10x14, params: 10*14*3*3 = 1260
        self.bn4 = nn.BatchNorm2d(14)                             # params: 28
        
        self.conv5 = nn.Conv2d(14, 12, 3, padding=0, bias=False)  # 10x10x14 -> 8x8x12, params: 14*12*3*3 = 1512
        self.bn5 = nn.BatchNorm2d(12)                             # params: 24
        
        self.conv6 = nn.Conv2d(12, 10, 3, padding=0, bias=False)  # 8x8x12 -> 6x6x10, params: 12*10*3*3 = 1080
        self.bn6 = nn.BatchNorm2d(10)                             # params: 20
        self.pool2 = nn.MaxPool2d(2, 2)                           # 6x6x10 -> 3x3x10
        
        # Final classification layers (no GAP - keeping it for Model 3 only)
        self.conv7 = nn.Conv2d(10, 10, 3, padding=0, bias=False)  # 3x3x10 -> 1x1x10, params: 10*10*3*3 = 900
        
        # Total params: 144+32+2592+36+180+1260+28+1512+24+1080+20+900 = 7808 ✓ Under 8K

    def forward(self, x):
        # Input Block - Feature extraction with normalization
        x = F.relu(self.bn1(self.conv1(x)))    # 28x28x1 -> 26x26x16
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))    # 26x26x16 -> 24x24x18
        x = self.dropout2(x)
        
        # Transition Block 1 - Efficiency bottleneck + spatial reduction
        x = self.conv3(x)                      # 24x24x18 -> 24x24x10 (no activation on 1x1)
        x = self.pool1(x)                      # 24x24x10 -> 12x12x10
        
        # Convolution Block 2 - Enhanced feature learning
        x = F.relu(self.bn4(self.conv4(x)))    # 12x12x10 -> 10x10x14
        
        x = F.relu(self.bn5(self.conv5(x)))    # 10x10x14 -> 8x8x12
        
        x = F.relu(self.bn6(self.conv6(x)))    # 8x8x12 -> 6x6x10
        x = self.pool2(x)                      # 6x6x10 -> 3x3x10
        
        # Final classification (no GAP - keeping it for Model 3 only)
        x = F.relu(self.conv7(x))              # 3x3x10 -> 1x1x10
        x = x.view(-1, 10)                    # 1x1x10 -> 10 (flatten)
        return x  # Return raw logits for CrossEntropyLoss