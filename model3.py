# model3.py

'''
For Model 3:

Targets: Achieve consistent 99.4%+ accuracy with optimized architecture, strategic parameter management and efficient design
Results: Got 99.47% best accuracy, 99.44% final accuracy with 7,928 parameters, achieved consistent 99.4%+ in last epochs
Analysis: Final optimized model with GAP, deep architecture, and strategic channel progression successfully met all criteria

File Link: model3.py
'''

import torch.nn as nn
import torch.nn.functional as F

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        # Input: 28x28x1
        
        # Input Block - Enhanced feature extraction  
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False)   # 28x28x1 -> 26x26x16, params: 1*16*3*3 = 144, RF=3
        self.bn1 = nn.BatchNorm2d(16)                             # params: 32
        
        # Convolution Block 1 - Optimized feature learning
        self.conv2 = nn.Conv2d(16, 18, 3, padding=0, bias=False)  # 26x26x16 -> 24x24x18, params: 16*18*3*3 = 2592, RF=5
        self.bn2 = nn.BatchNorm2d(18)                             # params: 36
        
        # Transition Block 1 - Efficient bottleneck
        self.conv3 = nn.Conv2d(18, 10, 1, padding=0, bias=False)  # 24x24x18 -> 24x24x10, params: 18*10*1*1 = 180, RF=5
        self.pool1 = nn.MaxPool2d(2, 2)                           # 24x24x10 -> 12x12x10, RF=6
        
        # Convolution Block 2 - Deep feature learning (optimized)
        self.conv4 = nn.Conv2d(10, 12, 3, padding=0, bias=False)  # 12x12x10 -> 10x10x12, params: 10*12*3*3 = 1080, RF=10
        self.bn4 = nn.BatchNorm2d(12)                             # params: 24
        
        self.conv5 = nn.Conv2d(12, 14, 3, padding=0, bias=False)  # 10x10x12 -> 8x8x14, params: 12*14*3*3 = 1512, RF=14
        self.bn5 = nn.BatchNorm2d(14)                            # params: 28
        
        self.conv6 = nn.Conv2d(14, 10, 3, padding=0, bias=False)  # 8x8x14 -> 6x6x10, params: 14*10*3*3 = 1260, RF=18
        self.bn6 = nn.BatchNorm2d(10)                             # params: 20
        
        # Final convolution with padding to maintain spatial size
        self.conv7 = nn.Conv2d(10, 10, 3, padding=1, bias=False)  # 6x6x10 -> 6x6x10, params: 10*10*3*3 = 900, RF=22
        self.bn7 = nn.BatchNorm2d(10)                             # params: 20
        
        # Additional 1x1 refinement layer for enhanced capability
        self.conv8 = nn.Conv2d(10, 10, 1, padding=0, bias=False)  # 6x6x10 -> 6x6x10, params: 10*10*1*1 = 100, RF=22

        # OUTPUT BLOCK - Global Average Pooling
        self.gap = nn.AvgPool2d(kernel_size=6)                    # 6x6x10 -> 1x1x10 (Global Average Pooling)
        
        # Total params: 144+32+2592+36+180+1080+24+1512+28+1260+20+900+20+100 = 7928 ✓ Under 8K!

    def forward(self, x):
        # Input Block - Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))    # 28x28x1 -> 26x26x16, RF=3
        
        # Convolution Block 1 - Initial feature learning
        x = F.relu(self.bn2(self.conv2(x)))    # 26x26x16 -> 24x24x18, RF=5
        
        # Transition Block 1 - Efficiency bottleneck + spatial reduction
        x = self.conv3(x)                      # 24x24x18 -> 24x24x10 (no activation on 1x1)
        x = self.pool1(x)                      # 24x24x10 -> 12x12x10, RF=6
        
        # Convolution Block 2 - Deep feature learning
        x = F.relu(self.bn4(self.conv4(x)))    # 12x12x10 -> 10x10x12, RF=10
        
        x = F.relu(self.bn5(self.conv5(x)))    # 10x10x12 -> 8x8x14, RF=14
        
        x = F.relu(self.bn6(self.conv6(x)))    # 8x8x14 -> 6x6x10, RF=18
        
        # Final convolution with padding (enhanced)
        x = F.relu(self.bn7(self.conv7(x)))    # 6x6x10 -> 6x6x10, RF=22
        
        # Additional 1x1 refinement layer for enhanced capability
        x = F.relu(self.conv8(x))              # 6x6x10 -> 6x6x10 (1x1 feature refinement), RF=22
        
        # OUTPUT BLOCK - Global Average Pooling
        x = self.gap(x)                        # 6x6x10 -> 1x1x10 (Global Average Pooling)
        x = x.view(-1, 10)                     # 1x1x10 -> 10 (flatten)
        return x  # Return raw logits for CrossEntropyLoss