# MNIST CNN Models - Progressive Architecture Development

## Project Requirements
- **Target Accuracy**: ≥99.4% (consistently in last few epochs)
- **Max Epochs**: ≤15
- **Max Parameters**: <8000
- **Architecture**: Modular CNN models

## Model Evolution Strategy

This project follows a systematic approach to achieve the target performance through three progressive model iterations:

1. **Model_1**: Baseline lightweight CNN to establish parameter efficiency
2. **Model_2**: Enhanced architecture with batch normalization and dropout
3. **Model_3**: Optimized design with strategic channel management

---

### Model 1: ❌ NEEDS IMPROVEMENT

**Target**: Establish baseline lightweight CNN with minimal parameters (<2000), basic architecture to understand MNIST patterns, ensure adequate receptive field coverage.

**Result**:
- Final Accuracy: 61.50%
- Best Accuracy: 61.50%
- Parameters: 5,976
- Epochs with ≥99.4%: None
- Consistent 99.4%+ (last 3 epochs): No
- Target Met: ❌ NO

**Analysis**: Simple CNN architecture without regularization. Uses basic conv layers with GAP to minimize parameters. May struggle with generalization due to lack of batch normalization and dropout.
Requires further optimization to meet target criteria.

#### Training Logs
```

============================================================
Training Model 1 - Target: 99.4%
============================================================
Using device: cuda
Data split: Standard MNIST (60K train, 10K test)
Model parameters: 5,976

Epoch | Train Loss | Train Acc | Test Loss | Test Acc | LR
-----------------------------------------------------------------
    1 |     1.6051 |     34.37% |    1.4137 |    40.41% | 0.027136
    2 |     1.4067 |     40.56% |    1.4025 |    40.68% | 0.019639
    3 |     1.3676 |     42.16% |    1.1940 |    50.36% | 0.010371
    4 |     1.1710 |     50.75% |    1.1649 |    50.89% | 0.002874
    5 |     1.1536 |     51.06% |    1.1572 |    51.03% | 0.030000
    6 |     1.1697 |     50.79% |    1.1661 |    50.86% | 0.029266
    7 |     0.9705 |     59.39% |    0.9219 |    61.10% | 0.027136
    8 |     0.9292 |     60.79% |    0.9257 |    61.22% | 0.023819
    9 |     0.9209 |     60.99% |    0.9198 |    61.23% | 0.019639
   10 |     0.9168 |     61.05% |    0.9125 |    61.39% | 0.015005
   11 |     0.9119 |     61.11% |    0.9156 |    61.48% | 0.010371
   12 |     0.9073 |     61.20% |    0.9077 |    61.43% | 0.006191
   13 |     0.9036 |     61.27% |    0.9099 |    61.45% | 0.002874
   14 |     0.9009 |     61.31% |    0.9048 |    61.49% | 0.000744
   15 |     0.8988 |     61.36% |    0.9052 |    61.50% | 0.030000

============================================================
Training Complete for Model 1
============================================================
Total training time: 239.41 seconds
Best test accuracy: 61.50%
Final test accuracy: 61.50%
Parameters: 5,976
Training samples: 60,000
Test samples: 10,000
❌ FAILED: Never achieved ≥99.4% accuracy

```

---

### Model 2: ❌ NEEDS IMPROVEMENT

**Target**: Improve training stability and generalization with Batch Normalization and Dropout. Target 99.0%+ accuracy as stepping stone. Optimize parameter count to stay under 8000.

**Result**:
- Final Accuracy: 78.91%
- Best Accuracy: 78.91%
- Parameters: 7,808
- Epochs with ≥99.4%: None
- Consistent 99.4%+ (last 3 epochs): No
- Target Met: ❌ NO

**Analysis**: Added BN and Dropout for better training dynamics and regularization. Optimized channel progression to balance capacity and parameter count. Should show improved stability and generalization.
Requires further optimization to meet target criteria.

#### Training Logs
```

============================================================
Training Model 2 - Target: 99.4%
============================================================
Using device: cuda
Data split: Standard MNIST (60K train, 10K test)
Model parameters: 7,808

Epoch | Train Loss | Train Acc | Test Loss | Test Acc | LR
-----------------------------------------------------------------
    1 |     0.8253 |     75.20% |    0.7449 |    77.52% | 0.027136
    2 |     0.7375 |     78.25% |    0.7344 |    78.08% | 0.019639
    3 |     0.7259 |     78.71% |    0.7347 |    78.16% | 0.010371
    4 |     0.7189 |     78.88% |    0.7219 |    78.79% | 0.002874
    5 |     0.7126 |     79.11% |    0.7200 |    78.57% | 0.030000
    6 |     0.7270 |     78.59% |    0.7290 |    78.43% | 0.029266
    7 |     0.7243 |     78.64% |    0.7272 |    78.64% | 0.027136
    8 |     0.7209 |     78.85% |    0.7254 |    78.82% | 0.023819
    9 |     0.7172 |     78.90% |    0.7275 |    78.27% | 0.019639
   10 |     0.7163 |     78.92% |    0.7258 |    78.65% | 0.015005
   11 |     0.7115 |     79.09% |    0.7206 |    78.83% | 0.010371
   12 |     0.7080 |     79.20% |    0.7177 |    78.86% | 0.006191
   13 |     0.7049 |     79.29% |    0.7173 |    78.72% | 0.002874
   14 |     0.7030 |     79.35% |    0.7158 |    78.84% | 0.000744
   15 |     0.7018 |     79.40% |    0.7148 |    78.91% | 0.030000

============================================================
Training Complete for Model 2
============================================================
Total training time: 244.14 seconds
Best test accuracy: 78.91%
Final test accuracy: 78.91%
Parameters: 7,808
Training samples: 60,000
Test samples: 10,000
❌ FAILED: Never achieved ≥99.4% accuracy

```

---

### Model 3: ✅ SUCCESS

**Target**: Achieve consistent 99.4%+ accuracy with optimized architecture. Use strategic parameter management and efficient design. Focus on receptive field optimization.

**Result**:
- Final Accuracy: 99.44%
- Best Accuracy: 99.47%
- Parameters: 7,928
- Epochs with ≥99.4%: None
- Consistent 99.4%+ (last 3 epochs): Yes
- Target Met: ✅ YES

**Analysis**: Final optimized model with careful parameter management. Uses efficient channel progression, strategic dropout placement, and 1x1 convolutions for parameter reduction while maintaining capacity.
The model successfully met all target criteria.

#### Training Logs
```

============================================================
Training Model 3 - Target: 99.4%
============================================================
Using device: cuda
Data split: Standard MNIST (60K train, 10K test)
Model parameters: 7,928

Epoch | Train Loss | Train Acc | Test Loss | Test Acc | LR
-----------------------------------------------------------------
    1 |     0.4309 |     85.20% |    0.0657 |    98.08% | 0.027136
    2 |     0.0551 |     98.42% |    0.0488 |    98.53% | 0.019639
    3 |     0.0424 |     98.80% |    0.0340 |    98.95% | 0.010371
    4 |     0.0329 |     99.09% |    0.0292 |    99.16% | 0.002874
    5 |     0.0256 |     99.32% |    0.0241 |    99.38% | 0.030000
    6 |     0.0443 |     98.80% |    0.0429 |    98.61% | 0.029266
    7 |     0.0412 |     98.86% |    0.0307 |    99.09% | 0.027136
    8 |     0.0383 |     98.90% |    0.0411 |    98.85% | 0.023819
    9 |     0.0358 |     98.99% |    0.0405 |    98.80% | 0.019639
   10 |     0.0306 |     99.18% |    0.0307 |    99.05% | 0.015005
   11 |     0.0282 |     99.21% |    0.0285 |    99.19% | 0.010371
   12 |     0.0235 |     99.37% |    0.0251 |    99.30% | 0.006191
   13 |     0.0204 |     99.52% |    0.0208 |    99.40% | 0.002874
   14 |     0.0171 |     99.63% |    0.0196 |    99.47% | 0.000744
   15 |     0.0157 |     99.66% |    0.0196 |    99.44% | 0.030000

============================================================
Training Complete for Model 3
============================================================
Total training time: 239.46 seconds
Best test accuracy: 99.47%
Final test accuracy: 99.44%
Parameters: 7,928
Training samples: 60,000
Test samples: 10,000
✅ Sucess: Achieved ≥99.4% accuracy

```

---

## Receptive Field Calculations

### Model_1 Receptive Field
- **Final Receptive Field**: 20x20

| Layer | Output Size | Receptive Field |
|-------|-------------|-----------------|
| Input | 28 | 1 |
| Conv1 (3x3, p=0) | 26 | 3 |
| Conv2 (3x3, p=0) | 24 | 5 |
| MaxPool (2x2) | 12 | 6 |
| Conv3 (1x1, p=0) | 12 | 6 |
| Conv4 (3x3, p=0) | 10 | 10 |
| MaxPool (2x2) | 5 | 12 |
| Conv5 (1x1, p=0) | 5 | 12 |
| Conv6 (3x3, p=0) | 3 | 16 |
| Direct Classification | 1 | 20 |

### Model_2 Receptive Field
- **Final Receptive Field**: 28x28

| Layer | Output Size | Receptive Field |
|-------|-------------|-----------------|
| Input | 28 | 1 |
| Conv1+BN (3x3, p=0) | 26 | 3 |
| Conv2+BN (3x3, p=0) | 24 | 5 |
| MaxPool (2x2) | 12 | 6 |
| Conv3 (1x1, p=0) | 12 | 6 |
| Conv4+BN (3x3, p=0) | 10 | 10 |
| Conv5+BN (3x3, p=0) | 8 | 14 |
| Conv6+BN (3x3, p=0) | 6 | 18 |
| MaxPool (2x2) | 3 | 20 |
| Conv7 (3x3, p=0) | 1 | 28 |
| Direct Classification | 1 | 28 |

### Model_3 Receptive Field
- **Final Receptive Field**: 22x22

| Layer | Output Size | Receptive Field |
|-------|-------------|-----------------|
| Input | 28 | 1 |
| Conv1+BN (3x3, p=0) | 26 | 3 |
| Conv2+BN (3x3, p=0) | 24 | 5 |
| MaxPool (2x2) | 12 | 6 |
| Conv3 (1x1, p=0) | 12 | 6 |
| Conv4+BN (3x3, p=0) | 10 | 10 |
| Conv5+BN (3x3, p=0) | 8 | 14 |
| Conv6+BN (3x3, p=0) | 6 | 18 |
| Conv7+BN (3x3, p=1) | 6 | 22 |
| Conv8 (1x1, p=0) | 6 | 22 |
| GAP (6x6→1x1) | 1 | 22 |
| Enhanced Classification | 1 | 22 |

## Results Summary

| Model | Parameters | Best Accuracy | Final Accuracy | Epochs ≥99.4% | Consistent 99.4%+ | Target Met |
|-------|------------|---------------|----------------|----------------|-------------------|------------|
| Model_1 | 5,976 | 61.50% | 61.50% | 0 | ❌ | ❌ |
| Model_2 | 7,808 | 78.91% | 78.91% | 0 | ❌ | ❌ |
| Model_3 | 7,928 | 99.47% | 99.44% | 0 | ✅ | ✅ |

## Final Assessment

- **Models meeting all criteria**: 1/3
- **Project status**: ✅ COMPLETED

### Key Learnings
1. **Parameter Efficiency**: Achieving high accuracy with <8000 parameters requires careful architecture design
2. **Regularization**: Batch normalization and dropout are crucial for stable training
3. **Progressive Improvement**: Each model iteration addressed specific limitations of the previous version
4. **Consistency**: Achieving consistent 99.4%+ accuracy is more challenging than one-time peak performance

### GitHub Repository Structure
```
S6_assignment/
├── model1.py          # Baseline CNN model
├── model2.py          # Enhanced model with BN/Dropout
├── model3.py          # Optimized final model
├── train.py           # Training script for all models
├── utils.py           # Utility functions
├── README.md          # This documentation
└── MNIST_CNN_Training.ipynb  # Colab notebook for training
```

---
*Generated automatically from training results*
