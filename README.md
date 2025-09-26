# MNIST CNN Assignment - ERA V4 Session 6 (Custom 50K/10K Split)

## Assignment Requirements
- **Target Accuracy**: ≥99.4% (consistently in last few epochs)
- **Max Epochs**: ≤15
- **Max Parameters**: <8000
- **Data Split**: Custom 50K training + 10K test (from 60K MNIST training set)
- **Architecture**: Modular CNN models

## Data Split Strategy
**Key Change**: Instead of using the official MNIST test set, we create a custom split:
- **Training**: 50,000 samples from MNIST training set
- **Test**: 10,000 samples from MNIST training set  
- **Benefits**: Better experimental control, more reliable test metrics, proper train/test split

## Model Evolution Strategy

### Model 1: Basic Lightweight CNN
**Target**: Establish baseline with minimal parameters using 50K/10K split
**Result**: [To be updated after training]
**Analysis**: [To be updated after training]

### Model 2: Enhanced with Batch Normalization and Dropout
**Target**: Improve training stability and reduce overfitting on custom split
**Result**: [To be updated after training]
**Analysis**: [To be updated after training]

### Model 3: Optimized Architecture with Better Receptive Field
**Target**: Achieve ≥99.4% test accuracy consistently with optimized design
**Result**: [To be updated after training]
**Analysis**: [To be updated after training]

## Training Logs

### Model 1 Training Logs
```
[Training logs will be updated here]
```

### Model 2 Training Logs
```
[Training logs will be updated here]
```

### Model 3 Training Logs
```
[Training logs will be updated here]
```

## Receptive Field Calculations

### Model 1 Receptive Field
- Input: 28x28
- Conv1 (3x3): RF = 3
- Conv2 (3x3): RF = 5  
- Conv3 (3x3): RF = 7
- Final RF: 7x7

### Model 2 Receptive Field
[To be calculated]

### Model 3 Receptive Field
[To be calculated]

## Parameter Count Summary
- Model 1: [To be calculated]
- Model 2: [To be calculated]
- Model 3: [To be calculated]

## Results Summary
| Model | Parameters | Best Accuracy | Epochs to 99.4% | Consistent 99.4%+ |
|-------|------------|---------------|------------------|-------------------|
| Model 1 | TBD | TBD | TBD | TBD |
| Model 2 | TBD | TBD | TBD | TBD |
| Model 3 | TBD | TBD | TBD | TBD |