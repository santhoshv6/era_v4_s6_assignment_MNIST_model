# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time
import random
import numpy as np

# Comprehensive reproducibility seeding
def set_seed(seed=42):
    """Set seeds for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For DataLoader workers
    torch.use_deterministic_algorithms(True, warn_only=True)

# Set global seed at import time
set_seed(42)

def get_model(model_num):
    """Import and return the specified model"""
    if model_num == 1:
        from model1 import Model_1
        return Model_1()
    elif model_num == 2:
        from model2 import Model_2
        return Model_2()
    elif model_num == 3:
        from model3 import Model_3
        return Model_3()
    else:
        raise ValueError("Model number must be 1, 2, or 3")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, loader, optimizer, criterion, device, epoch_num=None):
    """Train the model for one epoch with progress bar"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar for training with single line
    desc = f"Epoch {epoch_num:2d} Train" if epoch_num else "Training"
    pbar = tqdm(loader, desc=desc, ncols=80, leave=True, position=0)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
        # Update progress bar with current metrics (less frequent updates)
        if batch_idx % 50 == 0 or batch_idx == len(loader) - 1:
            current_acc = 100. * correct / total
            current_loss = train_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Acc': f'{current_acc:.1f}%'
            })
    
    return train_loss / len(loader), correct / total

def test_model(model, loader, device, epoch_num=None):
    """Test the model and return accuracy with progress bar"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar for testing with single line
    desc = f"Epoch {epoch_num:2d} Test " if epoch_num else "Testing"
    pbar = tqdm(loader, desc=desc, ncols=80, leave=True, position=0)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            
            # Update progress bar with current metrics (less frequent updates)
            if batch_idx % 5 == 0 or batch_idx == len(loader) - 1:
                current_acc = 100. * correct / total
                current_loss = test_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.3f}',
                    'Acc': f'{current_acc:.1f}%'
                })
    
    return test_loss / len(loader), correct / total

def train_model(model_num, epochs=15, lr=0.03, batch_size=128):
    """Train a specific model with advanced scheduling for 99.4% target"""
    print(f"\n{'='*60}")
    print(f"Training Model {model_num} - Target: 99.4%")
    print(f"{'='*60}")

    # Comprehensive seeding for this training run
    set_seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data split: Standard MNIST (60K train, 10K test)")
    
    # Ensure all prints are flushed before training starts
    sys.stdout.flush()
    # Seeding already handled by set_seed() function
    
    # Data transforms - no augmentation for better training fit
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load standard MNIST datasets
    train_set = datasets.MNIST('.', train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST('.', train=False, download=True, transform=test_transform)
    
    # Worker seeding function for reproducibility
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Data loaders with worker seeding for complete reproducibility
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  # Using 10K test split

    # Model
    model = get_model(model_num).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    if param_count > 8000:
        print(f"WARNING: Model has {param_count:,} parameters (>8000 limit)")
    
    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss with raw logits (more stable)

    # ENHANCED optimizer configuration for 99.4%+ guarantee (SGD works better!)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4, nesterov=True)
    # Cosine annealing with warm restart for better plateau breaking
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    
    # Training tracking
    best_acc = 0
    accuracies_above_994 = []
    training_log = []
    
    print(f"\nEpoch | Train Loss | Train Acc | Test Loss | Test Acc | LR")
    print("-" * 65)
    sys.stdout.flush()  # Ensure table header is printed before progress bars
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train (no scheduler stepping per batch)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Test (using our 10K test split)
        test_loss, test_acc = test_model(model, test_loader, device, epoch)
        
        # Step scheduler per epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc * 100,
            'test_loss': test_loss,  # Keeping same key name for compatibility
            'test_acc': test_acc * 100,  # Keeping same key name for compatibility
            'lr': current_lr
        }
        training_log.append(log_entry)
        
        # Print progress (single line per epoch)
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc*100:9.2f}% | {test_loss:9.4f} | {test_acc*100:8.2f}% | {current_lr:.6f}")
        
        # Track best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Track accuracies ≥99.4%
        if test_acc >= 99.4:
            accuracies_above_994.append(epoch)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Final analysis
    print(f"\n{'='*60}")
    print(f"Training Complete for Model {model_num}")
    print(f"{'='*60}")
    final_test_acc = training_log[-1]['test_acc'] if training_log else 0
    
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Best test accuracy: {best_acc*100:.2f}%")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Parameters: {param_count:,}")
    print(f"Training samples: 60,000")
    print(f"Test samples: 10,000")
    
    if accuracies_above_994:
        print(f"Epochs with ≥99.4% accuracy: {accuracies_above_994}")
        if len(accuracies_above_994) >= 2:
            print("✅ PASSED: Achieved ≥99.4% accuracy consistently!")
        else:
            print("❌ FAILED: Need consistent ≥99.4% accuracy in last few epochs")
    else:
        print(f"❌ FAILED: Never achieved ≥99.4% accuracy")
    
    return training_log, best_acc, accuracies_above_994

if __name__ == "__main__":
    # You can specify which model to train as command line argument
    # python train.py 1  # for Model_1
    # python train.py 2  # for Model_2  
    # python train.py 3  # for Model_3
    
    if len(sys.argv) > 1:
        model_num = int(sys.argv[1])
        if model_num in [1, 2, 3]:
            train_model(model_num)
        else:
            print("Please specify model number 1, 2, or 3")
    else:
        # Train all models by default
        for model_num in [1, 2, 3]:
            train_model(model_num)
            print("\n")  # Add space between models
