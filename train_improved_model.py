"""
Improved Neural Network for Polish Handwritten Word Recognition
Uses ResNet-style blocks with attention mechanism
"""

import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}\n")

# Configuration
CONFIG = {
    'csv_path': 'hpt_dataset.csv',
    'image_size': (64, 128),  # height, width - words are typically wider
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'min_class_samples': 3,  # Minimum samples per class
    'val_split': 0.2,
    'test_split': 0.2,
    'num_workers': 4,
}


# ============================================================================
# Custom Dataset
# ============================================================================

class HPTDataset(Dataset):
    """Custom dataset for HPT handwritten words"""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load image
        img_path = row['path']
        image = Image.open(img_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = row['label']
        
        return image, label


# ============================================================================
# Model Architecture
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ImprovedHandwritingCNN(nn.Module):
    """Improved CNN with ResNet blocks and attention"""
    
    def __init__(self, num_classes):
        super(ImprovedHandwritingCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Channel attention
        self.attention = ChannelAttention(512)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        x = x * self.attention(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(config):
    """Prepare datasets and dataloaders"""
    
    print("Loading dataset...")
    df = pd.read_csv(config['csv_path'])
    print(f"Total samples: {len(df)}")
    print(f"Unique words: {df['word'].nunique()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['word'])
    
    # Filter classes with too few samples
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= config['min_class_samples']].index
    df_filtered = df[df['label'].isin(valid_classes)].copy()
    
    print(f"\nAfter filtering (min {config['min_class_samples']} samples per class):")
    print(f"Samples: {len(df_filtered)}")
    print(f"Classes: {df_filtered['label'].nunique()}")
    
    # Re-encode labels after filtering
    label_encoder = LabelEncoder()
    df_filtered['label'] = label_encoder.fit_transform(df_filtered['word'])
    
    # Split data
    train_df, test_df = train_test_split(
        df_filtered, test_size=config['test_split'], 
        stratify=df_filtered['label'], random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_df, test_size=config['val_split']/(1-config['test_split']), 
        stratify=train_df['label'], random_state=42
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Calculate normalization statistics from training data
    print("\nCalculating normalization statistics...")
    temp_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
    ])
    
    temp_dataset = HPTDataset(train_df, transform=temp_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=100, shuffle=False)
    
    mean = 0.
    std = 0.
    for images, _ in tqdm(temp_loader, desc="Computing stats"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(temp_dataset)
    std /= len(temp_dataset)
    
    print(f"Mean: {mean.item():.4f}, Std: {std.item():.4f}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomRotation(5, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
    ])
    
    # Create datasets
    train_dataset = HPTDataset(train_df, transform=train_transform)
    val_dataset = HPTDataset(val_df, transform=val_test_transform)
    test_dataset = HPTDataset(test_df, transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=config['num_workers'], pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=config['num_workers'], pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=config['num_workers'], pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, label_encoder


# ============================================================================
# Training Functions
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(dataloader), 
                         'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, config, device):
    """Main training loop"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\nStarting training...\n")
    
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        print()
    
    return history


# ============================================================================
# Main
# ============================================================================

def main():
    # Prepare data
    train_loader, val_loader, test_loader, label_encoder = prepare_data(CONFIG)
    
    # Create model
    num_classes = len(label_encoder.classes_)
    model = ImprovedHandwritingCNN(num_classes).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Number of classes: {num_classes}")
    
    # Train model
    history = train_model(model, train_loader, val_loader, CONFIG, device)
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot([acc*100 for acc in history['train_acc']], label='Train Acc')
    axes[1].plot([acc*100 for acc in history['val_acc']], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved to 'training_history.png'")
    
    # Save label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to 'label_encoder.pkl'")

if __name__ == "__main__":
    main()