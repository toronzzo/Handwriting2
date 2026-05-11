"""
Inference Script for Trained Handwriting Recognition Model
Load a trained model and make predictions on new images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# Model Architecture 
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
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.attention = ChannelAttention(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        
        x = x * self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Inference Functions

class HandwritingRecognizer:
    """Class for handwriting recognition inference"""
    
    def __init__(self, model_path, label_encoder_path, device=None):
        """
        Initialize the recognizer
        
        Args:
            model_path: Path to saved model checkpoint
            label_encoder_path: Path to saved label encoder
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load label encoder
        print(f"Loading label encoder from {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        num_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {num_classes}")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = ImprovedHandwritingCNN(num_classes).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully (device: {self.device})")
        
        # Define transform (should match training)
        self.transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            # Note: Use the same normalization as training
            # For now, using common values; ideally load from config
            transforms.Normalize(mean=[0.9], std=[0.2])
        ])
    
    def predict(self, image_path, top_k=5):
        """
        Predict the word in an image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = probabilities.topk(top_k, dim=1)
            
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            # Convert indices to words
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                word = self.label_encoder.inverse_transform([idx])[0]
                predictions.append((word, prob))
        
        return predictions
    
    def predict_batch(self, image_paths):
        """
        Predict words for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction lists
        """
        predictions = []
        for image_path in image_paths:
            preds = self.predict(image_path)
            predictions.append(preds)
        return predictions
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize the prediction with the image
        
        Args:
            image_path: Path to the image
            save_path: Optional path to save the visualization
        """
        # Get predictions
        predictions = self.predict(image_path, top_k=5)
        
        # Load image
        image = Image.open(image_path).convert('L')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Show image
        ax1.imshow(image, cmap='gray')
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Show predictions
        words = [pred[0] for pred in predictions]
        probs = [pred[1] * 100 for pred in predictions]
        
        y_pos = np.arange(len(words))
        ax2.barh(y_pos, probs, align='center', color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(words)
        ax2.invert_yaxis()
        ax2.set_xlabel('Probability (%)')
        ax2.set_title('Top-5 Predictions')
        ax2.set_xlim([0, 100])
        
        # Add probability values
        for i, (word, prob) in enumerate(zip(words, probs)):
            ax2.text(prob + 2, i, f'{prob:.1f}%', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


# Command Line Interface

def main():
    parser = argparse.ArgumentParser(
        description='Handwriting Recognition Inference'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file or directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='label_encoder.pkl',
        help='Path to label encoder'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the prediction'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save visualization'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = HandwritingRecognizer(args.model, args.encoder)
    
    # Check if path is file or directory
    path = Path(args.image_path)
    
    if path.is_file():
        # Single image
        print(f"\nProcessing: {path}")
        predictions = recognizer.predict(str(path), top_k=args.top_k)
        
        print(f"\nTop-{args.top_k} Predictions:")
        print("-" * 40)
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"{i}. {word:20s} {prob*100:5.2f}%")
        
        if args.visualize:
            recognizer.visualize_prediction(str(path), args.output)
    
    elif path.is_dir():
        # Directory of images
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        image_files = [
            f for f in path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"\nFound {len(image_files)} images in {path}")
        
        for img_file in image_files:
            print(f"\n{img_file.name}")
            print("-" * 40)
            predictions = recognizer.predict(str(img_file), top_k=3)
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"{i}. {word:15s} {prob*100:5.2f}%")
    
    else:
        print(f"Error: {path} is not a valid file or directory")


if __name__ == "__main__":
    main()