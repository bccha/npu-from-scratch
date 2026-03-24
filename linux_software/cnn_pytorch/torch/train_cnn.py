import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../compiler')))
from cyclone_npu_sdk import export_model_to_fpga


class NPU_CNN(nn.Module):
    def __init__(self):
        super(NPU_CNN, self).__init__()
        # Conv1: 1 input channel (MNIST), 8 output channels, 3x3 kernel
        # Image drops from 28x28 -> 26x26
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(8)
        
        # MaxPool1: 2x2. Image drops from 26x26 -> 13x13
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # FC HEAD: 8 channels * 13 * 13 = 1352 input features
        self.fc1 = nn.Linear(8 * 13 * 13, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Output Layer: 128 -> 10 classes
        self.fc2 = nn.Linear(128, 10)
        # Note: No BN on output layer as it holds logits

    def forward(self, x):
        # Layer 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Flatten for Dense Layers
        x = torch.flatten(x, 1)
        
        # Layer 2: Linear -> BN -> ReLU
        x = self.fc1(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Layer 3: Linear Logits
        x = self.fc2(x)
        return x

def train_and_export():
    device = torch.device("cpu") # For deterministic HW mapping
    model = NPU_CNN().to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Train for 5 epochs to get high-accuracy weights
    print("Training Genuine PyTorch MNIST CNN Model...")
    for epoch in range(5):
        model.train()
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1} Accuracy: {100 * correct / total:.2f}%")
        
    model.eval()
    print("\nTraining Complete. (Exporting Logic to follow...)")
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # [Architecture Note] Memory vs. Compute Trade-off for CNN im2col:
    # We deliberately DO NOT perform `im2col` (sliding window geometry) here in Python.
    # If we statically expanded a 28x28 image (784 bytes) through a 3x3 kernel, 
    # the 676 overlapping patches would bloat a single image to >10KB. 
    # For a 10,000 image test set, `inputs.bin` would explode from 7.8MB to >100MB.
    # To prevent this extreme memory/communication bottleneck on the Edge Device, 
    # we export the raw flat 28x28 image here and force the FPGA's local ARM CPU to 
    # dynamically slide and crop the image inside its ultra-fast L2 cache at runtime.
    export_model_to_fpga(model, test_dataset, out_dir="./", out_c_file="./npu_auto_runtime.c")

if __name__ == '__main__':
    train_and_export()
