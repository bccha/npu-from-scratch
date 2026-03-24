import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../compiler')))
from cyclone_npu_sdk import export_model_to_fpga


class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.bn2 = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST_MLP().to(device)
    
    # Minimal PyTorch Transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Genuine PyTorch MNIST MLP Native Model...")
    for epoch in range(5):
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1} Accuracy: {100 * correct / total:.2f}%")
        
    return model, testset

    return model, testset

if __name__ == '__main__':
    model, testset = train()
    
    # MLP Hardware Logic exceptionally requires explicit pre-tiled 8x8 Block structures!
    X_test = testset.data.numpy().reshape(-1, 784)
    Y_test = testset.targets.numpy()
    X_q = (X_test / 255.0 * 127.0).astype(np.int8)
    
    batches = X_q.shape[0] // 8
    X_blocks = np.zeros((batches, 98, 8, 8), dtype=np.int8)
    for b in range(batches):
        X_batch = X_q[b*8:(b+1)*8]
        for i in range(98):
            X_blocks[b, i] = X_batch[:, i*8:(i+1)*8]
            
    with open("./inputs.bin", "wb") as f:
        f.write(X_blocks.tobytes())
    with open("./labels.bin", "wb") as f:
        f.write(Y_test[:batches*8].astype(np.int32).tobytes())
        
    print(f"  [+] Exported Pre-Tiled 8x8 Blocks for MLP Execution.")
    
    # Pass testset=None to SDK to prevent the standard flat 28x28 override.
    export_model_to_fpga(model, testset=None, out_dir="./", out_c_file="./npu_auto_runtime.c")
