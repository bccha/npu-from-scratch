import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

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
    for epoch in range(3):
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

def fuse_linear_bn(linear, bn):
    """
    Fuses a PyTorch Linear and BatchNorm1d into a single mathematically equivalent Linear Layer.
    W_fused = W * (gamma / sqrt(var + eps))
    b_fused = (b - mean) * (gamma / sqrt(var + eps)) + beta
    """
    w = linear.weight.data
    b = linear.bias.data if linear.bias is not None else torch.zeros_like(bn.running_mean)
    
    inv_std = 1.0 / torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight.data * inv_std
    
    # Broadcast scale across out_channels row vectors
    w_fused = w * scale.unsqueeze(1)
    b_fused = (b - bn.running_mean) * scale + bn.bias.data
    
    return w_fused.cpu().numpy().T, b_fused.cpu().numpy()

def export_to_npu(model, testset):
    print("\nFusing Batch Normalizations and Exporting Base Weights...")
    model.eval()
    
    # Extract PyTorch weights using BN FUSION!
    # Transposed automatically by our fuse_linear_bn function
    W1, b1 = fuse_linear_bn(model.fc1, model.bn1)
    W2, b2 = fuse_linear_bn(model.fc2, model.bn2)
    
    # 2. Extract Test Dataset
    X_test = testset.data.numpy().reshape(-1, 784)
    Y_test = testset.targets.numpy()
    
    X_q = (X_test / 255.0 * 127.0).astype(np.int8)
    
    # Quantize W1
    w1_max = np.max(np.abs(W1))
    w1_scale = 127.0 / w1_max
    W1_q = np.clip(np.round(W1 * w1_scale), -127, 127).astype(np.int8)
    
    # Quantize W2
    w2_max = np.max(np.abs(W2))
    w2_scale = 127.0 / w2_max
    W2_q = np.clip(np.round(W2 * w2_scale), -127, 127).astype(np.int8)
    
    # Calculate Scaling params mimicking the DRAIN logic
    target_scale_h = 64.0
    actual_scale_z1 = 127.0 * w1_scale
    shift1 = max(1, int(round(actual_scale_z1 / target_scale_h)))
    target_scale_h = actual_scale_z1 / shift1

    b1_q = np.round(b1 * actual_scale_z1).astype(np.int32)
    
    actual_scale_z2 = target_scale_h * w2_scale
    b2_q = np.round(b2 * actual_scale_z2).astype(np.int32)
    
    # Hardware Simulator test inside Python
    Z1_q_test = np.dot(X_q.astype(np.float64), W1_q.astype(np.float64))
    H_pre = (Z1_q_test + b1_q.astype(np.float64)) / float(shift1)
    H_q_test = np.clip(np.round(H_pre), 0, 127).astype(np.float64)
    Z2_q_test = np.dot(H_q_test, W2_q.astype(np.float64))
    Y_q_test = Z2_q_test + b2_q.astype(np.float64)
    pred_q = np.argmax(Y_q_test, axis=1)
    acc_q = np.mean(pred_q == Y_test)
    print(f"Quantized NPU-Equivalent Mathematical Accuracy: {acc_q * 100:.2f}%")

    def write_blob(filename, data):
        with open(filename, 'wb') as f:
            f.write(data.tobytes())
            
    # Format inputs into MSGDMA stream layout (8x8 blocks)
    batches = X_q.shape[0] // 8
    X_blocks = np.zeros((batches, 98, 8, 8), dtype=np.int8)
    for b in range(batches):
        X_batch = X_q[b*8:(b+1)*8]
        for i in range(98):
            X_blocks[b, i] = X_batch[:, i*8:(i+1)*8]
            
    # Output to the original mnist_test folder so legacy execution automatically absorbs it
    out_dir = "../"
    
    write_blob(os.path.join(out_dir, 'inputs.bin'), X_blocks)
    write_blob(os.path.join(out_dir, 'labels.bin'), Y_test[:batches*8].astype(np.int32))
    
    def format_weight_block(w_8x8):
        w_out = np.zeros((8, 8), dtype=np.int8)
        # Emulate MSGDMA Avalon-ST Big-Endian to Little-Endian mirror padding mapping
        for t in range(8):
            c = 7 - t
            w_out[t, :] = w_8x8[:, c]
        return w_out
        
    W1_npu = np.zeros((98, 8, 8, 8), dtype=np.int8)
    for i in range(98):
        for j in range(8):
            w_block = W1_q[i*8:(i+1)*8, j*8:(j+1)*8]
            W1_npu[i, j] = format_weight_block(w_block)
    
    write_blob(os.path.join(out_dir, 'weights_l1.bin'), W1_npu)
    write_blob(os.path.join(out_dir, 'bias_l1.bin'), b1_q)
    
    W2_padded = np.zeros((64, 16), dtype=np.int8)
    W2_padded[:, :10] = W2_q
    W2_npu = np.zeros((8, 2, 8, 8), dtype=np.int8)
    for i in range(8):
        for j in range(2):
            w_block = W2_padded[i*8:(i+1)*8, j*8:(j+1)*8]
            W2_npu[i, j] = format_weight_block(w_block)
            
    write_blob(os.path.join(out_dir, 'weights_l2.bin'), W2_npu)
    
    b2_padded = np.zeros(16, dtype=np.int32)
    b2_padded[:10] = b2_q
    write_blob(os.path.join(out_dir, 'bias_l2.bin'), b2_padded)
    print("PyTorch Compile Finished! Binaries injected backward into Linux Framework.")

if __name__ == '__main__':
    model, testset = train()
    export_to_npu(model, testset)
