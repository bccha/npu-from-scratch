import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import struct
import sys
import os

# Import the fusion compiler from the compiler directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../compiler'))
from npu_fusion_pass import fuse_conv_bn_relu_pass, NPU_Conv2d, emit_c_runtime

def load_mnist(image_file, label_file):
    with open(image_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
    with open(label_file, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return images, labels

class MNIST_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv Layer 1: 1 input channel, 8 output channels, 3x3 kernel, stride 2
        # Input: 28x28 -> Output: 13x13 (padding=0)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        # FC Layer: 8 * 13 * 13 = 1352
        self.fc = nn.Linear(8 * 13 * 13, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def format_weight_block_torch(w_8x8):
    w_out = np.zeros((8, 8), dtype=np.int8)
    for t in range(8):
        c = 7 - t  # Reverse columns for the Systolic Array Hardware
        w_out[t, :] = w_8x8[:, c]
    return w_out

def export_fc_binaries(fc_weight, fc_bias):
    # PyTorch Linear weight is (out_features, in_features). 
    # Our hardware/C-driver expects (in_features, out_features) = (1352, 10)
    W_fc = fc_weight.detach().numpy().T
    b_fc = fc_bias.detach().numpy()
    
    # PTQ simple symmetric quantization
    w_fc_max = max(np.max(np.abs(W_fc)), 1e-5)
    w_fc_scale = 127.0 / w_fc_max
    W_fc_q = np.clip(np.round(W_fc * w_fc_scale), -127, 127).astype(np.int8)
    
    # Because NPU_Conv2d shifts >> 8, its output is in the 0~127 range naturally.
    # Therefore, FC input scale is approx 1.0 (relative to int8 standard bounds).
    # We apply the weight scale to the bias to match the scaled dot product.
    # To prevent bias truncation, we scale it by a heuristic factor (127.0) to match Activation*Weight expansion.
    b_fc_q = np.round(b_fc * w_fc_scale * 127.0).astype(np.int32)
    
    # Pad FC from (1352, 10) to (1352, 16)
    W_fc_padded = np.zeros((1352, 16), dtype=np.int8)
    W_fc_padded[:, :10] = W_fc_q
    
    # Reshape FC into NPU blocks (169 rows of 8, 2 cols of 8 -> 169x2 blocks of 8x8)
    W_fc_npu = np.zeros((169, 2, 8, 8), dtype=np.int8)
    for i in range(169):
        for j in range(2):
            w_block = W_fc_padded[i*8:(i+1)*8, j*8:(j+1)*8]
            W_fc_npu[i, j] = format_weight_block_torch(w_block)
            
    with open("weights_l2.bin", "wb") as f:
        f.write(W_fc_npu.tobytes())
        
    b_fc_padded = np.zeros(16, dtype=np.int32)
    b_fc_padded[:10] = b_fc_q
    with open("bias_l2.bin", "wb") as f:
        f.write(b_fc_padded.tobytes())
        
    print(f"Exported FC weights (padded to 1352x16): {W_fc_npu.shape}")

def main():
    print("=== Loading MNIST Dataset ===")
    try:
        X_train, Y_train = load_mnist('../mnist_test/train-images-idx3-ubyte', '../mnist_test/train-labels-idx1-ubyte')
        X_test, Y_test = load_mnist('../mnist_test/t10k-images-idx3-ubyte', '../mnist_test/t10k-labels-idx1-ubyte')
    except Exception as e:
        print(f"Error loading MNIST dataset. Ensure run from linux_software/cnn_test directory: {e}")
        sys.exit(1)
    
    # Normalize PyTorch style: -1 to 1 mapping
    X_train = (X_train.astype(np.float32) - 128.0) / 128.0
    X_test = (X_test.astype(np.float32) - 128.0) / 128.0
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = MNIST_PyTorch()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("=== Training PyTorch Model (1 Conv + BN + 1 FC) ===")
    model.train()
    epochs = 3
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Test accuracy
        model.eval()
        with torch.no_grad():
            test_outputs = model(torch.tensor(X_test))
            preds = torch.argmax(test_outputs, dim=1)
            acc = (preds == torch.tensor(Y_test)).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {acc:.4f}")
        model.train()
        
    print("\n=== Running Custom NPU FX Compiler Fusion ===")
    model.eval()
    from torch.fx import symbolic_trace
    fx_graph = symbolic_trace(model)
    fused_model = fuse_conv_bn_relu_pass(fx_graph)
    
    print("\n=== Exporting HW Parameters ===")
    # Export L1 (NPU_Conv2d)
    for name, module in fused_model.named_modules():
        if isinstance(module, NPU_Conv2d):
            module.export_hw_params("weights_l1.bin", "bias_l1.bin")
            print(f"Exported L1 NPU parameters for '{name}'")
            
    # Export L2 (FC is still native PyTorch Linear)
    export_fc_binaries(model.fc.weight, model.fc.bias)
    
    print("\n=== Generating Built-in C Runtime (Emit C) ===")
    emit_c_runtime(fused_model, "npu_model_runtime.c")
    
    print("\n=== Exporting Test Inputs and Headers ===")
    X_q = (X_test * 127.0).astype(np.int8)
    with open('inputs.bin', 'wb') as f:
        f.write(X_q[:1000].tobytes())
    with open('labels.bin', 'wb') as f:
        f.write(Y_test[:1000].astype(np.int32).tobytes())
    print("Exported 1000 tests to inputs.bin and labels.bin")
        
    with open('model_params.h', 'w') as f:
        f.write("#pragma once\n")
        f.write("#define SHIFT1 8\n") # NPU Hardcoded Shift emulation
        f.write("#define SHIFT2 8\n") # Fallback shift for CPU FC execution
    print("Exported model_params.h")
        
    print("\n[SUCCESS] PyTorch training and NPU compilation finished.")

if __name__ == "__main__":
    main()
