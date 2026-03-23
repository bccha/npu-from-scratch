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
    print("\n[PyTorch Auto-Compiler] Parsing FX Graph for Hardware Emitting...")
    model.eval()
    
    from torch.fx import symbolic_trace
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../compiler'))
    from npu_compiler import NPUCompiler
    
    fx_graph = symbolic_trace(model)
    modules = dict(model.named_modules())
    
    compiler = NPUCompiler(out_c_file="../npu_auto_runtime.c")
    
    layer_idx = 1
    
    # 1. Prepare Base Input Vectors global formats
    X_test = testset.data.numpy().reshape(-1, 784)
    Y_test = testset.targets.numpy()
    X_q = (X_test / 255.0 * 127.0).astype(np.int8)
    
    out_dir = "../"
    def write_blob(filename, data):
        with open(filename, 'wb') as f:
            f.write(data.tobytes())
            
    batches = X_q.shape[0] // 8
    X_blocks = np.zeros((batches, 98, 8, 8), dtype=np.int8)
    for b in range(batches):
        X_batch = X_q[b*8:(b+1)*8]
        for i in range(98):
            X_blocks[b, i] = X_batch[:, i*8:(i+1)*8]
            
    write_blob(os.path.join(out_dir, 'inputs.bin'), X_blocks)
    write_blob(os.path.join(out_dir, 'labels.bin'), Y_test[:batches*8].astype(np.int32))
    
    target_scale_h = 64.0
    shift1 = None 
    
    # 2. Dynamic AST Traversal
    for node in fx_graph.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], nn.Linear):
            lin_mod = modules[node.target]
            in_feat = lin_mod.in_features
            out_feat = lin_mod.out_features
            
            bn_mod = None
            has_relu = False
            
            curr = node.next
            while curr and curr.op == 'call_module':
                mod = modules.get(curr.target, None)
                if mod is None:
                    curr = curr.next
                    continue
                if isinstance(mod, nn.Linear):
                    break # Stop looking when we hit the next operator hierarchy
                if isinstance(mod, nn.BatchNorm1d): 
                    if bn_mod is None: bn_mod = mod # only fetch immediate norm
                if isinstance(mod, nn.ReLU): 
                    has_relu = True
                curr = curr.next
                
            if bn_mod:
                W, b = fuse_linear_bn(lin_mod, bn_mod)
            else:
                W = lin_mod.weight.data.cpu().numpy().T
                b = lin_mod.bias.data.cpu().numpy()
                
            w_max = np.max(np.abs(W))
            w_scale = 127.0 / w_max
            W_q = np.clip(np.round(W * w_scale), -127, 127).astype(np.int8)
            
            if layer_idx == 1:
                actual_scale_z = 127.0 * w_scale
                shift1 = max(1, int(round(actual_scale_z / target_scale_h)))
                target_scale_h = actual_scale_z / shift1
                b_q = np.round(b * actual_scale_z).astype(np.int32)
            else:
                actual_scale_z = target_scale_h * w_scale
                b_q = np.round(b * actual_scale_z).astype(np.int32)
            
            in_tiles = in_feat // 8 + (1 if in_feat % 8 else 0)
            out_tiles = out_feat // 8 + (1 if out_feat % 8 else 0)
            
            W_padded = np.zeros((in_tiles * 8, out_tiles * 8), dtype=np.int8)
            W_padded[:in_feat, :out_feat] = W_q
            
            W_npu = np.zeros((in_tiles, out_tiles, 8, 8), dtype=np.int8)
            for i in range(in_tiles):
                for j in range(out_tiles):
                    w_block = W_padded[i*8:(i+1)*8, j*8:(j+1)*8]
                    w_out = np.zeros((8, 8), dtype=np.int8)
                    for t in range(8):
                        w_out[t, :] = w_block[:, 7 - t]
                    W_npu[i, j] = w_out
                    
            b_padded = np.zeros(out_tiles * 8, dtype=np.int32)
            b_padded[:out_feat] = b_q
            
            w_file = f'weights_l{layer_idx}.bin'
            b_file = f'bias_l{layer_idx}.bin'
            write_blob(os.path.join(out_dir, w_file), W_npu)
            write_blob(os.path.join(out_dir, b_file), b_padded)
            
            compiler.register_linear_fused_layer(node.target, in_feat, out_feat, has_relu)
            print(f"  [+] Discovered Layer: {node.target} ({in_feat}->{out_feat}), ReLU={has_relu}")
            
            layer_idx += 1
            
    print(f"\n[PyTorch Auto-Compiler] Graph fully parsed! Emitting C Script ({len(compiler.layer_meta)} Layers)...")
    compiler.generate_c_code()

if __name__ == '__main__':
    model, testset = train()
    export_to_npu(model, testset)
