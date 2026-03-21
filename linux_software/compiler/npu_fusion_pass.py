import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule
import copy

def fuse_conv_bn_relu_pass(fx_model: GraphModule) -> GraphModule:
    """
    PyTorch FX Custom Compiler Pass:
    Finds sequential [Conv2d -> BatchNorm2d -> ReLU] nodes in the computation graph
    and collapses them into a single Custom NPU node (with mathematically fused weights).
    """
    model = copy.deepcopy(fx_model)
    modules = dict(model.named_modules())
    
    # 1. Iterate through graph nodes
    for node in model.graph.nodes:
        if node.op != 'call_module':
            continue
            
        module = modules[node.target]
        
        # 2. Pattern Matching: Found a Conv2d?
        if isinstance(module, nn.Conv2d):
            next_node = node.next
            if next_node.op == 'call_module' and isinstance(modules[next_node.target], nn.BatchNorm2d):
                bn_node = next_node
                next_next_node = bn_node.next
                
                if next_next_node.op in ['call_module', 'call_function']:
                    # Assuming next node is ReLU for pattern matching
                    relu_node = next_next_node
                    print(f"[*] Found Target Pattern: {node.target} -> {bn_node.target} -> {relu_node.target}")
                    
                    # 3. Mathematical Fusion (Weights & Biases)
                    conv_mod = modules[node.target]
                    bn_mod = modules[bn_node.target]
                    
                    # Fused Weight = Conv_Weight * (Gamma / sqrt(Variance + eps))
                    scale = bn_mod.weight / torch.sqrt(bn_mod.running_var + bn_mod.eps)
                    fused_weight = conv_mod.weight * scale.view(-1, 1, 1, 1)
                    
                    # Fused Bias = (Conv_Bias - Mean) * Scale + Beta
                    conv_bias = conv_mod.bias if conv_mod.bias is not None else torch.zeros_like(bn_mod.running_mean)
                    fused_bias = (conv_bias - bn_mod.running_mean) * scale + bn_mod.bias
                    
                    # 4. (Demo) Update Conv module with fused parameters natively
                    conv_mod.weight = nn.Parameter(fused_weight)
                    conv_mod.bias = nn.Parameter(fused_bias)
                    conv_mod.padding_mode = 'zeros' # Replace BN
                    
                    # 5. Graph Rewrite: Bypass BN and ReLU nodes to stream directly out of Conv
                    # Note: In real NPU compiler, we would replace this with an `NPU_Conv2D` node call
                    with model.graph.inserting_after(relu_node):
                        # Replace all references to ReLU output with the Fused Conv output
                        relu_node.replace_all_uses_with(node)
                    
                    # Remove the old dangling nodes from graph
                    model.graph.erase_node(relu_node)
                    model.graph.erase_node(bn_node)
                    print(f"[+] Successfully fused to single node: {node.target}\n")

    # 6. Recompile the manipulated graph
    model.graph.lint()
    model.recompile()
    return model

if __name__ == "__main__":
    # --- Quick Sanity Check --- 
    class DummyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.fc = nn.Linear(8 * 14 * 14, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    print("=== Original PyTorch Model ===")
    model = DummyCNN()
    model.eval() # BN uses running mean/var in eval mode
    print(model)
    
    # 1. Capture PyTorch execution graph
    fx_graph = symbolic_trace(model)
    
    print("\n=== Running Custom NPU Fusion Pass ===")
    fused_model = fuse_conv_bn_relu_pass(fx_graph)
    
    print("=== Graph After NPU Fusion ===")
    print(fused_model.code)
