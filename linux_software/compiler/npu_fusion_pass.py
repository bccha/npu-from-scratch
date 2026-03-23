import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule
import torch.nn.functional as F
import copy

class NPU_Conv2d(nn.Module):
    """
    Custom PyTorch Module bridging the compilation graph with the NPU Emulator/Hardware.
    In Phase 2, this acts as the "Software Bit-exact Simulator" combining:
    Im2Col + Tiling -> 8x8 Hardware MAC -> Fused Bias Add -> Right Shift -> ReLU.
    In Phase 3, the `forward` function here will invoke the C++ Extension holding DMA commands.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, shift_val=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shift_val = shift_val
        
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # 1. Hardware Quantization Emulation (Float -> Int)
        # Input float [-1.0, 1.0] is scaled by 127.0 (Simulating C-driver INT8 cast)
        x_scaled = x * 127.0
        x_int = torch.clamp(torch.round(x_scaled), -128, 127).to(torch.int32)
        w_int = torch.clamp(torch.round(self.weight), -128, 127).to(torch.int32)
        b_int = torch.round(self.bias).to(torch.int32)
        
        batch_size, in_c, in_h, in_w = x_int.shape
        k = self.kernel_size
        out_h = (in_h + 2 * self.padding - k) // self.stride + 1
        out_w = (in_w + 2 * self.padding - k) // self.stride + 1
        
        # 2. Im2Col Unrolling (Simulating MSGDMA packing routine)
        x_unfold = F.unfold(x_int.float(), kernel_size=k, stride=self.stride, padding=self.padding).to(torch.int32)
        w_flat = w_int.view(self.out_channels, -1)
        
        # 3. MAC Emulation (8x8 Hardware limits emulated via matrix math)
        mac_result = torch.matmul(w_flat, x_unfold)
        
        # 4. Hardware DRAIN Mode & Post-Processing (Bias -> Shift -> ReLU)
        fused_result = mac_result + b_int.view(-1, 1)
        # Shift equivalent to integer division for simplicity (handling signed semantics)
        shifted_result = torch.bitwise_right_shift(fused_result, self.shift_val)
        relu_result = torch.clamp(shifted_result, 0, 127).to(torch.int8)
        
        # 5. Col2Im formatting back to standard feature map
        out = relu_result.view(batch_size, self.out_channels, out_h, out_w)
        return out.float()

    def export_hw_params(self, weight_path="weights.bin", bias_path="bias.bin"):
        """
        Packs the quantized 4D PyTorch Weights into 8x8 Contiguous blocks suitable 
        for NPU DMA transfers directly to the Systolic Array.
        """
        import numpy as np
        # Get quantized params
        w_int = torch.clamp(torch.round(self.weight), -128, 127).to(torch.int8).detach().numpy()
        b_int = torch.round(self.bias).to(torch.int32).detach().numpy()
        
        # Flatten input dims: [out_c, in_c * k * k] = [F, C*K*K] -> Transpose to [C*K*K, F]
        w_flat = w_int.reshape(self.out_channels, -1).T
        kernel_elements = w_flat.shape[0]
        filters = w_flat.shape[1]
        
        # We need to pad to multiples of 8 for both dimensions
        pad_rows = (8 - (kernel_elements % 8)) % 8
        pad_cols = (8 - (filters % 8)) % 8
        w_padded = np.pad(w_flat, ((0, pad_rows), (0, pad_cols)), mode='constant')
        
        # Format into 8x8 blocks
        num_blocks_row = w_padded.shape[0] // 8
        num_blocks_col = w_padded.shape[1] // 8
        w_npu_blocks = np.zeros((num_blocks_row, num_blocks_col, 8, 8), dtype=np.int8)
        
        for i in range(num_blocks_row):
            for j in range(num_blocks_col):
                block = w_padded[i*8:(i+1)*8, j*8:(j+1)*8]
                # Systolic array requires columns to be reversed due to staggered execution wiring
                w_npu_blocks[i, j] = block[:, ::-1]
                
        # Write to binaries
        with open(weight_path, "wb") as f:
            f.write(w_npu_blocks.tobytes())
            
        # Pad bias to multiple of 8 (hardware expects groups of 8)
        b_padded = np.pad(b_int, (0, pad_cols), mode='constant')
        with open(bias_path, "wb") as f:
            f.write(b_padded.tobytes())
            
        print(f"[NPU Compiler] Exported Parameters - Weights: {w_npu_blocks.shape}, Bias: {b_padded.shape}")



def fuse_conv_bn_relu_pass(fx_model: GraphModule) -> GraphModule:
    """
    PyTorch FX Custom Compiler Pass:
    Finds sequential [Conv2d -> BatchNorm2d -> ReLU] nodes in the computation graph
    and collapses them into a single Custom NPU_Conv2d node.
    """
    model = copy.deepcopy(fx_model)
    modules = dict(model.named_modules())
    
    # Track nodes to erase to avoid invalidating the iterator
    nodes_to_erase = []
    
    for node in model.graph.nodes:
        if node.op != 'call_module':
            continue
            
        # Dynamically fetch the module to support newly added nodes
        target_atoms = node.target.split('.')
        module = model
        for atom in target_atoms:
            if not hasattr(module, atom):
                module = None
                break
            module = getattr(module, atom)
            
        if module is None:
            continue
        
        # Match Target: Conv2d -> BatchNorm2d -> ReLU
        if isinstance(module, nn.Conv2d):
            next_node = node.next
            if next_node.op == 'call_module' and isinstance(modules[next_node.target], nn.BatchNorm2d):
                bn_node = next_node
                next_next_node = bn_node.next
                
                if next_next_node.op in ['call_module', 'call_function']:
                    relu_node = next_next_node
                    print(f"[*] Extracting Subgraph: {node.target} -> {bn_node.target} -> {relu_node.target}")
                    
                    conv_mod = modules[node.target]
                    bn_mod = modules[bn_node.target]
                    
                    # Mathematical BN Parameter Folding
                    scale = bn_mod.weight / torch.sqrt(bn_mod.running_var + bn_mod.eps)
                    fused_weight = conv_mod.weight * scale.view(-1, 1, 1, 1)
                    conv_bias = conv_mod.bias if conv_mod.bias is not None else torch.zeros_like(bn_mod.running_mean)
                    fused_bias = (conv_bias - bn_mod.running_mean) * scale + bn_mod.bias
                    
                    # --- INT8 Quantization Scaling (Post-Training Quantization) ---
                    w_max = torch.max(torch.abs(fused_weight)).clamp(min=1e-5)
                    w_scale = 127.0 / w_max
                    fused_weight_scaled = fused_weight * w_scale
                    
                    # The C-driver maps input X to int8 (-128~127) using factor 127.0.
                    # Total MAC sum = (X * 127.0) * (W * w_scale) = True_MAC * (127.0 * w_scale)
                    # Therefore, Bias must be scaled by (127.0 * w_scale) to be physically identical in the MAC.
                    bias_scale = 127.0 * w_scale
                    fused_bias_scaled = fused_bias * bias_scale
                    
                    # Create Custom NPU Hardware Node
                    npu_conv = NPU_Conv2d(
                        in_channels=conv_mod.in_channels,
                        out_channels=conv_mod.out_channels,
                        kernel_size=conv_mod.kernel_size if isinstance(conv_mod.kernel_size, int) else conv_mod.kernel_size[0],
                        stride=conv_mod.stride if isinstance(conv_mod.stride, int) else conv_mod.stride[0],
                        padding=conv_mod.padding if isinstance(conv_mod.padding, int) else conv_mod.padding[0]
                    )
                    
                    # Freeze quantized offline weights
                    npu_conv.weight = nn.Parameter(torch.clamp(torch.round(fused_weight_scaled), -128, 127))
                    npu_conv.bias = nn.Parameter(torch.round(fused_bias_scaled))
                    
                    # Register the new module into PyTorch network
                    npu_node_name = f"npu_{node.target.replace('.', '_')}"
                    model.add_module(npu_node_name, npu_conv)
                    
                    # Graph Rewiring
                    with model.graph.inserting_after(relu_node):
                        new_npu_node = model.graph.call_module(npu_node_name, node.args, node.kwargs)
                        relu_node.replace_all_uses_with(new_npu_node)
                    
                    nodes_to_erase.extend([relu_node, bn_node, node])
                    print(f"[+] Replaced with single Fusion Node: {npu_node_name}\n")

    # Clean up old graph elements
    for n in nodes_to_erase: # Erase end-to-start topological order (relu -> bn -> conv)
        model.graph.erase_node(n)

    model.graph.lint()
    model.recompile()
    return model

def emit_c_runtime(fx_model: GraphModule, c_filepath="npu_model_runtime.c"):
    """
    Bring Your Own Codegen (BYOC) / Emit C Phase:
    Translates the optimized PyTorch Graph directly into Bare-metal/Linux C Code
    that calls the DE10-Nano DMA API.
    """
    c_code = [
        "// ===========================================================================",
        "// AUTO-GENERATED BY NPU COMPILE PASS",
        "// PyTorch FX -> Embedded C Transpiler (Emit C)",
        "// ===========================================================================",
        "#include \"npu_api.h\"",
        "#include <stdio.h>",
        "#include <stdint.h>",
        "",
        "void run_npu_inference(int8_t* system_input_ptr, int8_t* system_output_ptr) {",
        "    printf(\"[NPU Runtime] Starting FPGA Hardware Execution...\\n\");",
        ""
    ]
    
    var_counter = 0
    buffer_map = {}
    
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder': # Model Input
            buffer_map[node.name] = "system_input_ptr"
            c_code.append(f"    // --- Model Input: {node.name} ---")
            
        elif node.op == 'call_module':
            module = dict(fx_model.named_modules())[node.target]
            if isinstance(module, NPU_Conv2d):
                in_buf = buffer_map.get(node.args[0].name, "unknown_in")
                out_buf = f"intermediate_buf_{var_counter}"
                buffer_map[node.name] = out_buf
                
                c_code.append(f"    // --- Hardware Matrix Fusion execution for {node.target} ---")
                c_code.append(f"    int8_t* {out_buf} = allocate_npu_buffer(MAX_LAYER_SIZE);")
                c_code.append(f"    npu_load_weights(\"{node.target}_weights.bin\");")
                c_code.append(f"    npu_load_bias(\"{node.target}_bias.bin\");")
                c_code.append(f"    npu_set_shift({module.shift_val});")
                c_code.append(f"    npu_execute_dma({in_buf}, {out_buf}, {module.out_channels}, {module.kernel_size});")
                c_code.append("")
                var_counter += 1
            elif isinstance(module, nn.Linear):
                in_buf = buffer_map.get(node.args[0].name, "unknown_in")
                out_buf = f"intermediate_buf_{var_counter}"
                buffer_map[node.name] = out_buf
                
                c_code.append(f"    // --- CPU Fallback execution for Fully Connected: {node.target} ---")
                c_code.append(f"    int8_t* {out_buf} = allocate_npu_buffer(MAX_LAYER_SIZE);")
                c_code.append(f"    cpu_execute_fc({in_buf}, \"{node.target}_weights.bin\", \"{node.target}_bias.bin\", {out_buf});")
                c_code.append("")
                var_counter += 1
            else:
                # Softmax, Flatten, etc. (No-ops in pure memory terms or CPU delegates)
                buffer_map[node.name] = buffer_map.get(node.args[0].name, "unknown_in")
                
        elif node.op == 'output':
            # End of graph
            in_buf = buffer_map.get(node.args[0].name, "unknown_in")
            c_code.append(f"    // --- Output Transfer ---")
            c_code.append(f"    copy_to_system_output({in_buf}, system_output_ptr);")
            c_code.append(f"    printf(\"[NPU Runtime] Graph Execution Completed.\\n\");")
            
    c_code.append("}")
    
    with open(c_filepath, "w") as f:
        f.write("\n".join(c_code) + "\n")
    print(f"[Emit C] Generated Hardware Runtime -> {c_filepath}")


if __name__ == "__main__":
    # --- Integration Unit Test --- 
    class DummyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8 * 4 * 4, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    print("=== Original PyTorch Model ===")
    model = DummyCNN()
    model.eval() # Freeze running stats
    
    # Initialize some mock test BN parameters so it creates viable weights
    model.bn1.running_mean.uniform_(-1, 1)
    model.bn1.running_var.uniform_(0.1, 1.5)
    model.bn1.weight.data.uniform_(0.5, 2.0)
    model.bn1.bias.data.uniform_(-1, 1)
    
    # Capture FX Compute Graph
    fx_graph = symbolic_trace(model)
    
    # Run the Custom NPU Fusion Compiler
    print("\n=== Running Custom NPU Fusion Pass ===")
    fused_model = fuse_conv_bn_relu_pass(fx_graph)
    
    print("\n=== Python Graph After NPU Fusion ===")
    print(fused_model.code)
    
    # Pass a Dummy Input Tensor (Random 4x4 image)
    dummy_in = torch.randint(0, 50, (1, 1, 4, 4)).float()
    print("\n[Input 4x4 Image]")
    print(dummy_in[0][0])
    
    # Execute Original network
    out_orig = fx_graph(dummy_in)
    # Execute NPU-Accelerated network
    out_fused = fused_model(dummy_in)
    
    print("\n[Original Graph FC Output]")
    print(out_orig)
    print("\n[NPU Fused Hardware FC Output]")
    print(out_fused)
    
    # Remember: Because NPU forces 8-bit int truncation during Requantization, 
    # out_orig and out_fused won't be mathematically equal unless original has exact same FakeQuant nodes.
    # But this proves the compiled graph wires the NPU operator natively!
    print("\n[SUCCESS] Fusion Graph executed smoothly!")
    
    print("\n=== Testing HW Parameter Export ===")
    for name, module in fused_model.named_modules():
        if isinstance(module, NPU_Conv2d):
            module.export_hw_params(weight_path=f"{name}_weights.bin", bias_path=f"{name}_bias.bin")
