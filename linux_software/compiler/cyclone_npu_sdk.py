import torch
import torch.nn as nn
import numpy as np
import os
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from npu_compiler import NPUCompiler

def fuse_conv_bn_2d(conv, bn):
    w = conv.weight.data # shape: (out_c, in_c, kH, kW)
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    
    inv_std = 1.0 / torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight.data * inv_std
    
    # Broadcast scale across out_channels
    scale = scale.view(-1, 1, 1, 1)
    w_fused = w * scale
    b_fused = (b - bn.running_mean) * bn.weight.data * inv_std + bn.bias.data
    
    # Spatial Unrolling Flatten to 2D
    w_fused_2d = w_fused.view(w_fused.size(0), -1).cpu().numpy().T
    b_fused_1d = b_fused.cpu().numpy()
    return w_fused_2d, b_fused_1d

def fuse_linear_bn(linear, bn):
    w = linear.weight.data
    b = linear.bias.data if linear.bias is not None else torch.zeros_like(bn.running_mean)
    inv_std = 1.0 / torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight.data * inv_std
    w_fused = w * scale.unsqueeze(1)
    b_fused = (b - bn.running_mean) * scale + bn.bias.data
    return w_fused.cpu().numpy().T, b_fused.cpu().numpy()

def export_model_to_fpga(model, testset=None, out_dir="./", out_c_file="./npu_auto_runtime.c"):
    print("\n[Cyclone V NPU SDK] Parsing PyTorch FX Graph for Hardware Emitting...")
    model.eval()
    
    # Trace Graph
    fx_graph = symbolic_trace(model)
    
    # Shape Propagate using dummy input to capture all spatial dimensions!
    dummy_input = torch.zeros(1, 1, 28, 28).to(next(model.parameters()).device)
    ShapeProp(fx_graph).propagate(dummy_input)
    
    modules = dict(model.named_modules())
    compiler = NPUCompiler(out_c_file=out_c_file)
    
    # --- PTQ Legacy Backend ---
    
    target_scale_h = 64.0
    layer_idx = 1
    
    # Graph Parser Loop Framework
    for node in fx_graph.graph.nodes:
        if node.op == 'call_module':
            mod = modules[node.target]
            tensor_meta = node.meta.get('tensor_meta')
            if tensor_meta is None: continue
            
            output_shape = tensor_meta.shape # Contains [1, Channels, Height, Width]
            input_node = node.args[0]
            input_shape = input_node.meta['tensor_meta'].shape if 'tensor_meta' in input_node.meta else output_shape
            
            if isinstance(mod, nn.Conv2d):
                print(f"  [+] Discovered Conv2D Layer: {node.target} - In: {input_shape} Out: {output_shape}")
                
                kernel_size = mod.kernel_size[0]
                stride = mod.stride[0]
                padding = mod.padding[0]
                
                has_relu = False
                bn_mod = None
                pool_mod = None
                
                curr = node.next
                while curr and curr.op != 'output':
                    if curr.op == 'call_function' and 'relu' in str(curr.target): has_relu = True
                    if curr.op == 'call_module':
                        submod = modules.get(curr.target, None)
                        if submod is None:
                            curr = curr.next
                            continue
                        if isinstance(submod, (nn.Linear, nn.Conv2d, nn.Flatten)): break
                        if isinstance(submod, nn.BatchNorm2d): bn_mod = submod
                        if isinstance(submod, nn.ReLU): has_relu = True
                    if curr.op == 'call_module' and isinstance(modules.get(curr.target), nn.MaxPool2d): pool_mod = modules.get(curr.target)
                    curr = curr.next
                
                if bn_mod:
                    W, b = fuse_conv_bn_2d(mod, bn_mod)
                else:
                    W = mod.weight.data.view(mod.out_channels, -1).cpu().numpy().T
                    b = mod.bias.data.cpu().numpy() if mod.bias is not None else np.zeros(mod.out_channels)
                    
                w_max = np.max(np.abs(W))
                w_scale = 127.0 / w_max if w_max != 0 else 1.0
                
                # [SW FIX] Prevent 127 Int8 Hardware Saturation limit against the constant `256.0` RTL Divisor.
                # Target safe scalar mapping of ~64.0 (allows native PyTorch float activations up to 2.0 without clipping!)
                max_allowed_w_scale = 16384.0 / (127.0 if layer_idx == 1 else target_scale_h)
                w_scale = min(w_scale, max_allowed_w_scale)
                
                W_q = np.clip(np.round(W * w_scale), -127, 127).astype(np.int8)
                
                if layer_idx == 1:
                    actual_scale_z = 127.0 * w_scale
                else:
                    actual_scale_z = target_scale_h * w_scale
                
                target_scale_h = actual_scale_z / 256.0
                b_q = np.round(b * actual_scale_z).astype(np.int32)
                    
                out_feat = mod.out_channels
                in_feat = mod.in_channels * kernel_size * kernel_size
                
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
                
                # Write to disk
                w_file = f'weights_l{layer_idx}.bin'
                b_file = f'bias_l{layer_idx}.bin'
                with open(os.path.join(out_dir, w_file), 'wb') as f:
                    f.write(W_npu.tobytes())
                with open(os.path.join(out_dir, b_file), 'wb') as f:
                    f.write(b_padded.tobytes())
                    
                compiler.register_conv_fused_layer(
                    name=node.target,
                    in_channels=mod.in_channels,
                    out_channels=mod.out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    in_h=input_shape[2],
                    in_w=input_shape[3],
                    has_relu=has_relu,
                    pool_size=(2 if node.target == 'conv1' else (pool_mod.kernel_size if pool_mod else 1)),
                    shift_val=8
                )
                
                layer_idx += 1
            
            if isinstance(mod, nn.Linear):
                print(f"  [+] Discovered Linear Layer: {node.target} - Shape: {output_shape}")
                
                has_relu = False
                bn_mod = None
                
                curr = node.next
                while curr and curr.op != 'output':
                    if curr.op == 'call_function' and 'relu' in str(curr.target): has_relu = True
                    if curr.op == 'call_module':
                        submod = modules.get(curr.target, None)
                        if submod is None:
                            curr = curr.next
                            continue
                        if isinstance(submod, (nn.Linear, nn.Conv2d)): break
                        if isinstance(submod, nn.BatchNorm1d): bn_mod = submod
                        if isinstance(submod, nn.ReLU): has_relu = True
                    curr = curr.next
                    
                if bn_mod:
                    W, b = fuse_linear_bn(mod, bn_mod)
                else:
                    W = mod.weight.data.cpu().numpy().T
                    b = mod.bias.data.cpu().numpy() if mod.bias is not None else np.zeros(mod.out_features)
                    
                w_max = np.max(np.abs(W))
                w_scale = 127.0 / w_max if w_max != 0 else 1.0
                
                # [SW FIX] Throttle integer mapping multiplier explicitly avoiding hardcoded divisor boundaries.
                max_allowed_w_scale = 16384.0 / (127.0 if layer_idx == 1 else target_scale_h)
                w_scale = min(w_scale, max_allowed_w_scale)
                
                W_q = np.clip(np.round(W * w_scale), -127, 127).astype(np.int8)
                
                if layer_idx == 1:
                    actual_scale_z = 127.0 * w_scale
                else:
                    actual_scale_z = target_scale_h * w_scale
                
                target_scale_h = actual_scale_z / 256.0
                b_q = np.round(b * actual_scale_z).astype(np.int32)
                    
                in_feat = mod.in_features
                out_feat = mod.out_features
                
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
                with open(os.path.join(out_dir, w_file), 'wb') as f:
                    f.write(W_npu.tobytes())
                with open(os.path.join(out_dir, b_file), 'wb') as f:
                    f.write(b_padded.tobytes())
                    
                compiler.register_linear_fused_layer(node.target, in_feat, out_feat, has_relu, shift_val=8)
                layer_idx += 1

    # Dump the Base Images (Shape: [10000, 28, 28]) directly for the Target Hardware Im2Col routines
    if testset is not None:
        X_test = testset.data.numpy()
        Y_test = testset.targets.numpy()
        X_q = (X_test / 255.0 * 127.0).astype(np.int8)
        
        with open(os.path.join(out_dir, "inputs.bin"), "wb") as f:
            f.write(X_q.tobytes())
        with open(os.path.join(out_dir, "labels.bin"), "wb") as f:
            f.write(Y_test.astype(np.int32).tobytes())
        print(f"  [+] Exported {X_q.shape[0]} Validation Images and Labels dynamically.")

    print(f"\n[Cyclone V NPU SDK] Graph fully parsed! Emitting C Script ({len(compiler.layers)} Layers)...")
    compiler.generate_c_code()
