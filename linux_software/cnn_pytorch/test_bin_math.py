import numpy as np
import struct
import os

def load_weights(file_path, shape):
    with open(file_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.int8).reshape(shape)

def load_bias(file_path, shape):
    with open(file_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.int32).reshape(shape)

def systolic_multiply(X, W):
    # X: (N, in_channels)  Wait, X in HW is 8 features wide. 
    # W: NPU format is (in_tiles, out_tiles, 8, 8) 
    # where W[i, j, t, :] is reversed column
    in_tiles, out_tiles, _, _ = W.shape
    out = np.zeros(out_tiles * 8, dtype=np.int32)
    for j in range(out_tiles):
        for t in range(in_tiles):
            # HW w block
            w_block = np.zeros((8, 8), dtype=np.int8)
            for row in range(8):
                w_block[:, 7 - row] = W[t, j, row, :]
            
            # X comes sequentially.
            x_slice = X[t*8 : t*8+8]
            
            # Explicitly cast to Int32 mathematically to prevent Numpy's native 8-bit dot product invisible local C-overflow!
            out[j*8 : j*8+8] += np.dot(x_slice.astype(np.int32), w_block.astype(np.int32))
            
    return out

def run_numpy():
    print("=== Numpy Int8 Simulation (100 Images) ===")
    
    with open("inputs.bin", "rb") as f:
        inputs_all = np.frombuffer(f.read(10000*28*28), dtype=np.int8).reshape(10000, 28, 28)
    
    with open("labels.bin", "rb") as f:
        labels_all = np.frombuffer(f.read(40000), dtype=np.int32)
        
    W1 = load_weights("weights_l1.bin", (2, 1, 8, 8))
    B1 = load_bias("bias_l1.bin", (8,))
    W2 = load_weights("weights_l2.bin", (169, 16, 8, 8))
    B2 = load_bias("bias_l2.bin", (128,))
    W3 = load_weights("weights_l3.bin", (16, 2, 8, 8))
    B3 = load_bias("bias_l3.bin", (16,))
    
    correct = 0
    for img_idx in range(100):
        inputs = inputs_all[img_idx]
        
        # Im2Col
        Y1 = np.zeros((8, 26, 26), dtype=np.int32)
        for py in range(26):
            for px in range(26):
                patch = inputs[py:py+3, px:px+3].flatten()
                X1 = np.zeros(16, dtype=np.int8)
                X1[:9] = patch
                out = systolic_multiply(X1, W1) + B1
                out = out >> 8
                out = np.clip(out, 0, 127)
                for c in range(8):
                    Y1[c, py, px] = out[c]
        
        H_flat = np.zeros((8, 13, 13), dtype=np.int8)
        for c in range(8):
            for py in range(13):
                for px in range(13):
                    H_flat[c, py, px] = np.max(Y1[c, py*2:py*2+2, px*2:px*2+2])
        
        H_flat_1d = H_flat.flatten()
        
        Y2 = systolic_multiply(H_flat_1d, W2) + B2
        Y2 = Y2 >> 8
        Y2 = np.clip(Y2, 0, 127)
        
        Y3_full = systolic_multiply(Y2, W3) + B3
        
        if img_idx == 0:
            x_flat = X1.flatten()
            x_range = [np.min(x_flat), np.max(x_flat)]
            x_nz = np.count_nonzero(x_flat)
            x_first = x_flat[x_flat != 0][:10]
            print("--- Image 0 Telemetry ---")
            print(f"Im2Col X_hw Range: {x_range}, Non-Zeros: {x_nz} / {len(x_flat)}")
            print(f"      First 10 non-zero: {x_first}")
            
            print(f"Conv1 Max: {np.max(Y1)}, Mod: {Y1.flatten()[Y1.flatten() != 0][:10]}")
            print(f"Pool1 Max: {np.max(H_flat_1d)}, Mod: {H_flat_1d[H_flat_1d != 0][:10]}")
            print(f"FC1 Max: {np.max(Y2)}, Mod: {Y2[Y2 != 0][:10]}")
            print(f"FC2 Logits: {Y3_full[:10]}")
            print("-------------------------")
            
        pred = np.argmax(Y3_full[:10])
        
        if pred == labels_all[img_idx]:
            correct += 1
            
    print(f"Accuracy over 100 images: {correct}%")

if __name__ == "__main__":
    run_numpy()
