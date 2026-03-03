import numpy as np
import urllib.request
import gzip
import os
import struct

def fetch_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    dataset = {}
    for f in files:
        if not os.path.exists(f):
            print(f"Downloading {f}...")
            urllib.request.urlretrieve(base_url + f, f)
            print(f"Downloaded {f}")
        
        with gzip.open(f, 'rb') as gz:
            if 'images' in f:
                magic, num, rows, cols = struct.unpack(">IIII", gz.read(16))
                data = np.frombuffer(gz.read(), dtype=np.uint8).reshape(num, rows * cols)
            else:
                magic, num = struct.unpack(">II", gz.read(8))
                data = np.frombuffer(gz.read(), dtype=np.uint8)
            
            key = "train_" if "train" in f else "test_"
            key += "x" if "images" in f else "y"
            dataset[key] = data
            
    return dataset

def relu(x):
    return np.maximum(0, x)
    
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def to_categorical(y, num_classes=10):
    return np.eye(num_classes)[y]

def train_mlp(X, Y, epochs=15, lr=0.1, batch_size=128):
    np.random.seed(42)
    # 784 -> 64 -> 10
    W1 = np.random.randn(784, 64) * np.sqrt(2. / 784)
    b1 = np.zeros(64)
    W2 = np.random.randn(64, 10) * np.sqrt(2. / 64)
    b2 = np.zeros(10)
    
    Y_cat = to_categorical(Y)
    
    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_cat[indices]
        
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            # Forward
            Z1 = X_batch @ W1 + b1
            A1 = relu(Z1)
            Z2 = A1 @ W2 + b2
            A2 = softmax(Z2)
            
            # Backward
            dZ2 = A2 - Y_batch
            dW2 = A1.T @ dZ2 / batch_size
            db2 = np.sum(dZ2, axis=0) / batch_size
            
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * (Z1 > 0)
            dW1 = X_batch.T @ dZ1 / batch_size
            db1 = np.sum(dZ1, axis=0) / batch_size
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            
        # Eval
        pred = np.argmax(relu(X @ W1 + b1) @ W2 + b2, axis=1)
        acc = np.mean(pred == Y)
        print(f"Epoch {epoch+1}/{epochs} | Acc: {acc:.4f}")
        
    return W1, b1, W2, b2

def export_binaries(X_test, Y_test, W1, b1, W2, b2):
    # Quantize to 8-bit Int
    # Best practice is to train quantization aware, but post-training we will use fixed shift factors
    
    # 1. Inputs: 0.0 ~ 1.0 -> 0 ~ 127
    X_q = (X_test / 255.0 * 127.0).astype(np.int8)
    
    # 2. Weights: Max Abs scaling -> -127 ~ 127
    w1_max = np.max(np.abs(W1))
    w1_scale = 127.0 / w1_max
    W1_q = np.clip(np.round(W1 * w1_scale), -127, 127).astype(np.int8)
    
    w2_max = np.max(np.abs(W2))
    w2_scale = 127.0 / w2_max
    W2_q = np.clip(np.round(W2 * w2_scale), -127, 127).astype(np.int8)
    
    # 3. Layer 1 Scaling
    # Float eq: H = ReLU(X * W1 + b1)
    # Int eq:   H_q = ReLU( (X_q * W1_q + b1_q) >> shift1 )
    # Let's set the target scale for H_q to be something less aggressive than 127.0
    # because if H_q has scale 127.0, a float value of 2.0 becomes 254 and clips to 127.
    # A standard choice is target_scale_h = 32.0 or 64.0 to allow some headroom for ReLU outputs.
    # Let's use target_scale_h = 64.0 to allow float activations up to ~2.0 without clipping.
    target_scale_h = 64.0
    
    actual_scale_z1 = 127.0 * w1_scale
    shift1 = max(1, int(round(actual_scale_z1 / target_scale_h)))
    
    # Recalculate true target_scale based on integer shift1
    target_scale_h = actual_scale_z1 / shift1

    # Bias 1 must match the pre-shift scale (X_q * W1_q)
    b1_q = np.round(b1 * actual_scale_z1).astype(np.int32)
    
    print(f"L1 -> w1_scale: {w1_scale:.2f}, shift1: {shift1} (target H scale: {target_scale_h:.2f})")
    
    # 4. Layer 2 Scaling
    # Float eq: Y = H * W2 + b2
    # Int eq:   Y_q = H_q * W2_q + b2_q
    #           (H * target_scale_h) * (W2 * w2_scale) = H * W2 * target_scale_h * w2_scale
    
    actual_scale_z2 = target_scale_h * w2_scale
    b2_q = np.round(b2 * actual_scale_z2).astype(np.int32)
    shift2 = 1 

    print(f"L2 -> w2_scale: {w2_scale:.2f}, final scale Y: {actual_scale_z2:.2f}")
    
    # 5. Verify Quantization Accuracy in Python
    # Simplify python eval to mimic hardware exactly. Note // shift1 can cause issues in numpy.
    Z1_q_test = np.dot(X_q.astype(np.float64), W1_q.astype(np.float64))
    
    H_pre = (Z1_q_test + b1_q.astype(np.float64)) / float(shift1)
    H_q_test = np.clip(np.round(H_pre), 0, 127).astype(np.float64)
    
    Z2_q_test = np.dot(H_q_test, W2_q.astype(np.float64))
    Y_q_test = Z2_q_test + b2_q.astype(np.float64)
    
    pred_q = np.argmax(Y_q_test, axis=1)
    acc_q = np.mean(pred_q == Y_test)
    print(f"Quantized Integer Model Acc: {acc_q:.4f}")


    def write_blob(filename, data):
        with open(filename, 'wb') as f:
            f.write(data.tobytes())
            
    # Process X_q: format into 8x8 blocks.
    batches = X_q.shape[0] // 8
    X_blocks = np.zeros((batches, 98, 8, 8), dtype=np.int8)
    for b in range(batches):
        X_batch = X_q[b*8:(b+1)*8]
        for i in range(98):
            X_blocks[b, i] = X_batch[:, i*8:(i+1)*8]
            
    write_blob('inputs.bin', X_blocks)
    write_blob('labels.bin', Y_test[:batches*8].astype(np.int32))
    
    def format_weight_block(w_8x8):
        w_out = np.zeros((8, 8), dtype=np.int8)
        for t in range(8):
            c = 7 - t
            w_out[t, :] = w_8x8[:, c]
        return w_out
        
    W1_npu = np.zeros((98, 8, 8, 8), dtype=np.int8)
    for i in range(98):
        for j in range(8):
            w_block = W1_q[i*8:(i+1)*8, j*8:(j+1)*8]
            W1_npu[i, j] = format_weight_block(w_block)
    
    write_blob('weights_l1.bin', W1_npu)
    write_blob('bias_l1.bin', b1_q)
    
    W2_padded = np.zeros((64, 16), dtype=np.int8)
    W2_padded[:, :10] = W2_q
    W2_npu = np.zeros((8, 2, 8, 8), dtype=np.int8)
    for i in range(8):
        for j in range(2):
            w_block = W2_padded[i*8:(i+1)*8, j*8:(j+1)*8]
            W2_npu[i, j] = format_weight_block(w_block)
            
    write_blob('weights_l2.bin', W2_npu)
    
    b2_padded = np.zeros(16, dtype=np.int32)
    b2_padded[:10] = b2_q
    write_blob('bias_l2.bin', b2_padded)

    with open('model_params.h', 'w') as f:
        f.write("#pragma once\n")
        f.write(f"#define SHIFT1 {shift1}\n")
        f.write(f"#define SHIFT2 {shift2}\n")

if __name__ == '__main__':
    print("Loading MNIST...")
    dataset = fetch_mnist()
    
    print("Training 2-layer MLP (100 Epochs)...")
    X_train = dataset['train_x'] / 255.0
    Y_train = dataset['train_y']
    X_test = dataset['test_x'] / 255.0
    Y_test = dataset['test_y']
    
    W1, b1, W2, b2 = train_mlp(X_train, Y_train, epochs=100, lr=0.1)
    
    print("Exporting to NPU binaries...")
    export_binaries(dataset['test_x'], Y_test, W1, b1, W2, b2)
    print("Done! Generated .bin and .h files.")
