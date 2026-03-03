import os
import struct
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def im2col(X, kernel_size, stride=1, padding=0):
    """
    Python implementation of im2col (Image to Column) for Conv2D.
    X: Input image tensor of shape (N, H, W, C)
    kernel_size: Int or tuple (Kh, Kw)
    stride: Int or tuple (Sh, Sw)
    padding: Int (zero padding around H, W)
    
    Returns:
    X_col: Flattened image patches of shape (N * H_out * W_out, Kh * Kw * C)
    """
    N, H, W, C = X.shape
    
    if isinstance(kernel_size, int):
        Kh = Kw = kernel_size
    else:
        Kh, Kw = kernel_size
        
    if isinstance(stride, int):
        Sh = Sw = stride
    else:
        Sh, Sw = stride
        
    # Apply padding
    if padding > 0:
        X_padded = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')
    else:
        X_padded = X
        
    H_p, W_p = X_padded.shape[1], X_padded.shape[2]
    
    # Calculate output dimensions
    H_out = (H_p - Kh) // Sh + 1
    W_out = (W_p - Kw) // Sw + 1
    
    # Extract patches
    patches = []
    for i in range(N):
        img_patches = []
        for h in range(H_out):
            for w in range(W_out):
                h_start, h_end = h * Sh, h * Sh + Kh
                w_start, w_end = w * Sw, w * Sw + Kw
                # Extract patch [Kh, Kw, C]
                patch = X_padded[i, h_start:h_end, w_start:w_end, :]
                img_patches.append(patch.flatten())
        patches.append(img_patches)
        
    # Shape: (N, H_out * W_out, Kh * Kw * C)
    X_col = np.array(patches)
    # Reshape to 2D: (N * H_out * W_out, Kh * Kw * C)
    X_col = X_col.reshape(N * H_out * W_out, Kh * Kw * C)
    
    return X_col, H_out, W_out


def test_im2col_vs_conv2d():
    print("--- Verifying im2col Logic ---")
    # Synthetic Image: N=1, H=4, W=4, Channels=1
    X = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    
    # Synthetic Weights: F=2 (filters), Kh=3, Kw=3, C=1
    # Shape for standard TF/Keras conv2d: (Kh, Kw, C, F)
    # To use im2col dot product, we flatten weights to (Kh*Kw*C, F)
    W = np.ones((3, 3, 1, 2), dtype=np.float32)
    # Give filter 2 a different value
    W[:, :, :, 1] = 2.0 
    
    W_col = W.reshape(3*3*1, 2)
    
    # im2col Transform
    X_col, h_out, w_out = im2col(X, kernel_size=3, stride=1, padding=0)
    
    print(f"Original X shape: {X.shape}")
    print(f"im2col X_col shape: {X_col.shape} (Expected N*H_out*W_out x Kh*Kw*C)")
    print(f"Flattened W_col shape: {W_col.shape} (Expected Kh*Kw*C x F)")
    
    # Matrix Multiplication (The NPU action)
    # Y = X_col * W_col => (N*H_out*W_out, Kh*Kw*C) x (Kh*Kw*C, F) => (N*H_out*W_out, F)
    Y_col = np.dot(X_col, W_col)
    
    # Reshape back to Image topology
    N = X.shape[0]
    Y_conv = Y_col.reshape(N, h_out, w_out, 2)
    
    print(f"Output Convolution Shape: {Y_conv.shape}")
    print("Sample Output (Filter 1):\n", Y_conv[0, :, :, 0])
    print("Sample Output (Filter 2):\n", Y_conv[0, :, :, 1])
    print("im2col Verification PASSED.\n")
    return True

def load_mnist(image_file, label_file):
    with open(image_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols, 1)
    with open(label_file, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return images, labels

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def one_hot(Y, num_classes=10):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def train_cnn(X_train, Y_train, X_test, Y_test, epochs=3, batch_size=64, learning_rate=0.01):
    N, H, W, C = X_train.shape
    num_classes = 10
    
    # Conv Layer 1: 8 filters of 3x3x1
    F = 8
    Kh, Kw = 3, 3
    stride = 2 # Stride 2 to reduce spatial dimensions quickly (14x14 output)
    
    # Calculate Conv Output dimensions
    H_out = (H - Kh) // stride + 1
    W_out = (W - Kw) // stride + 1
    
    # FC Layer dimensions
    fc_input_dim = H_out * W_out * F
    
    # Initialize Weights
    W_conv = np.random.randn(Kh * Kw * C, F) * np.sqrt(2.0 / (Kh * Kw * C))
    b_conv = np.zeros(F)
    
    W_fc = np.random.randn(fc_input_dim, num_classes) * np.sqrt(2.0 / fc_input_dim)
    b_fc = np.zeros(num_classes)
    
    num_batches = N // batch_size
    
    print("Starting CNN Training (1 Conv + 1 FC)...")
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]
            Y_batch_oh = one_hot(Y_batch)
            
            # --- FORWARD PASS ---
            X_col, h_out, w_out = im2col(X_batch, kernel_size=(Kh, Kw), stride=stride)
            
            # Conv Layer (Executed as GEMM)
            # Z_conv_col: (batch_size * h_out * w_out, F)
            Z_conv_col = np.dot(X_col, W_conv) + b_conv
            A_conv_col = relu(Z_conv_col)
            
            # Reshape for FC
            # Flatten spatial dimensions per image: (batch_size, h_out * w_out * F)
            A_fc_in = A_conv_col.reshape(batch_size, h_out * w_out * F)
            
            # FC Layer (Execution as GEMM)
            Z_fc = np.dot(A_fc_in, W_fc) + b_fc
            A_fc_out = softmax(Z_fc)
            
            # --- BACKWARD PASS ---
            # Output Layer Error
            dZ_fc = (A_fc_out - Y_batch_oh) / batch_size
            
            dW_fc = np.dot(A_fc_in.T, dZ_fc)
            db_fc = np.sum(dZ_fc, axis=0)
            
            # Error back to Conv
            dA_fc_in = np.dot(dZ_fc, W_fc.T)
            dA_conv_col = dA_fc_in.reshape(batch_size * h_out * w_out, F)
            
            dZ_conv_col = dA_conv_col * relu_deriv(Z_conv_col)
            
            dW_conv = np.dot(X_col.T, dZ_conv_col)
            db_conv = np.sum(dZ_conv_col, axis=0)
            
            # Update weights
            W_fc -= learning_rate * dW_fc
            b_fc -= learning_rate * db_fc
            W_conv -= learning_rate * dW_conv
            b_conv -= learning_rate * db_conv
            
        # Evaluation step
        X_test_col, _, _ = im2col(X_test, kernel_size=(Kh, Kw), stride=stride)
        Z_test_conv_col = np.dot(X_test_col, W_conv) + b_conv
        A_test_conv_col = relu(Z_test_conv_col)
        
        A_test_fc_in = A_test_conv_col.reshape(X_test.shape[0], h_out * w_out * F)
        Z_test_fc = np.dot(A_test_fc_in, W_fc) + b_fc
        predictions = np.argmax(Z_test_fc, axis=1)
        
        acc = get_accuracy(predictions, Y_test)
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {acc:.4f}")
        
    return W_conv, b_conv, W_fc, b_fc

def quantize_and_export(W, b, file_prefix, scale=127.0):
    # Quantize to Int8 (Symmetric, simple magnitude scaling)
    # The actual scaling mechanism will be improved with QAT later.
    max_val = max(np.max(np.abs(W)), np.max(np.abs(b)), 1e-5)
    scale_factor = scale / max_val
    
    W_q = np.round(W * scale_factor).astype(np.int8)
    b_q = np.round(b * scale_factor).astype(np.int8)
    
    # Export weights
    with open(f"{file_prefix}_weights.bin", "wb") as f:
        f.write(W_q.tobytes())
    with open(f"{file_prefix}_bias.bin", "wb") as f:
        f.write(b_q.tobytes())
        
    print(f"Exported {file_prefix}_weights.bin : {W_q.shape}")
    print(f"Exported {file_prefix}_bias.bin    : {b_q.shape}")
    return scale_factor

if __name__ == "__main__":
    test_im2col_vs_conv2d()
    
    # Real Training
    try:
        X_train, Y_train = load_mnist('../mnist_test/train-images-idx3-ubyte', '../mnist_test/train-labels-idx1-ubyte')
        X_test, Y_test = load_mnist('../mnist_test/t10k-images-idx3-ubyte', '../mnist_test/t10k-labels-idx1-ubyte')
        
        # Normalize and zero-center
        X_train = (X_train.astype(np.float32) - 128.0) / 128.0
        X_test = (X_test.astype(np.float32) - 128.0) / 128.0
        
        W_conv, b_conv, W_fc, b_fc = train_cnn(X_train[:10000], Y_train[:10000], X_test[:1000], Y_test[:1000], epochs=5)
        
        print("\n--- Quantizing and Exporting for NPU ---")
        
        # Note: Pad W_fc because NPU needs multiples of 8. 
        # Current logic is 8 filters out of the conv layer, output map size 13x13 -> 13x13x8 = 1352
        # FC matches (1352, 10). Pad 10 to 16.
        W_fc_padded = np.zeros((W_fc.shape[0], 16), dtype=np.float32)
        W_fc_padded[:, :10] = W_fc
        b_fc_padded = np.zeros(16, dtype=np.float32)
        b_fc_padded[:10] = b_fc
        
        quantize_and_export(W_conv, b_conv, "conv")
        quantize_and_export(W_fc_padded, b_fc_padded, "fc")
        
    except Exception as e:
        print(f"Execution Error: {e}")

