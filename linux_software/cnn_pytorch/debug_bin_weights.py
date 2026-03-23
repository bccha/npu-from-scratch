import numpy as np

def analyze(name, W):
    total = W.size
    nz = np.count_nonzero(W)
    print(f"[{name}] Shape: {W.shape}, Non-Zero Elements: {nz}/{total} ({nz/total*100:.2f}%)")
    print(f"[{name}] Absolute Max Weight: {np.max(np.abs(W))}")
    
def load_weights(file_path, shape):
    with open(file_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.int8).reshape(shape)

def load_bias(file_path, shape):
    with open(file_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.int32).reshape(shape)

print("=== Examining Raw Int8 Quantization Quality ===")
analyze("W1", load_weights("weights_l1.bin", (2, 1, 8, 8)))
analyze("W2", load_weights("weights_l2.bin", (169, 16, 8, 8)))
analyze("W3", load_weights("weights_l3.bin", (16, 2, 8, 8)))

print("\n=== Examining Bias Scales ===")
b1 = load_bias("bias_l1.bin", (8,))
b2 = load_bias("bias_l2.bin", (128,))
b3 = load_bias("bias_l3.bin", (16,))
print(f"[B1] Max: {np.max(np.abs(b1))}")
print(f"[B2] Max: {np.max(np.abs(b2))}")
print(f"[B3] Max: {np.max(np.abs(b3))}")
