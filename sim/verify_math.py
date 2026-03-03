import struct
import random

def to_int8(val):
    val = val & 0xFF
    if val & 0x80:
        return val - 256
    return val

def to_int32(val):
    val = val & 0xFFFFFFFF
    if val & 0x80000000:
        return val - 0x100000000
    return val

def bswap32(val):
    val = val & 0xFFFFFFFF
    return ((val & 0xFF) << 24) | (((val >> 8) & 0xFF) << 16) | (((val >> 16) & 0xFF) << 8) | (((val >> 24) & 0xFF))

print("=== DE10-Nano Layer 2 Output Emulation ===")

# Create deterministic mock data mimicking actual NPU Layer 2 output
Z2_CPU = [[0]*8 for _ in range(8)]
for r in range(8):
    for c in range(8):
        Z2_CPU[r][c] = random.randint(-50000, 50000)

bias_l2 = [to_int32(300) for _ in range(8)]
shift_val2 = 8

# 1. CPU Layer 2 Math (from main.c)
print("\n[Layer 2 CPU S/W Output]")
Y_final = [[0]*16 for _ in range(8)]
for j in range(2):
    for r in range(8):
        for c in range(8):
            # In C code: Y_final[r][j * 8 + c] = Z2[r][c] + bias_l2[j * 8 + c]
            Y_final[r][j * 8 + c] = Z2_CPU[r][c] + bias_l2[c] # Note: test mock bias is length 8, C code was j*8+c for a 16 array

print("Row 0: ", Y_final[0])

# Argmax on CPU array
best_class_cpu = 0
max_val_cpu = Y_final[0][0]
for c in range(1, 10):
    if Y_final[0][c] > max_val_cpu:
        max_val_cpu = Y_final[0][c]
        best_class_cpu = c
        
print(f"Argmax CPU (Row 0): {best_class_cpu} (val {max_val_cpu})")


# 2. HW Layer 2 Math
print("\n[Layer 2 HW Output]")
# 1. HW generates byte-swapped output and XORs columns
Z_OCM_Memory = [0]*64
for r in range(8):
    for c in range(8):
        hw_c = c ^ 1
        Z_OCM_Memory[r * 8 + hw_c] = to_int32(bswap32(to_int32(Z2_CPU[r][c])))

# 2. Hardware Post Processor reads it
out_byte_array = [0]*64
for i in range(64):
    mem_z = Z_OCM_Memory[i]
    hw_z = to_int32(bswap32(mem_z))
    
    raw = hw_z + bias_l2[i % 8]
    shifted = raw >> shift_val2
    
    # HARDWARE ALWAYS CLAMPS AND SHIFTS!
    if shifted < 0: final = 0
    elif shifted > 127: final = 127
    else: final = shifted
    
    word_idx = i // 4
    local_idx = i % 4
    hw_pos = local_idx ^ 1
    
    out_byte_array[word_idx * 4 + hw_pos] = final

# How `main.c` reads HW final result
print("Row 0 Byte Array as read by `int8_t *final_results`:")
print([to_int8(x) for x in out_byte_array[0:16]]) 

# Argmax on HW byte array (main.c logic)
best_class_hw = 0
max_val_hw = to_int8(out_byte_array[0])
for c in range(1, 10):
    val = to_int8(out_byte_array[c])
    if val > max_val_hw:
        max_val_hw = val
        best_class_hw = c

print(f"Argmax HW (Row 0): {best_class_hw} (val {max_val_hw})")

if best_class_cpu == best_class_hw:
    print("MATCH: Argmax selects the same class.")
else:
    print("MISMATCH: Hardware Clamp/Shift alters Argmax outcome!!")
