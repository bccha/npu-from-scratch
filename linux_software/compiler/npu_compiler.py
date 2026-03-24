import os

class NPULayer:
    """Base Polymorphic NPU Compiler Node"""
    def __init__(self, layer_type, name, weight_addr, has_relu, shift_val=1):
        self.type = layer_type
        self.name = name
        self.weight_addr = weight_addr
        self.has_relu = has_relu
        self.shift_val = shift_val

    def emit_c_code(self, idx, is_cnn_mode, is_last_layer):
        raise NotImplementedError("Layer subclasses must implement emit_c_code()")


class NPUConv2DLayer(NPULayer):
    def __init__(self, name, in_c, out_c, k, stride, pad, in_h, in_w, out_h, out_w, patches, in_tiles, out_tiles, w_addr, has_relu, pool_size, shift_val):
        super().__init__('conv2d', name, w_addr, has_relu, shift_val)
        self.in_c, self.out_c = in_c, out_c
        self.k, self.stride, self.pad = k, stride, pad
        self.in_h, self.in_w, self.out_h, self.out_w = in_h, in_w, out_h, out_w
        self.patches, self.in_tiles, self.out_tiles = patches, in_tiles, out_tiles
        self.pool_size = pool_size

    def emit_c_code(self, idx, is_cnn_mode, is_last_layer):
        c = []
        c.append(f"        // -------- Layer {idx+1} [Conv2D]: {self.name} --------")
        c.append(f"        int8_t* X_hw = (int8_t*)(virt_ddr_base + 0x900000);")
        c.append(f"        memset(X_hw, 0, (({self.patches} + 7)/8) * {self.in_tiles} * 64);")
        
        if idx == 0:
            c.append(f"        uint8_t* img_src = (uint8_t*)(virt_ddr_base + 0x000000 + b * 28 * 28);")
        else:
            c.append(f"        uint8_t* img_src = (uint8_t*)H_flat;")
        
        c.append("        // Software Im2Col HW-Aligned Padding & Unrolling")
        c.append(f"        int patch_idx = 0;")
        c.append(f"        for(int py = 0; py < {self.out_h}; py++) {{")
        c.append(f"            for(int px = 0; px < {self.out_w}; px++) {{")
        c.append(f"                int feature_idx = 0;")
        c.append(f"                for(int c_in = 0; c_in < {self.in_c}; c_in++) {{")
        c.append(f"                    for(int ky = 0; ky < {self.k}; ky++) {{")
        c.append(f"                        for(int kx = 0; kx < {self.k}; kx++) {{")
        c.append(f"                            int orig_y = (py * {self.stride}) - {self.pad} + ky;")
        c.append(f"                            int orig_x = (px * {self.stride}) - {self.pad} + kx;")
        c.append(f"                            int8_t pval = 0;")
        c.append(f"                            if (orig_y >= 0 && orig_y < {self.in_h} && orig_x >=0 && orig_x < {self.in_w}) {{")
        c.append(f"                                pval = (int8_t)img_src[c_in * {self.in_h * self.in_w} + orig_y * {self.in_w} + orig_x];")
        c.append(f"                            }}")
        c.append(f"                            int p_tile = patch_idx / 8; int p_rem = patch_idx % 8;")
        c.append(f"                            int f_tile = feature_idx / 8; int f_rem = feature_idx % 8;")
        c.append(f"                            int mem_offset = (p_tile * {self.in_tiles} + f_tile) * 64 + (p_rem * 8 + f_rem);")
        c.append(f"                            X_hw[mem_offset] = pval;")
        c.append("                            feature_idx++;")
        c.append("                        }")
        c.append("                    }")
        c.append("                }")
        c.append("                patch_idx++;")
        c.append("            }")
        c.append("        }\n")
        
        c.append(f"        int out_patches_tiles = ({self.patches} + 7) / 8;")
        c.append(f"        int8_t* Y_img_out = (int8_t*)(virt_ddr_base + 0x820000); // Temporary Spatial Output")
        c.append(f"        memset(Y_img_out, 0, {self.out_c} * {self.patches});")
        
        c.append(f"        for (int p_t = 0; p_t < out_patches_tiles; p_t++) {{")
        c.append(f"            for (int j = 0; j < {self.out_tiles}; j++) {{")
        c.append("                npu_clear_ocm(); npu_set_seq_rows(8);")
        c.append(f"                for (int t = 0; t < {self.in_tiles}; t++) {{")
        c.append(f"                    npu_load_weights(0x{self.weight_addr:06X} + (t * {self.out_tiles} + j) * 64);")
        c.append("                    npu_accum_start();")
        c.append(f"                    npu_load_inputs(0x900000 + (p_t * {self.in_tiles} + t) * 64);")
        c.append("                    npu_wait_accum();")
        c.append("                }")
        c.append(f"                npu_set_shift({self.shift_val});")
        c.append(f"                npu_load_bias(&bias_l{idx+1}[j * 8]);")
        c.append(f"                npu_drain_to_ddr(0x910000);")
        c.append("                for(int r=0; r<8; r++) {")
        c.append(f"                    int absolute_patch = p_t * 8 + r; if (absolute_patch >= {self.patches}) continue;")
        c.append("                    for(int c=0; c<8; c++) {")
        c.append(f"                        int absolute_channel = j * 8 + c; if (absolute_channel >= {self.out_c}) continue;")
        c.append("                        int8_t clamped = ((int8_t*)(virt_ddr_base + 0x910000))[r * 8 + (7 - c)];")
        c.append(f"                        Y_img_out[absolute_channel * {self.patches} + absolute_patch] = clamped;")
        c.append("                    }")
        c.append("                }")
        c.append("            }")
        c.append("        }\n")
        
        ps = self.pool_size[0] if isinstance(self.pool_size, tuple) else self.pool_size
        if ps > 1:
            ph, pw = self.out_h // ps, self.out_w // ps
            c.append("        // MaxPool2D execution (CPU Reference Fallback)")
            c.append(f"        for(int c_out=0; c_out<{self.out_c}; c_out++) {{")
            c.append(f"            for(int py=0; py<{ph}; py++) {{")
            c.append(f"                for(int px=0; px<{pw}; px++) {{")
            c.append("                    int8_t max_v = -128;")
            c.append(f"                    for(int ky=0; ky<{ps}; ky++) {{")
            c.append(f"                        for(int kx=0; kx<{ps}; kx++) {{")
            c.append(f"                            int orig_y = py*{ps} + ky; int orig_x = px*{ps} + kx;")
            c.append(f"                            int8_t v = Y_img_out[c_out*{self.out_h * self.out_w} + orig_y*{self.out_w} + orig_x];")
            c.append(f"                            if (v > max_v) max_v = v;")
            c.append("                        }")
            c.append("                    }")
            c.append(f"                    H_flat[c_out*{ph * pw} + py*{pw} + px] = max_v;")
            c.append("                }")
            c.append("            }")
            c.append("        }\n")
        else:
            c.append(f"        memcpy(H_flat, Y_img_out, {self.out_c} * {self.patches});\n")
        return c


class NPULinearLayer(NPULayer):
    def __init__(self, name, in_f, out_f, in_tiles, out_tiles, w_addr, has_relu, shift_val):
        super().__init__('linear', name, w_addr, has_relu, shift_val)
        self.in_f, self.out_f = in_f, out_f
        self.in_tiles, self.out_tiles = in_tiles, out_tiles

    def emit_c_code(self, idx, is_cnn_mode, is_last_layer):
        c = []
        c.append(f"        // -------- Layer {idx+1} [Linear]: {self.name} --------")
        c.append(f"        for (int j = 0; j < {self.out_tiles}; j++) {{")
        c.append("            npu_clear_ocm(); npu_set_seq_rows(8);")
        c.append(f"            for (int t = 0; t < {self.in_tiles}; t++) {{")
        
        if idx == 0 and not is_cnn_mode:
            c.append(f"                uint32_t curr_x_off = 0x000000 + (b * {self.in_tiles} + t) * 64;")
        else:
            c.append("                uint8_t *h_ptr = virt_ddr_base + 0x904000;")
            c.append("                for(int r = 0; r < 8; r++) {")
            c.append("                    uint32_t low = 0, high = 0;")
            c.append("                    for(int c_i=0; c_i<4; c_i++) {")
            if is_cnn_mode:
                c.append(f"                        int8_t v0 = (r == 0 && (t*8+c_i) < {self.in_f}) ? H_flat[t*8+c_i] : 0;")
                c.append(f"                        int8_t v1 = (r == 0 && (t*8+c_i+4) < {self.in_f}) ? H_flat[t*8+c_i+4] : 0;")
                c.append("                        low |= (((uint32_t)(uint8_t)v0) << (c_i*8));")
                c.append("                        high |= (((uint32_t)(uint8_t)v1) << (c_i*8));")
            else:
                c.append(f"                        low |= (((uint32_t)(uint8_t)H_flat[r * {self.in_f} + t*8+c_i]) << (c_i*8));")
                c.append(f"                        high |= (((uint32_t)(uint8_t)H_flat[r * {self.in_f} + t*8+c_i+4]) << (c_i*8));")
            c.append("                    }")
            c.append("                    ((volatile uint32_t*)h_ptr)[r*2+0] = low; ((volatile uint32_t*)h_ptr)[r*2+1] = high;")
            c.append("                }")
            c.append("                volatile uint32_t dummy = ((volatile uint32_t*)h_ptr)[0]; (void)dummy;")
            c.append("                uint32_t curr_x_off = 0x904000;")
            
        c.append(f"                npu_load_weights(0x{self.weight_addr:06X} + (t * {self.out_tiles} + j) * 64);")
        c.append("                npu_accum_start();")
        c.append("                npu_load_inputs(curr_x_off);")
        c.append("                npu_wait_accum();")
        c.append("            }")
        
        if not is_last_layer:
            c.append(f"            npu_set_shift({self.shift_val});")
            c.append(f"            npu_load_bias(&bias_l{idx+1}[j * 8]);")
            c.append(f"            npu_drain_to_ddr(0x910000);")
            c.append("            for(int r=0; r<8; r++) { for(int c_i=0; c_i<8; c_i++) {")
            c.append("                int8_t clamped = ((int8_t*)(virt_ddr_base + 0x910000))[r * 8 + (7 - c_i)];")
            if is_cnn_mode:
                c.append("                if (r==0) ((int8_t*)(virt_ddr_base + 0x930000))[j*8+c_i] = clamped;")
            else:
                c.append(f"                ((int8_t*)(virt_ddr_base + 0x930000))[r * {self.out_f} + (j*8+c_i)] = clamped;")
            c.append("            } }")
        else:
            c.append("            int32_t z_b[64] = {0};")
            c.append(f"            npu_extract_32bit_ocm(z_b, &bias_l{idx+1}[j * 8]);")
            c.append("            for(int r=0; r<8; r++) { for(int c_i=0; c_i<8; c_i++) { ((int32_t*)virt_ddr_base)[0x910000/4 + (r*16) + (j*8+c_i)] = z_b[r*8+c_i]; } }")
        c.append("        }\n")
        
        if not is_last_layer:
            c.append(f"        memcpy(H_flat, (int8_t*)(virt_ddr_base + 0x930000), {self.out_f if is_cnn_mode else self.out_f * 8});\n")

        return c


class NPUCompiler:
    """
    Bring-Your-Own-Compiler (BYOC) Auto-Codegen Backend for Cyclone V Systolic Arrays.
    Dynamically emits a bare-metal Linux C driver mimicking PyTorch Graph topology.
    Fully leverages Polymorphic Strategy classes (`NPUConv2DLayer`, `NPULinearLayer`).
    """
    def __init__(self, out_c_file="npu_model_runtime.c"):
        self.code = []
        self.out_c_file = out_c_file
        self.mem_weights = 0x800000
        self.layers = []  # Stores polymorphic NPULayer objects
        
    def allocate_weights(self, num_bytes):
        addr = self.mem_weights
        self.mem_weights += num_bytes
        self.mem_weights = (self.mem_weights + 0xFFFF) & ~0xFFFF
        return addr
        
    def register_linear_fused_layer(self, name, in_features, out_features, has_relu, shift_val=8):
        in_tiles = in_features // 8 + (1 if in_features % 8 else 0)
        out_tiles = out_features // 8 + (1 if out_features % 8 else 0)
        w_addr = self.allocate_weights(in_tiles * out_tiles * 64)
        
        layer = NPULinearLayer(name, in_features, out_features, in_tiles, out_tiles, w_addr, has_relu, shift_val)
        self.layers.append(layer)
        return layer

    def register_conv_fused_layer(self, name, in_channels, out_channels, kernel_size, stride, padding, in_h, in_w, has_relu, pool_size, shift_val=8):
        out_h = (in_h + 2 * padding - kernel_size) // stride + 1
        out_w = (in_w + 2 * padding - kernel_size) // stride + 1
        patches = out_h * out_w
        
        in_features = in_channels * kernel_size * kernel_size
        in_tiles = in_features // 8 + (1 if in_features % 8 else 0)
        out_tiles = out_channels // 8 + (1 if out_channels % 8 else 0)
        w_addr = self.allocate_weights(in_tiles * out_tiles * 64)
        
        layer = NPUConv2DLayer(
            name, in_channels, out_channels, kernel_size, stride, padding,
            in_h, in_w, out_h, out_w, patches, in_tiles, out_tiles, w_addr, has_relu, pool_size, shift_val
        )
        self.layers.append(layer)
        return layer

    def generate_c_code(self):
        c = []
        c.append("// ===========================================================================")
        c.append("// AUTO-GENERATED BY NPU COMPILE PASS (TRUE OOP POLYMORPHISM)")
        c.append("// ===========================================================================")
        c.append("#include \"npu_api.h\"")
        c.append("#include <stdio.h>")
        c.append("#include <stdlib.h>")
        c.append("#include <string.h>")
        c.append("#include <sys/time.h>\n")
        
        for idx, layer in enumerate(self.layers):
            c.append(f"int32_t bias_l{idx+1}[{layer.out_tiles * 8}];")
        c.append("int32_t labels[10000];\n")
        
        c.append("void npu_auto_inference(int num_batches, double *time_ms, double *accuracy) {")
        c.append("    int correct_predictions = 0;")
        c.append("    struct timeval tv; gettimeofday(&tv, NULL);")
        c.append("    double start_time = (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;")
        c.append("    int8_t *l1_drain_buf = (int8_t*)(virt_ddr_base + 0x900000); // Scratchpad Ping")
        c.append("    int8_t *H_flat = (int8_t*)(virt_ddr_base + 0x920000); // 1MB Global Flat Bridge Tensor\n")
        
        is_cnn_mode = any(l.type == 'conv2d' for l in self.layers)
        c.append(f"    for (int b = 0; b < num_batches; b++) {{")
        c.append("        if (b % 100 == 0) printf(\"    [EmitC] Processing Sequence %d...\\n\", b);\n")
        
        for idx, layer in enumerate(self.layers):
            is_last = (idx == len(self.layers) - 1)
            # DYNAMIC DISPATCH! The pure polymorphic emission of C constraints!
            c.extend(layer.emit_c_code(idx, is_cnn_mode, is_last))
        
        c.append("        int32_t* Y_final = (int32_t*)(virt_ddr_base + 0x910000);")
        if is_cnn_mode:
            c.append("        int best_class = 0; int32_t max_val = Y_final[0];")
            c.append("        for (int k = 1; k < 10; k++) { if (Y_final[k] > max_val) { max_val = Y_final[k]; best_class = k; } }")
            c.append("        if (best_class == labels[b]) correct_predictions++;")
        else:
            c.append("        for (int r = 0; r < 8; r++) {")
            c.append("            int best_class = 0; int32_t max_val = Y_final[r*16 + 0];")
            c.append("            for (int k = 1; k < 10; k++) { if (Y_final[r*16 + k] > max_val) { max_val = Y_final[r*16 + k]; best_class = k; } }")
            c.append("            if (best_class == labels[b * 8 + r]) correct_predictions++;")
            c.append("        }")
        c.append("    }\n")
        
        c.append("    gettimeofday(&tv, NULL);")
        c.append("    double e_t = (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;")
        c.append("    *time_ms = (e_t - start_time) / 1000.0;")
        if is_cnn_mode:
            c.append("    *accuracy = (double)correct_predictions / num_batches * 100.0;")
        else:
            c.append("    *accuracy = (double)correct_predictions / (num_batches * 8) * 100.0;")
        c.append("}\n")
        
        c.append("int main(int argc, char **argv) {")
        if is_cnn_mode:
            c.append("    int num_test_batches = 100;")
            c.append("    if (argc > 1) { num_test_batches = atoi(argv[1]); }")
        else:
            c.append("    int num_test_batches = 125;")
            c.append("    if (argc > 1) { num_test_batches = atoi(argv[1]) / 8; }")
            
        c.append("    printf(\"=== Auto-Generated EmitC OOP Benchmark ===\\n\");")
        c.append("    if (npu_init() < 0) return -1;")
        c.append("    printf(\"\\nLoading Binaries...\\n\");")
        c.append("    npu_load_binary_file(\"inputs.bin\", virt_ddr_base + 0x000000, 8000000);")
        
        for idx, layer in enumerate(self.layers):
            c.append(f"    npu_load_binary_file(\"weights_l{idx+1}.bin\", virt_ddr_base + 0x{layer.weight_addr:06X}, 200000);")
            c.append(f"    npu_load_binary_file(\"bias_l{idx+1}.bin\", bias_l{idx+1}, 256);")
            
        c.append("    npu_load_binary_file(\"labels.bin\", labels, 40000);")
        c.append("    double time_ms, acc;")
        c.append("    npu_auto_inference(num_test_batches, &time_ms, &acc);")
        c.append("    printf(\"    EmitC Accuracy : %.2f%%\\n\", acc);")
        if is_cnn_mode:
            c.append("    printf(\"    EmitC Time     : %.2f ms (%.3f ms/img)\\n\", time_ms, time_ms / num_test_batches);")
        else:
            c.append("    printf(\"    EmitC Time     : %.2f ms (%.3f ms/img)\\n\", time_ms, time_ms / (num_test_batches * 8));")
        c.append("    npu_close(); return 0;")
        c.append("}\n")
        
        with open(self.out_c_file, "w") as f:
            f.write("\n".join(c))
        print(f"[EmitC] Generated NPU Standalone Execution Runtime: {self.out_c_file}")
