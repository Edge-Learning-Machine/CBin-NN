import os
import numpy as np

if os.path.exists("bnn_params.h"): 
    os.remove("bnn_params.h") 
if os.path.exists("bnn_params.c"): 
    os.remove("bnn_params.c") 
if os.path.exists("CBin-NN.c"): 
    os.remove("CBin-NN.c") 

def createArray(type, arrName, arr, n_elements):
    stri = f'{type} {arrName}[{n_elements}] = {{'
    for i, n in enumerate(arr):
        if type=='int':
            n = n.astype(int)
        stri = stri + str(n) + ','
    stri = stri + '};\n'
    stri = stri.replace(',}', '}')
    return stri

def save_bnn_params(weights, bias, layer_name, layer_idx, in_shape, out_shape, ker_size, stride, padding, first_file_write, in_buffer, out_buffer, BN_idx):
    myFile = open(f"bnn_params.h","a+")
    if layer_name.startswith('QBConv2D') or layer_name.startswith('QBConv2D_Optimized') or layer_name.startswith('QBConv2D_Optimized_PReLU') or layer_name.startswith('BBConv2D') or layer_name.startswith('BBConv2D_Optimized') or layer_name.startswith('BBConv2D_Optimized_PReLU') or layer_name.startswith('BBPointwiseConv2D') or layer_name.startswith('BBPointwiseConv2D_Optimized') or layer_name.startswith('BBPointwiseConv2D_Optimized_PReLU'):
        conv_wt = 'CONV' + str(layer_idx) + '_WT'
        conv_bias = 'CONV' + str(layer_idx) + '_BIAS'
        in_dim = 'CONV' + str(layer_idx) + '_IN_DIM'
        in_ch = 'CONV' + str(layer_idx) + '_IN_CH'
        conv_ker_size = 'CONV' + str(layer_idx) + '_KER_SIZE'
        out_ch = 'CONV' + str(layer_idx) + '_OUT_CH'
        out_dim = 'CONV' + str(layer_idx) + '_OUT_DIM'
        conv_stride = 'CONV' + str(layer_idx) + '_STRIDE'
        conv_padding = 'CONV' + str(layer_idx) + '_PADDING'

        alpha1_wt = 'BN' + str(BN_idx) + '_ALPHA1' 
        alpha2_wt = 'BN' + str(BN_idx) + '_ALPHA2'
        bn_wt = 'BN' + str(BN_idx) + '_WT'
        PReLU_shift = 'SHIFT' + str(BN_idx)


        weights = weights.ravel()
        weights = np.packbits(weights, bitorder='little')
        bit_packed_weights = []
        for i in range(0, len(weights), 4):
            w = weights[i:i + 4]
            w = int.from_bytes(w, "little", signed=True)
            bit_packed_weights = np.append(bit_packed_weights, w)
        if bias != None:
            bias = np.packbits(bias, bitorder='little')
            bit_packed_bias = []
            for i in range(0, len(bias), 4):
                b = bias[i:i + 4]
                b = int.from_bytes(b, "little", signed=True)
                bit_packed_bias = np.append(bit_packed_bias, b)

        myFile.write(f"#define N_{conv_wt} {bit_packed_weights.size}\n")
        if bias != None:
            myFile.write(f"#define N_{conv_bias} {bit_packed_bias.size}\n")
        myFile.write(f"#define {in_ch} {in_shape[3]}\n")
        myFile.write(f"#define {in_dim} {in_shape[1]}\n")
        myFile.write(f"#define {conv_ker_size} {ker_size}\n")
        myFile.write(f"#define {out_ch} {out_shape[3]}\n")
        myFile.write(f"#define {out_dim} {out_shape[1]}\n")
        myFile.write(f"#define {conv_stride} {stride}\n")
        myFile.write(f"#define {conv_padding} {padding}\n")
        myFile.write(f"extern int {conv_wt.lower()}[N_{conv_wt}];\n")
        if bias != None:
            myFile.write(f"extern int {conv_bias.lower()}[N_{conv_bias}];\n")
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        if first_file_write == True:
            myFile.write(f"#include \"bnn_params.h\"\n")
        stri = createArray('int', conv_wt.lower(), bit_packed_weights, 'N_'+conv_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        if bias != None:
            stri = createArray('int', conv_bias.lower(), bit_packed_bias, 'N_'+conv_bias)
            myFile.write(stri)
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"CBin-NN.c","a+")
        if first_file_write == True:
            myFile.write(f"#include \"CBin-NN.h\"\n\n")
            myFile.write(f"int buffer1[N_BUFFER1]\n")
            myFile.write(f"int buffer2[N_BUFFER1]\n")
            myFile.write(f"float classification[N_CLASSES]\n\n")
            myFile.write(f"int bnn_main()\n")
            stri = '{'
            myFile.write(stri)
            myFile.write(f"\n")
            
        if layer_name == 'QBConv2d':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {alpha1_wt.lower()}, {alpha2_wt.lower()}, input_image);\n")
        elif layer_name == 'QBConv2D_Optimized':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {alpha1_wt.lower()}, {alpha2_wt.lower()}, input_image);\n")
        elif layer_name == 'QBConv2D_Optimized_PReLU':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {alpha1_wt.lower()}, {alpha2_wt.lower()}, {PReLU_shift.lower()}, input_image);\n")
        elif layer_name == 'BBConv2D':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
        elif layer_name == 'BBConv2D_Optimized':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
        elif layer_name == 'BBConv2D_Optimized_PReLU':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {bn_wt.lower()}, {PReLU_shift.lower()}, {in_buffer});\n")
        elif layer_name == 'BBPointwiseConv2D':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
        elif layer_name == 'BBPointwiseConv2D_Optimized':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
        elif layer_name == 'BBPointwiseConv2D_Optimized_PReLU':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {bn_wt.lower()}, {PReLU_shift.lower()}, {in_buffer});\n")
        myFile.close()

    elif layer_name.startswith('QQConv2D') or layer_name.startswith('QQConv2D_Optimized') or layer_name.startswith('QQConv2D_Optimized_PReLU'):
        conv_wt = 'CONV' + str(layer_idx) + '_WT' 
        conv_bias = 'CONV' + str(layer_idx) + '_BIAS' 
        in_dim = 'CONV' + str(layer_idx) + '_IN_DIM'
        in_ch = 'CONV' + str(layer_idx) + '_IN_CH'
        conv_ker_size = 'CONV' + str(layer_idx) + '_KER_SIZE'
        out_ch = 'CONV' + str(layer_idx) + '_OUT_CH'
        out_dim = 'CONV' + str(layer_idx) + '_OUT_DIM'
        conv_stride = 'CONV' + str(layer_idx) + '_STRIDE'
        conv_padding = 'CONV' + str(layer_idx) + '_PADDING'

        alpha1_wt = 'BN' + str(BN_idx) + '_ALPHA1' 
        alpha2_wt = 'BN' + str(BN_idx) + '_ALPHA2'
        bn_wt = 'BN' + str(BN_idx) + '_WT'
        PReLU_shift = 'SHIFT' + str(BN_idx)


        weights = weights.ravel()

        myFile.write(f"#define N_{conv_wt} {weights.size}\n")
        if bias != None:
            myFile.write(f"#define N_{conv_bias} {bias.size}\n")
        myFile.write(f"#define {in_ch} {in_shape[3]}\n")
        myFile.write(f"#define {in_dim} {in_shape[1]}\n")
        myFile.write(f"#define {conv_ker_size} {ker_size}\n")
        myFile.write(f"#define {out_ch} {out_shape[3]}\n")
        myFile.write(f"#define {out_dim} {out_shape[1]}\n")
        myFile.write(f"#define {conv_stride} {stride}\n")
        myFile.write(f"#define {conv_padding} {padding}\n")
        myFile.write(f"extern int8_t {conv_wt.lower()}[N_{conv_wt}];\n")
        if bias != None:
            myFile.write(f"extern int8_t {conv_bias.lower()}[N_{conv_bias}];\n")
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        if first_file_write == True:
            myFile.write(f"#include \"bnn_params.h\"\n")
        stri = createArray('int8_t', conv_wt.lower(), weights, 'N_'+conv_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        if bias != None:
            stri = createArray('int8_t', conv_bias.lower(), bias, 'N_'+conv_bias)
            myFile.write(stri)
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"CBin-NN.c","a+")
        if first_file_write == True:
            myFile.write(f"#include \"CBin-NN.h\"\n\n")
            myFile.write(f"int buffer1[N_BUFFER1]\n")
            myFile.write(f"int buffer2[N_BUFFER1]\n")
            myFile.write(f"float classification[N_CLASSES]\n\n")
            myFile.write(f"int bnn_main()\n")
            stri = '{'
            myFile.write(stri)
            myFile.write(f"\n")
            
        if layer_name == 'QQConv2d':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {alpha1_wt.lower()}, {alpha2_wt.lower()}, input_image);\n")
        elif layer_name == 'QQConv2D_Optimized':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {alpha1_wt.lower()}, {alpha2_wt.lower()}, input_image);\n")
        elif layer_name == 'QQConv2D_Optimized_PReLU':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_ch}, {out_dim}, {in_ch}, {in_dim}, {conv_ker_size}, {conv_stride}, {conv_padding}, NULL, {conv_wt.lower()}, {alpha1_wt.lower()}, {alpha2_wt.lower()}, {PReLU_shift.lower()}, input_image);\n")
        myFile.close()

    elif layer_name.startswith('BBFC_Optimized_PReLU') or layer_name.startswith('BBFC_Optimized') or layer_name.startswith('BBFC') or layer_name.startswith('BBQFC_Optimized_PReLU') or layer_name.startswith('BBQFC_Optimized') or layer_name.startswith('BBQFC'):
        fc_wt = 'FC' + str(layer_idx) + '_WT' 
        fc_bias = 'FC' + str(layer_idx) + '_BIAS'
        in_dim = 'FC' + str(layer_idx) + '_IN_DIM'
        out_dim = 'FC' + str(layer_idx) + '_OUT_DIM'

        bn_wt = 'BN' + str(BN_idx) + '_WT'
        PReLU_shift = 'SHIFT' + str(BN_idx)

        weights = weights.ravel()
        weights = np.packbits(weights, bitorder='little')
        bit_packed_weights = []
        for i in range(0, len(weights), 4):
            w = weights[i:i + 4]
            w = int.from_bytes(w, "little", signed=True)
            bit_packed_weights = np.append(bit_packed_weights, w)
        if bias != None:
            B_fc = np.packbits(bias, bitorder='little')
            bit_packed_bias = []
            for i in range(0, len(B_fc), 4):
                b = B_fc[i:i + 4]
                b = int.from_bytes(b, "little", signed=True)
                bit_packed_bias = np.append(bit_packed_bias, b)
        myFile.write(f"#define N_{fc_wt} {bit_packed_weights.size}\n")
        if bias != None:
            myFile.write(f"#define N_{fc_bias} {bit_packed_bias.size}\n")
        myFile.write(f"#define {in_dim} {in_shape}\n")
        myFile.write(f"#define {out_dim} {out_shape}\n")
        myFile.write(f"extern int {fc_wt.lower()}[N_{fc_wt}];\n")
        if bias != None:
            myFile.write(f"extern int {fc_bias.lower()}[N_{fc_bias}];\n")
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        stri = createArray('int', fc_wt.lower(), bit_packed_weights, 'N_'+fc_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        if bias != None:
            stri = createArray('int', fc_bias.lower(), bit_packed_bias, 'N_'+fc_bias)
            myFile.write(stri)
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"CBin-NN.c","a+")
        if first_file_write == True:
            myFile.write(f"#include \"CBin-NN.h\"\n\n")
            myFile.write(f"int buffer1[N_BUFFER1]\n")
            myFile.write(f"int buffer2[N_BUFFER1]\n")
            myFile.write(f"float classification[N_CLASSES]\n\n")
            myFile.write(f"int bnn_main()\n")
            stri = '{'
            myFile.write(stri)
            myFile.write(f"\n")

        if layer_name == 'BBFC':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_dim}, {in_dim}, NULL, {fc_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
        elif layer_name == 'BBFC_Optimized':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_dim}, {in_dim}, NULL, {fc_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
        elif layer_name == 'BBFC_Optimized_PReLU':
            myFile.write(f"\t{layer_name}({out_buffer}, {out_dim}, {in_dim}, NULL, {fc_wt.lower()}, {bn_wt.lower()}, {PReLU_shift.lower()}, {in_buffer});\n")
        if layer_name == 'BBQFC':
            myFile.write(f"\t{layer_name}(classification, {out_dim}, {in_dim}, NULL, {fc_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
            stri = '}'
            myFile.write(stri)
        elif layer_name == 'BBQFC_Optimized':
            myFile.write(f"\t{layer_name}(classification, {out_dim}, {in_dim}, NULL, {fc_wt.lower()}, {bn_wt.lower()}, {in_buffer});\n")
            stri = '}'
            myFile.write(stri)
        elif layer_name == 'BBQFC_Optimized_PReLU':
            myFile.write(f"\t{layer_name}(classification, {out_dim}, {in_dim}, NULL, {fc_wt.lower()}, {bn_wt.lower()}, {PReLU_shift.lower()}, {in_buffer});\n")
            stri = '}'
            myFile.write(stri)
        myFile.close()
    
    elif layer_name.startswith('quantized_fc'):
        fc_wt = 'FC' + str(layer_idx) + '_WT' 
        fc_bias = 'FC' + str(layer_idx) + '_BIAS'
        in_dim = 'FC' + str(layer_idx) + '_IN_DIM'
        out_dim = 'FC' + str(layer_idx) + '_OUT_DIM'
        weights = weights.ravel()
        
        myFile.write(f"#define N_{fc_wt} {weights.size}\n")
        if bias != None:
            myFile.write(f"#define N_{fc_bias} {bias.size}\n")
        myFile.write(f"#define {in_dim} {in_shape}\n")
        myFile.write(f"#define {out_dim} {out_shape}\n")
        myFile.write(f"extern int8_t {fc_wt.lower()}[N_{fc_wt}];\n")
        if bias != None:
            myFile.write(f"extern int8_t {fc_bias.lower()}[N_{fc_bias}];\n")
            myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        stri = createArray('int8_t', fc_wt.lower(), bit_packed_weights, 'N_'+fc_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        if bias != None:
            stri = createArray('int8_t', fc_bias.lower(), bit_packed_bias, 'N_'+fc_bias)
            myFile.write(stri)
            myFile.write(f"\n")
        myFile.close()

    elif layer_name.startswith('quantized_BN'):
        alpha1_wt = 'BN' + str(layer_idx) + '_ALPHA1' 
        alpha2_wt = 'BN' + str(layer_idx) + '_ALPHA2'
        # weight = np.ceil(weight)
        myFile.write(f"#define N_{alpha1_wt} {weights.shape[0]}\n")
        myFile.write(f"#define N_{alpha2_wt} {weights.shape[0]}\n")
        myFile.write(f"extern float {alpha1_wt.lower()}[N_{alpha1_wt}];\n")
        myFile.write(f"extern float {alpha2_wt.lower()}[N_{alpha2_wt}];\n")
        myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        stri = createArray('float', alpha1_wt.lower(), weights[0], 'N_'+alpha1_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        stri = createArray('float', alpha2_wt.lower(), weights[1], 'N_'+alpha2_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        myFile.close()

    elif layer_name.startswith('binary_BN'):
        bn_wt = 'BN' + str(layer_idx) + '_WT'
        myFile.write(f"#define N_{bn_wt} {weights.size}\n")
        myFile.write(f"extern float {bn_wt.lower()}[N_{bn_wt}];\n")
        myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        stri = createArray('float', bn_wt.lower(), weights, 'N_'+bn_wt)
        myFile.write(stri)
        myFile.write(f"\n")
        myFile.close()

    elif layer_name.startswith('PReLU'):
        PReLU_shift = 'SHIFT' + str(layer_idx)
        myFile.write(f"#define N_{PReLU_shift} {weights.size}\n")
        myFile.write(f"extern float {PReLU_shift.lower()}[N_{PReLU_shift}];\n")
        myFile.write(f"\n")
        myFile.close()

        myFile = open(f"bnn_params.c","a+")
        stri = createArray('float', PReLU_shift.lower(), weights, 'N_'+PReLU_shift)
        myFile.write(stri)
        myFile.write(f"\n")
        myFile.close()
        
    elif layer_name.startswith('BMaxPool2D') or layer_name.startswith('BMaxPool2D_Optimized'):
        in_dim = 'POOL' + str(layer_idx) + '_IN_DIM'
        out_dim = 'POOL' + str(layer_idx) + '_OUT_DIM'
        in_ch = 'POOL' + str(layer_idx) + '_CH'
        pool_ker_size = 'POOL' + str(layer_idx) + '_KER_SIZE'
        pool_stride = 'POOL' + str(layer_idx) + '_STRIDE'
        pool_padding = 'POOL' + str(layer_idx) + '_PADDING'

        myFile.write(f"#define {pool_ker_size} {ker_size}\n")
        myFile.write(f"#define {pool_stride} {stride}\n")
        myFile.write(f"#define {pool_padding} {padding}\n")
        myFile.write(f"#define {in_dim} {in_shape[1]}\n")
        myFile.write(f"#define {out_dim} {out_shape[1]}\n")
        myFile.write(f"#define {in_ch} {in_shape[3]}\n")
        myFile.write(f"\n")
        myFile.close()

        myFile = open(f"CBin-NN.c","a+")
        if first_file_write == True:
            myFile.write(f"#include \"CBin-NN.h\"\n\n")
            myFile.write(f"int buffer1[N_BUFFER1]\n")
            myFile.write(f"int buffer2[N_BUFFER1]\n")
            myFile.write(f"float classification[N_CLASSES]\n\n")
            myFile.write(f"int bnn_main()\n")
            stri = '{'
            myFile.write(stri)
            myFile.write(f"\n")

        if layer_name == 'BMaxPool2D':
            myFile.write(f"\t{layer_name}({out_buffer}, {in_ch}, {in_dim}, {out_dim}, {pool_ker_size}, {pool_stride}, {pool_padding}, {in_buffer});\n")
        elif layer_name == 'BMaxPool2D_Optimized':
            myFile.write(f"\t{layer_name}({out_buffer}, {in_ch}, {in_dim}, {out_dim}, {pool_ker_size}, {pool_stride}, {pool_padding}, {in_buffer});\n")
        myFile.close()

    elif layer_name.startswith('Global_Max_Pooling'):
        in_dim = 'POOL' + str(layer_idx) + '_IN_DIM'
        out_dim = 'POOL' + str(layer_idx) + '_OUT_DIM'
        in_ch = 'POOL' + str(layer_idx) + '_CH'
        pool_ker_size = 'POOL' + str(layer_idx) + '_KER_SIZE'
        pool_stride = 'POOL' + str(layer_idx) + '_STRIDE'
        pool_padding = 'POOL' + str(layer_idx) + '_PADDING'

        myFile.write(f"#define {pool_ker_size} {in_shape[1]}\n")
        myFile.write(f"#define {pool_stride} {stride}\n")
        myFile.write(f"#define {pool_padding} {padding}\n")
        myFile.write(f"#define {in_dim} {in_shape[1]}\n")
        myFile.write(f"#define {out_dim} {1}\n")
        myFile.write(f"#define {in_ch} {in_shape[3]}\n")
        myFile.write(f"\n")
        myFile.close()

    elif layer_name.startswith('classification_layer'):
        myFile.write(f"#define N_CLASSES {out_shape}\n")
    
    elif layer_name.startswith('activation_size'):
        myFile.write(f"#define N_BUFFER1 {in_shape//32}\n")
        myFile.write(f"#define N_BUFFER2 {out_shape//32}\n")
    
    myFile.close()
