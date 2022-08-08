from asyncio.windows_events import NULL
from unittest import skip
import tensorflow as tf
from keras.models import load_model
import larq as lq
import numpy as np
import math
import torch
import save_params as sp
import utils as utils

# User defined params:
model_name = 'lenet_PReLU.h5'
loop_unrolling = True
# Faster kernel during inference 
optimized_Max_Pool = True
# Faster kernel during inference 
optimized_FC = True

# Model saving params:
first_file_write = True
BN_fused = False
Conv_idx = 1
FC_idx = 1
MaxPool_idx = 1
BN_idx = 1
PReLU_idx = 1
in_buffer = 'buffer2'
out_buffer = 'buffer1'

model = load_model(model_name, compile = False)

list_of_activation_size = []
def max_activation_size(list_of_activation_size):
    max_value = list_of_activation_size[0]
    for i in list_of_activation_size:
        if i > max_value:
            max_value = i
    list_of_activation_size.remove(max_value)
    return max_value

def quantize_weights(weights):
    min_wt = weights.min()
    max_wt = weights.max()
    int_bits = int(np.ceil(np.log2(max(abs(min_wt),abs(max_wt))))) 
    frac_bits = 7-abs(int_bits) 
    weights = np.round(weights*(2**frac_bits)).astype(int)
    return weights

####################### ADD POINTWISE CONVOLUTION ####################
with lq.context.quantized_scope(True):
    for layer_idx, layer in enumerate(model.layers):
        # Extract convolutional layer params
        if 'conv' in layer.name:
            in_shape = np.array(layer.input.shape)
            out_shape = np.array(layer.output.shape)
            list_of_activation_size.append(out_shape[1]*out_shape[2]*out_shape[3])
            ker_size = layer.kernel_size[0]
            stride = layer.strides[0]
            if layer.padding == 'valid':
                padding = 0
            else:
                padding = int((ker_size - stride)/2)

            weights = layer.get_weights()[0]

            # Binarized weights
            if weights.max() == 1.0 and weights.min() == -1.0:
                if layer_idx == 0 and loop_unrolling == True and 'p_re_lu' in model.layers[layer_idx+1].name and ker_size != 1:
                    layer_name = 'QBConv2D_Optimized_PReLU'
                elif layer_idx == 0 and loop_unrolling == True and ker_size != 1:
                    layer_name = 'QBConv2D_Optimized'
                elif loop_unrolling == False and layer_idx == 0 and ker_size != 1:
                    layer_name = 'QBConv2D'
                elif loop_unrolling == True and 'p_re_lu' in model.layers[layer_idx+1].name and ker_size != 1:
                    layer_name = 'BBConv2D_Optimized_PReLU'
                elif loop_unrolling == True and ker_size != 1:
                    layer_name = 'BBConv2D_Optimized'
                elif loop_unrolling == False and ker_size != 1:
                    layer_name = 'BBConv2D'
                elif loop_unrolling == True and 'p_re_lu' in model.layers[layer_idx+1].name and ker_size == 1:
                    layer_name = 'BBPointwiseConv2D_Optimized_PReLU'
                elif loop_unrolling == True and ker_size == 1:
                    layer_name = 'BBPointwiseConv2D_Optimized'
                elif loop_unrolling == False and ker_size == 1:
                    layer_name = 'BBPointwiseConv2D'
                
                weights = weights.transpose(3,0,1,2)
                weights = weights.astype(int)
                weights = np.where(weights==-1, 0, weights)
                bias = None

                if layer_idx != 0 and weights.shape[3]%32 !=0:
                    padding_value = int(np.ceil(weights.shape[3]/32)*32)
                    weights = np.pad(weights, ((0,0),(0,0),(0,0),(0,padding_value-weights.shape[3])), 'constant', constant_values=0)
                
                if loop_unrolling == True:
                    if weights.shape[0]%32 !=0:
                        padding_value = int(np.ceil(weights.shape[0]/32)*32)
                        weights = np.pad(weights, ((0,padding_value-weights.shape[0]),(0,0),(0,0),(0,0)), 'constant', constant_values=1)
            
            # Quantized weights for the first layer 
            else:
                if layer_idx == 0 and loop_unrolling == True and 'p_re_lu' in model.layers[layer_idx+1].name and ker_size != 1:
                    layer_name = 'QQConv2D_Optimized_PReLU'
                elif layer_idx == 0 and loop_unrolling == True and ker_size != 1:
                    layer_name = 'QQConv2D_Optimized'
                elif loop_unrolling == False and layer_idx == 0 and ker_size != 1:
                    layer_name = 'QQConv2D'

                weights = weights.transpose(3,0,1,2)
                # w_new = utils.quantize_tensor(weights, 128, -1.5)
                weights = quantize_weights(weights)
                if layer.bias != None:
                    bias = layer.get_weights()[1]
                    bias = quantize_weights(bias)
                else:
                    bias = None
                    
            sp.save_bnn_params(weights, bias, layer_name, Conv_idx, in_shape, out_shape, ker_size, stride, padding, first_file_write, in_buffer, out_buffer, BN_idx)
            Conv_idx += 1
            first_file_write = False
            if in_buffer == 'buffer1':
                in_buffer = 'buffer2'
            else: 
                in_buffer = 'buffer1'
            if out_buffer == 'buffer1':
                out_buffer = 'buffer2'
            else: 
                out_buffer = 'buffer1'

        elif 'dense' in layer.name:
            if optimized_FC == True and 'p_re_lu' in model.layers[layer_idx+1].name and 'activation' in model.layers[layer_idx+3].name:
                layer_name = 'BBQFC_Optimized_PReLU'
            elif optimized_FC == True and 'activation' in model.layers[layer_idx+2].name:
                layer_name = 'BBQFC_Optimized'
            elif optimized_FC == False and 'activation' in model.layers[layer_idx+2].name:
                layer_name = 'BBQFC'
            elif optimized_FC == True and 'p_re_lu' in model.layers[layer_idx+1].name:
                layer_name = 'BBFC_Optimized_PReLU'
            elif optimized_FC == True:
                layer_name = 'BBFC_Optimized'
            elif optimized_FC == False:
                layer_name = 'BBFC'
            
            weights = layer.get_weights()[0]
            in_dim = weights.shape[1]
            out_dim =  weights.shape[0]
            # Binarized weights
            if weights.max() == 1.0 and weights.min() == -1.0:
                weights = np.transpose(weights,(1,0))
                weights = weights.astype(int)
                weights = np.where(weights==-1, 0, weights)
                bias = None

            # To be tested
            # Quantized weights for the last layer 
            else:
                layer_name = 'quantized_fc'
                weights = np.transpose(weights,(1,0))
                weights = quantize_weights(weights)
                if layer.bias != None:
                    bias = layer.get_weights()[1]
                    bias = quantize_weights(bias)
                else:
                    bias = None

            sp.save_bnn_params(weights, bias, layer_name, FC_idx, in_dim, out_dim, NULL, NULL, NULL, first_file_write, in_buffer, out_buffer, BN_idx)
            FC_idx += 1
            first_file_write = False
            if in_buffer == 'buffer1':
                in_buffer = 'buffer2'
            else: 
                in_buffer = 'buffer1'
            if out_buffer == 'buffer1':
                out_buffer = 'buffer2'
            else: 
                out_buffer = 'buffer1'

        elif 'batch' in layer.name:
            gamma = layer.get_weights()[0]
            beta = layer.get_weights()[1]
            mean = layer.get_weights()[2]
            std = layer.get_weights()[3]
            epsilon = layer.epsilon
            dim = gamma.size
            # First BN layer is handled separetly, for more information refer to the paper
            if BN_idx == 1:
                layer_name = 'quantized_BN'
                alpha1 = np.zeros(gamma.size)
                alpha2 = np.zeros(gamma.size)
                for i in range(gamma.size):
                    alpha1[i] = mean[i] - (np.sqrt(std[i]+epsilon)*beta[i])/gamma[i]
                    alpha2[i] = gamma[i]/np.sqrt(std[i]+epsilon)
                if loop_unrolling == True:
                    if alpha1.size %32 !=0 and alpha1.size>32:
                        padding_value = int(np.ceil(alpha1.size/32)*32)
                        alpha1 = np.pad(alpha1, (0,padding_value-alpha1.size), 'constant')
                        alpha2 = np.pad(alpha2, (0,padding_value-alpha2.size), 'constant')
                alpha = np.array([alpha1, alpha2])
                sp.save_bnn_params(alpha, NULL, layer_name, BN_idx, dim, NULL, NULL, NULL, NULL, first_file_write, NULL, NULL, NULL)
                BN_idx = BN_idx + 1
            # All other BN layers are handled as follows, for more information refer to the paper
            else:
                layer_name = 'binary_BN'
                integer_bias = np.zeros(gamma.shape[0])
                for i in range(gamma.size):
                    integer_bias[i] = -mean[i] + (math.sqrt((std[i])+epsilon)/gamma[i])*beta[i]
                if loop_unrolling == True and 'dense' not in model.layers[layer_idx-1].name:
                    if integer_bias.size %32 !=0 and integer_bias.size>32:
                        padding_value = int(np.ceil(integer_bias.size/32)*32)
                        integer_bias = np.pad(integer_bias, (0,padding_value-integer_bias.size), 'constant')
                sp.save_bnn_params(integer_bias, NULL, layer_name, BN_idx, dim, NULL, NULL, NULL, NULL, first_file_write, NULL, NULL, NULL)
                BN_idx = BN_idx + 1
            first_file_write = False
                
        elif 'p_re_lu' in layer.name:
            layer_name = 'PReLU'
            PReLU_weights = layer.get_weights()
            PReLU_weights = np.array(PReLU_weights).flatten()
            dim = PReLU_weights.size
            if loop_unrolling == True:
                if PReLU_weights.size %32 !=0 and PReLU_weights.size>32:
                    padding_value = int(np.ceil(PReLU_weights.size/32)*32)
                    PReLU_weights = np.pad(PReLU_weights, (0,padding_value-PReLU_weights.size), 'constant')
            sp.save_bnn_params(weights, NULL, layer_name, PReLU_idx, dim, NULL, NULL, NULL, NULL, first_file_write, NULL, NULL, NULL)
            PReLU_idx = PReLU_idx + 1
            first_file_write = False
        
        elif 'max_pooling' in layer.name:
            if optimized_Max_Pool == True:
                layer_name = 'BMaxPool2D_Optimized'
            else:
                layer_name = 'BMaxPool2D'
            in_shape = np.array(layer.input.shape)
            out_shape = np.array(layer.output.shape)
            list_of_activation_size.append(out_shape[1]*out_shape[2]*out_shape[3])
            pool_size = layer.pool_size[0]
            stride = layer.strides[0]
            if layer.padding == 'valid':
                padding = 0
            else:
                padding = (out_shape[1] - 1) * stride + pool_size - in_shape[1]
            
            sp.save_bnn_params(NULL, NULL, layer_name, MaxPool_idx, in_shape, out_shape, pool_size, stride, padding, first_file_write, in_buffer, out_buffer, NULL)
            MaxPool_idx = MaxPool_idx + 1
            first_file_write = False
            if in_buffer == 'buffer1':
                in_buffer = 'buffer2'
            else: 
                in_buffer = 'buffer1'
            if out_buffer == 'buffer1':
                out_buffer = 'buffer2'
            else: 
                out_buffer = 'buffer1'
        
        ### To be tested 
        elif 'global_max_pooling' in layer.name:
            layer_name = 'Global_Max_Pooling'
            in_shape = np.array(layer.input.shape)
            out_shape = np.array(layer.output.shape)
            pool_size = layer.pool_size[0]
            stride = layer.strides[0]
            if layer.padding == 'valid':
                padding = 0
            else:
                padding = (out_shape[1] - 1) * stride + pool_size - in_shape[1]

            sp.save_bnn_params(NULL, NULL, layer_name, MaxPool_idx, in_shape, out_shape, pool_size, stride, padding, first_file_write, in_buffer, out_buffer, NULL)
            MaxPool_idx = MaxPool_idx + 1
            first_file_write = False
            if in_buffer == 'buffer1':
                in_buffer = 'buffer2'
            else: 
                in_buffer = 'buffer1'
            if out_buffer == 'buffer1':
                out_buffer = 'buffer2'
            else: 
                out_buffer = 'buffer1'
        
        elif 'activation' in layer.name:
            layer_name = 'classification_layer'
            n_classes = layer.output.shape[1]
            sp.save_bnn_params(NULL, NULL, layer_name, NULL, NULL, n_classes, NULL, NULL, NULL, first_file_write, NULL, NULL, NULL)
            first_file_write = False

layer_name = 'activation_size'
max_activation = max_activation_size(list_of_activation_size)
second_max_activation = max_activation_size(list_of_activation_size)
sp.save_bnn_params(NULL, NULL, layer_name, NULL, max_activation, second_max_activation, NULL, NULL, NULL, first_file_write, NULL, NULL, NULL)