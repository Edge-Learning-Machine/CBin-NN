#include "CBin-NN.h"       

void QQConv2D(int out[], 
                   int out_ch, 
                   int out_dim, 
                   int in_ch, 
                   int in_dim, 
                   int ker_size,
                   int stride,
                   int padding, 
                   int bias[], 
                   int8_t weight[],
                   float bn_alpha1[], 
                   float bn_alpha2[], 
                   uint8_t input[])
{
    int i, j, k, l, m, n, input_idx, weight_idx, in_row, in_col, out_idx, out_ch_pad;
    float conv_out = 0;

    /* Channel padding to a mulitple of 32 */
    if(out_ch % 32 != 0)
    {
        out_ch_pad = ceil((float)(out_ch)/32) * 32;
    }  
    else 
    {
        out_ch_pad = out_ch;
    }

    for (i = 0; i < out_ch; i++)
    {
        for (j = 0; j < out_dim; j++)
        {
            for (k = 0; k < out_dim; k++)
            {
                for (m = 0; m < ker_size; m++)
                {
                    for (n = 0; n < ker_size; n++)
                    {
                        /* Check padding */
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < in_dim && in_col < in_dim)
                        {
                            for (l = 0; l < in_ch; l++)
                            {
                                input_idx = (in_row * in_dim + in_col) * in_ch + l;
                                weight_idx = i * in_ch * ker_size * ker_size + (m * ker_size + n) * in_ch + l;
                                conv_out += weight[weight_idx] * input[input_idx];
                            }
                        }
                    }
                }
                /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
                // conv_out += bias[i];
                /* Batch Normalization Fusion*/
                conv_out = (conv_out - bn_alpha1[i]) * bn_alpha2[i];
                /* output buffer index */
                out_idx = (i + (j * out_dim + k) * out_ch_pad);
                /* Applying the sign function to the output activations */
                conv_out >= 0 ? SET_BIT(out[out_idx>>5], out_idx&31) : CLEAR_BIT(out[out_idx>>5], out_idx&31);
                conv_out = 0;
            }
        }
    }
}
