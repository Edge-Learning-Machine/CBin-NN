#include "CBin-NN.h"       

void BBPointwiseConv2D(int out[], 
               int out_ch, 
               int out_dim, 
               int in_ch, 
               int in_dim, 
               int ker_size,
               int stride,
               int padding, 
               int bias[], 
               int weight[],
               float bn_wt[],               
               int input[])
{
    int i, j, k, m, n, l, N, in_row, in_col, input_idx, weight_idx, out_idx, in_ch_pad, out_ch_pad;
    int sum, pop_count;
    float conv_out;

    /* Channel padding to a mulitple of 32 */
    if(out_ch % 32 != 0)
    {
        out_ch_pad = ceil((float)(out_ch)/32) * 32;
    }  
    else 
    {
        out_ch_pad = out_ch;
    }

    if (in_ch % 32 != 0)
    {
        in_ch_pad = ceil((float)(in_ch)/32) * 32;
        if (in_ch < 32) N = in_ch; else N = 32;
    }
    else 
    {
        in_ch_pad = in_ch;
        N = 32;
    }
    
    for (i = 0; i < out_ch; i++)
    {    
        for (j = 0; j < out_dim; j++)
        {
            for (k = 0; k < out_dim; k++)
            {
                in_row = stride * j - padding;
                in_col = stride * k - padding;
                if (in_row >= 0 && in_col >= 0 && in_row < in_dim && in_col < in_dim)
                {
                    for (l = 0; l < in_ch_pad>>5; l++)
                    {
                        input_idx = (in_row * in_dim + in_col) * in_ch_pad>>5 + l;
                        weight_idx = i * in_ch_pad>>5 * ker_size * ker_size + l;
                        sum = __builtin_popcount(weight[weight_idx]^input[input_idx]);
                        pop_count += N - (sum<<1);
                    }
                }
                /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
                // pop_count += bias[i];
                /* Batch Normalization Fusion*/
                conv_out = pop_count + bn_wt[i];
                out_idx = (i + (j * out_dim + k) * out_ch_pad);
                /* Applying the sign function to the output activations */
                conv_out >= 0 ? SET_BIT(out[out_idx>>5], out_idx&31) : CLEAR_BIT(out[out_idx>>5], out_idx&31);
                pop_count = 0;
            }
        }
    }
}