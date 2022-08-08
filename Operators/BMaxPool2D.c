#include "CBin-NN.h"

void BMaxPool2D(int out[], 
               int in_ch, 
               int in_dim, 
               int out_dim, 
               int ker_size,
               int stride,
               int padding,
               int input[])
{    
    int i, j, k, m, n, input_idx, out_idx;
    
    for (i = 0; i < in_ch; i++)
    {
        for (j = 0; j < out_dim; j++)
        {
            for (k = 0; k < out_dim; k++)
            {
                int max = 0;
                for (m = j * stride - padding; m < j * stride - padding + ker_size; m++)
                {
                    for (n = k * stride - padding; n < k * stride - padding + ker_size; n++)
                    {
                        if (m >= 0 && n >= 0 && m < in_dim && n < in_dim)
                        {
                            input_idx = (i + in_ch * (n + m * in_dim));
                            max = (bool)CHECK_BIT(input[input_idx>>5], input_idx&31) | max;
                        }
                    }
                }
                out_idx = (i + in_ch * (k + j * out_dim));
                max == 1 ? SET_BIT(out[out_idx>>5], out_idx&31) : CLEAR_BIT(out[out_idx>>5], out_idx&31);
            }
        }
    }
}