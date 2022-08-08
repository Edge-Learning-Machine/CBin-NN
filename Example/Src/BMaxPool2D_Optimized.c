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
    int i, j, k, m, n;
    
    for (i = 0; i < in_ch>>5; i++)
    {
        for (j = 0; j < out_dim; j++)
        {
            for (k = 0; k < out_dim; k++)
            {
                int max = 0x00000000;
                for (m = j * stride - padding; m < j * stride - padding + ker_size; m++)
                {
                    for (n = k * stride - padding; n < k * stride - padding + ker_size; n++)
                    {
                        if (m >= 0 && n >= 0 && m < in_dim && n < in_dim)
                        {
                            max = input[(i + in_ch>>5 * (n + m * in_dim))] | max;
                        }
                    }
                }
                out[(i + in_ch/32 * (k + j * out_dim))] = max;
            }
        }
    }
}