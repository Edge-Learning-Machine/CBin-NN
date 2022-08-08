#include "CBin-NN.h"       

void QQConv2D_Optimized(int out[], 
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
    int i, j, k, m, n, P, LU_factor, input_idx, weight_idx, in_row, in_col, out_idx, out_ch_pad;
    int filters[32] = {0};
    int sum[32] = {0};
    int next = 0;
    float conv_out;

    /* Channel padding to a mulitple of 32 */
    if(out_ch % 32 != 0)
    {
        out_ch_pad = ceil((float)(out_ch)/32) * 32;
        if(out_ch<32) P = out_ch; else P = 32;
    }  
    else 
    {
        out_ch_pad = out_ch;
        P = 32;
    }

    int next_filter = in_ch * ker_size * ker_size;

    for (i = 0; i < out_ch_pad>>5; i++)
    {
        LU_factor = i<<5; 
        filters[0] = filters[31] + next;
        filters[1] = filters[0] + next_filter;
        filters[2] = filters[1] + next_filter;
        filters[3] = filters[2] + next_filter;
        filters[4] = filters[3] + next_filter;
        filters[5] = filters[4] + next_filter;
        filters[6] = filters[5] + next_filter;
        filters[7] = filters[6] + next_filter;
        filters[8] = filters[7] + next_filter;
        filters[9] = filters[8] + next_filter;
        filters[10] = filters[9] + next_filter;
        filters[11] = filters[10] + next_filter;
        filters[12] = filters[11] + next_filter;
        filters[13] = filters[12] + next_filter;
        filters[14] = filters[13] + next_filter;
        filters[15] = filters[14] + next_filter;
        filters[16] = filters[15] + next_filter;
        filters[17] = filters[16] + next_filter;
        filters[18] = filters[17] + next_filter;
        filters[19] = filters[18] + next_filter;
        filters[20] = filters[19] + next_filter;
        filters[21] = filters[20] + next_filter;
        filters[22] = filters[21] + next_filter;
        filters[23] = filters[22] + next_filter;
        filters[24] = filters[23] + next_filter;
        filters[25] = filters[24] + next_filter;
        filters[26] = filters[25] + next_filter;
        filters[27] = filters[26] + next_filter;
        filters[28] = filters[27] + next_filter;
        filters[29] = filters[28] + next_filter;
        filters[30] = filters[29] + next_filter;
        filters[31] = filters[30] + next_filter;        
        next = next_filter;

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
                            weight_idx = (m * ker_size + n) * in_ch;
                            input_idx = (in_row * in_dim + in_col) * in_ch;

                            sum[0] += weight[(filters[0] + weight_idx)] * input[input_idx];
                            sum[0] += weight[(filters[0] + weight_idx+1)] * input[input_idx+1];
                            sum[0] += weight[(filters[0] + weight_idx+2)] * input[input_idx+2];

                            sum[1] += weight[(filters[1] + weight_idx)] * input[input_idx];
                            sum[1] += weight[(filters[1] + weight_idx+1)] * input[input_idx+1];
                            sum[1] += weight[(filters[1] + weight_idx+2)] * input[input_idx+2];

                            sum[2] += weight[(filters[2] + weight_idx)] * input[input_idx];
                            sum[2] += weight[(filters[2] + weight_idx+1)] * input[input_idx+1];
                            sum[2] += weight[(filters[2] + weight_idx+2)] * input[input_idx+2];

                            sum[3] += weight[(filters[3] + weight_idx)] * input[input_idx];
                            sum[3] += weight[(filters[3] + weight_idx+1)] * input[input_idx+1];
                            sum[3] += weight[(filters[3] + weight_idx+2)] * input[input_idx+2];

                            sum[4] += weight[(filters[4] + weight_idx)] * input[input_idx];
                            sum[4] += weight[(filters[4] + weight_idx+1)] * input[input_idx+1];
                            sum[4] += weight[(filters[4] + weight_idx+2)] * input[input_idx+2];

                            sum[5] += weight[(filters[5] + weight_idx)] * input[input_idx];
                            sum[5] += weight[(filters[5] + weight_idx+1)] * input[input_idx+1];
                            sum[5] += weight[(filters[5] + weight_idx+2)] * input[input_idx+2];

                            sum[6] += weight[(filters[6] + weight_idx)] * input[input_idx];
                            sum[6] += weight[(filters[6] + weight_idx+1)] * input[input_idx+1];
                            sum[6] += weight[(filters[6] + weight_idx+2)] * input[input_idx+2];

                            sum[7] += weight[(filters[7] + weight_idx)] * input[input_idx];
                            sum[7] += weight[(filters[7] + weight_idx+1)] * input[input_idx+1];
                            sum[7] += weight[(filters[7] + weight_idx+2)] * input[input_idx+2];

                            sum[8] += weight[(filters[8] + weight_idx)] * input[input_idx];
                            sum[8] += weight[(filters[8] + weight_idx+1)] * input[input_idx+1];
                            sum[8] += weight[(filters[8] + weight_idx+2)] * input[input_idx+2];

                            sum[9] += weight[(filters[9] + weight_idx)] * input[input_idx];
                            sum[9] += weight[(filters[9] + weight_idx+1)] * input[input_idx+1];
                            sum[9] += weight[(filters[9] + weight_idx+2)] * input[input_idx+2];

                            sum[10] += weight[(filters[10] + weight_idx)] * input[input_idx];
                            sum[10] += weight[(filters[10] + weight_idx+1)] * input[input_idx+1];
                            sum[10] += weight[(filters[10] + weight_idx+2)] * input[input_idx+2];

                            sum[11] += weight[(filters[11] + weight_idx)] * input[input_idx];
                            sum[11] += weight[(filters[11] + weight_idx+1)] * input[input_idx+1];
                            sum[11] += weight[(filters[11] + weight_idx+2)] * input[input_idx+2];

                            sum[12] += weight[(filters[12] + weight_idx)] * input[input_idx];
                            sum[12] += weight[(filters[12] + weight_idx+1)] * input[input_idx+1];
                            sum[12] += weight[(filters[12] + weight_idx+2)] * input[input_idx+2];

                            sum[13] += weight[(filters[13] + weight_idx)] * input[input_idx];
                            sum[13] += weight[(filters[13] + weight_idx+1)] * input[input_idx+1];
                            sum[13] += weight[(filters[13] + weight_idx+2)] * input[input_idx+2];

                            sum[14] += weight[(filters[14] + weight_idx)] * input[input_idx];
                            sum[14] += weight[(filters[14] + weight_idx+1)] * input[input_idx+1];
                            sum[14] += weight[(filters[14] + weight_idx+2)] * input[input_idx+2];

                            sum[15] += weight[(filters[15] + weight_idx)] * input[input_idx];
                            sum[15] += weight[(filters[15] + weight_idx+1)] * input[input_idx+1];
                            sum[15] += weight[(filters[15] + weight_idx+2)] * input[input_idx+2];

                            sum[16] += weight[(filters[16] + weight_idx)] * input[input_idx];
                            sum[16] += weight[(filters[16] + weight_idx+1)] * input[input_idx+1];
                            sum[16] += weight[(filters[16] + weight_idx+2)] * input[input_idx+2];

                            sum[17] += weight[(filters[17] + weight_idx)] * input[input_idx];
                            sum[17] += weight[(filters[17] + weight_idx+1)] * input[input_idx+1];
                            sum[17] += weight[(filters[17] + weight_idx+2)] * input[input_idx+2];

                            sum[18] += weight[(filters[18] + weight_idx)] * input[input_idx];
                            sum[18] += weight[(filters[18] + weight_idx+1)] * input[input_idx+1];
                            sum[18] += weight[(filters[18] + weight_idx+2)] * input[input_idx+2];

                            sum[19] += weight[(filters[19] + weight_idx)] * input[input_idx];
                            sum[19] += weight[(filters[19] + weight_idx+1)] * input[input_idx+1];
                            sum[19] += weight[(filters[19] + weight_idx+2)] * input[input_idx+2];

                            sum[20] += weight[(filters[20] + weight_idx)] * input[input_idx];
                            sum[20] += weight[(filters[20] + weight_idx+1)] * input[input_idx+1];
                            sum[20] += weight[(filters[20] + weight_idx+2)] * input[input_idx+2];

                            sum[21] += weight[(filters[21] + weight_idx)] * input[input_idx];
                            sum[21] += weight[(filters[21] + weight_idx+1)] * input[input_idx+1];
                            sum[21] += weight[(filters[21] + weight_idx+2)] * input[input_idx+2];

                            sum[22] += weight[(filters[22] + weight_idx)] * input[input_idx];
                            sum[22] += weight[(filters[22] + weight_idx+1)] * input[input_idx+1];
                            sum[22] += weight[(filters[22] + weight_idx+2)] * input[input_idx+2];

                            sum[23] += weight[(filters[23] + weight_idx)] * input[input_idx];
                            sum[23] += weight[(filters[23] + weight_idx+1)] * input[input_idx+1];
                            sum[23] += weight[(filters[23] + weight_idx+2)] * input[input_idx+2];

                            sum[24] += weight[(filters[24] + weight_idx)] * input[input_idx];
                            sum[24] += weight[(filters[24] + weight_idx+1)] * input[input_idx+1];
                            sum[24] += weight[(filters[24] + weight_idx+2)] * input[input_idx+2];

                            sum[25] += weight[(filters[25] + weight_idx)] * input[input_idx];
                            sum[25] += weight[(filters[25] + weight_idx+1)] * input[input_idx+1];
                            sum[25] += weight[(filters[25] + weight_idx+2)] * input[input_idx+2];

                            sum[26] += weight[(filters[26] + weight_idx)] * input[input_idx];
                            sum[26] += weight[(filters[26] + weight_idx+1)] * input[input_idx+1];
                            sum[26] += weight[(filters[26] + weight_idx+2)] * input[input_idx+2];

                            sum[27] += weight[(filters[27] + weight_idx)] * input[input_idx];
                            sum[27] += weight[(filters[27] + weight_idx+1)] * input[input_idx+1];
                            sum[27] += weight[(filters[27] + weight_idx+2)] * input[input_idx+2];

                            sum[28] += weight[(filters[28] + weight_idx)] * input[input_idx];
                            sum[28] += weight[(filters[28] + weight_idx+1)] * input[input_idx+1];
                            sum[28] += weight[(filters[28] + weight_idx+2)] * input[input_idx+2];

                            sum[29] += weight[(filters[29] + weight_idx)] * input[input_idx];
                            sum[29] += weight[(filters[29] + weight_idx+1)] * input[input_idx+1];
                            sum[29] += weight[(filters[29] + weight_idx+2)] * input[input_idx+2];

                            sum[30] += weight[(filters[30] + weight_idx)] * input[input_idx];
                            sum[30] += weight[(filters[30] + weight_idx+1)] * input[input_idx+1];
                            sum[30] += weight[(filters[30] + weight_idx+2)] * input[input_idx+2];

                            sum[31] += weight[(filters[31] + weight_idx)] * input[input_idx];
                            sum[31] += weight[(filters[31] + weight_idx+1)] * input[input_idx+1];
                            sum[31] += weight[(filters[31] + weight_idx+2)] * input[input_idx+2];
                        }
                    }
                }
                for(int p = 0; p < P; p++)
                {
                    /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
                    // sum[p] += bias[LU_factor+p];
                    /* Batch Normalization Fusion*/
                    conv_out = (sum[p] - bn_alpha1[LU_factor+p]) * bn_alpha2[LU_factor+p];
                    out_idx = (LU_factor + p + (j * out_dim + k) * out_ch_pad);
                    /* Applying the sign function to the output activations */
                    conv_out >= 0 ? SET_BIT(out[out_idx>>5], out_idx&31) : CLEAR_BIT(out[out_idx>>5], out_idx&31);
                }
                memset(sum, 0, sizeof sum);
            }
        }
    }
}