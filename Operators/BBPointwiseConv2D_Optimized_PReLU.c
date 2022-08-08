#include "CBin-NN.h"       

void BBPointwiseConv2D_Optimized_PReLU(int out[], 
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
               float shift[],             
               int input[])
{
    int i, j, k, m, n, l, N, P, in_row, in_col, input_idx, weight_idx, out_idx, in_ch_pad, out_ch_pad, LU_factor;
    int pop_count[32] = {0};
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

    int next_filter = in_ch_pad>>5 * ker_size * ker_size;
    
    for (i = 0; i < out_ch_pad; i++)
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
                /* Check padding */
                in_row = stride * j + m - padding;
                in_col = stride * k + n - padding;
                if (in_row >= 0 && in_col >= 0 && in_row < in_dim && in_col < in_dim)
                {
                    for (l = 0; l < in_ch_pad>>5; l++)
                    {
                        input_idx = (in_row * in_dim + in_col) * in_ch_pad/32 + l;

                        sum[0] = __builtin_popcount(weight[filters[0] + l]^input[input_idx]);
                        pop_count[0] += N - (sum[0]<<1);

                        sum[1] = __builtin_popcount(weight[filters[1] + l]^input[input_idx]);
                        pop_count[1] += N - (sum[1]<<1);

                        sum[2] = __builtin_popcount(weight[filters[2] + l]^input[input_idx]);
                        pop_count[2] += N - (sum[2]<<1);

                        sum[3] = __builtin_popcount(weight[filters[3] + l]^input[input_idx]);
                        pop_count[3] += N - (sum[3]<<1);

                        sum[4] = __builtin_popcount(weight[filters[4] + l]^input[input_idx]);
                        pop_count[4] += N - (sum[4]<<1);

                        sum[5] = __builtin_popcount(weight[filters[5] + l]^input[input_idx]);
                        pop_count[5] += N - (sum[5]<<1);

                        sum[6] = __builtin_popcount(weight[filters[6] + l]^input[input_idx]);
                        pop_count[6] += N - (sum[6]<<1);

                        sum[7] = __builtin_popcount(weight[filters[7] + l]^input[input_idx]);
                        pop_count[7] += N - (sum[7]<<1);

                        sum[8] = __builtin_popcount(weight[filters[8] + l]^input[input_idx]);
                        pop_count[8] += N - (sum[8]<<1);

                        sum[9] = __builtin_popcount(weight[filters[9] + l]^input[input_idx]);
                        pop_count[9] += N - (sum[9]<<1);

                        sum[10] = __builtin_popcount(weight[filters[10] + l]^input[input_idx]);
                        pop_count[10] += N - (sum[10]<<1);

                        sum[11] = __builtin_popcount(weight[filters[11] + l]^input[input_idx]);
                        pop_count[11] += N - (sum[11]<<1);

                        sum[12] = __builtin_popcount(weight[filters[12] + l]^input[input_idx]);
                        pop_count[12] += N - (sum[12]<<1);

                        sum[13] = __builtin_popcount(weight[filters[13] + l]^input[input_idx]);
                        pop_count[13] += N - (sum[13]<<1);

                        sum[14] = __builtin_popcount(weight[filters[14] + l]^input[input_idx]);
                        pop_count[14] += N - (sum[14]<<1);

                        sum[15] = __builtin_popcount(weight[filters[15] + l]^input[input_idx]);
                        pop_count[15] += N - (sum[15]<<1);

                        sum[16] = __builtin_popcount(weight[filters[16] + l]^input[input_idx]);
                        pop_count[16] += N - (sum[16]<<1);

                        sum[17] = __builtin_popcount(weight[filters[17] + l]^input[input_idx]);
                        pop_count[17] += N - (sum[17]<<1);

                        sum[18] = __builtin_popcount(weight[filters[18] + l]^input[input_idx]);
                        pop_count[18] += N - (sum[18]<<1);

                        sum[19] = __builtin_popcount(weight[filters[19] + l]^input[input_idx]);
                        pop_count[19] += N - (sum[19]<<1);

                        sum[20] = __builtin_popcount(weight[filters[20] + l]^input[input_idx]);
                        pop_count[20] += N - (sum[20]<<1);

                        sum[21] = __builtin_popcount(weight[filters[21] + l]^input[input_idx]);
                        pop_count[21] += N - (sum[21]<<1);

                        sum[22] = __builtin_popcount(weight[filters[22] + l]^input[input_idx]);
                        pop_count[22] += N - (sum[22]<<1);

                        sum[23] = __builtin_popcount(weight[filters[23] + l]^input[input_idx]);
                        pop_count[23] += N - (sum[23]<<1);

                        sum[24] = __builtin_popcount(weight[filters[24] + l]^input[input_idx]);
                        pop_count[24] += N - (sum[24]<<1);

                        sum[25] = __builtin_popcount(weight[filters[25] + l]^input[input_idx]);
                        pop_count[25] += N - (sum[25]<<1);

                        sum[26] = __builtin_popcount(weight[filters[26] + l]^input[input_idx]);
                        pop_count[26] += N - (sum[26]<<1);

                        sum[27] = __builtin_popcount(weight[filters[27] + l]^input[input_idx]);
                        pop_count[27] += N - (sum[27]<<1);

                        sum[28] = __builtin_popcount(weight[filters[28] + l]^input[input_idx]);
                        pop_count[28] += N - (sum[28]<<1);

                        sum[29] = __builtin_popcount(weight[filters[29] + l]^input[input_idx]);
                        pop_count[29] += N - (sum[29]<<1);

                        sum[30] = __builtin_popcount(weight[filters[30] + l]^input[input_idx]);
                        pop_count[30] += N - (sum[30]<<1);

                        sum[31] = __builtin_popcount(weight[filters[31] + l]^input[input_idx]);
                        pop_count[31] += N - (sum[31]<<1);
                    }
                }
                for(int p = 0; p < P; p++)
                {
                    /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
                    // pop_count[p] += bias[LU_factor+p];
                    conv_out = pop_count[p];
                    if(pop_count[p] <= 0)
					{
                		conv_out = pop_count[p] * shift[LU_factor+p];
					}
                    /* Batch Normalization Fusion*/
                    conv_out = conv_out + bn_wt[LU_factor+p];
                    out_idx = (LU_factor+p + (j * out_dim + k) * out_ch_pad);
                    /* Applying the sign function to the output activations */
                    conv_out >= 0 ? SET_BIT(out[out_idx>>5], out_idx&31) : CLEAR_BIT(out[out_idx>>5], out_idx&31);
                }
                memset(pop_count, 0, sizeof pop_count);
            }
        }
    }
}