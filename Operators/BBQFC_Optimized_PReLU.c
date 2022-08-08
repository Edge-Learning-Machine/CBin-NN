#include "CBin-NN.h"

void BBQFC_Optimized_PReLU(float out[], 
         int out_dim, 
         int in_dim, 
         int bias[],
         int weight[], 
         float bn_wt[],
         float shift[],
         int input[])
{
    int i, j, weight_idx, sum;
    float pop_count = 0;

    for (i = 0; i < out_dim; i++)
    {
        for (j = 0; j < in_dim/32; j++)
        {
            weight_idx = (i * in_dim/32 + j);
            sum = __builtin_popcount(input[j] ^ weight[weight_idx]);
            pop_count += 32 - 2*sum;
        }
        /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
        // pop_count += bias[i];
        pop_count += bn_wt[i];
        out[i] = pop_count;
        pop_count = 0;
    }
}