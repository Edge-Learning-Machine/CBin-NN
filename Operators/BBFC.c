#include "CBin-NN.h"

void BBFC(int out[], 
         int out_dim, 
         int in_dim, 
         int bias[],
         int weight[], 
         float bn_wt[],
         int input[])
{
    int i, j, weight_idx;
    float sum = 0;

    for (i = 0; i < out_dim; i++)
    {
        for (j = 0; j < in_dim; j++)
        {
            weight_idx = (i * in_dim + j);
            if(((bool)CHECK_BIT(input[j>>5], j&31) == (bool)CHECK_BIT(weight[weight_idx>>5], weight_idx&31))) sum++; else sum--;
        }

        /* Bias is usually not used in BNNs, however, in case it is used uncomment the next line of code */
        // sum += bias[i];
        sum += bn_wt[i];
        sum >= 0? SET_BIT(out[i>>5], i&31) : CLEAR_BIT(out[i>>5], i&31);
        sum = 0;
    }
}