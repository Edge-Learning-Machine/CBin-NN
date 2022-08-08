// Uncomment next line when using an IDE and in main.c (in the used IDE) add bnn_main() inside their main function;
// #include "main.h"
#include <stdio.h>
#include <stdbool.h>
#include "bnn_params.h"
#include "Test_image.h"
#include <stdint.h>
#include <cstring>
#include <math.h>

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))
#define SET_BIT(var,pos) ((var) |= (1<<(pos)))         
#define CLEAR_BIT(var,pos) ((var) &= ~(1<<(pos)))  
#define SIGN(x,var,pos) ((x < 0) ? CLEAR_BIT(var,pos) : SET_BIT(var,pos))

void QBConv2D(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_alpha1[], float bn_alpha2[], uint8_t input[]);
void QBConv2D_Optimized(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_alpha1[], float bn_alpha2[], uint8_t input[]);
void QBConv2D_Optimized_PReLU(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_alpha1[], float bn_alpha2[], float shift[], uint8_t input[]);
void QQConv2D(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int8_t weight[], float bn_alpha1[], float bn_alpha2[], uint8_t input[]);
void QQConv2D_Optimized(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int8_t weight[], float bn_alpha1[], float bn_alpha2[], uint8_t input[]);
void QQConv2D_Optimized_PReLU(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int8_t weight[], float bn_alpha1[], float bn_alpha2[], float shift[], uint8_t input[]);
void BBConv2D(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_wt[], int input[]);
void BBConv2D_Optimized(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_wt[], int input[]);
void BBConv2D_Optimized_PReLU(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_wt[], float shift[], int input[]);
void BBPointwiseConv2D(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_wt[], int input[]);
void BBPointwiseConv2D_Optimized(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_wt[], int input[]);
void BBPointwiseConv2D_Optimized_PReLU(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int weight[], float bn_wt[], float shift[], int input[]);
void BMaxPool2D(int out[], int in_ch, int in_dim, int out_dim, int ker_size, int stride, int padding, int input[]);
void BMaxPool2D_Optimized(int out[], int in_ch, int in_dim, int out_dim, int ker_size, int stride, int padding, int input[]);
void BBFC(int out[], int out_dim, int in_dim, int bias[], int weight[], float bn_wt[], int input[]);
void BBFC_Optimized(int out[], int out_dim, int in_dim, int bias[], int weight[], float bn_wt[], int input[]);
void BBFC_Optimized_PReLU(int out[], int out_dim, int in_dim, int bias[], int weight[], float bn_wt[], float shift[], int input[]);
void BBQFC(float out[], int out_dim, int in_dim, int bias[], int weight[], float bn_wt[], int input[]);
void BBQFC_Optimized(float out[], int out_dim, int in_dim, int bias[], int weight[], float bn_wt[], int input[]);
void BBQFC_Optimized_PReLU(float out[], int out_dim, int in_dim, int bias[], int weight[], float bn_wt[], float shift[], int input[]);

// TBD
// void QQConv2D_Fused_BN(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int8_t weight[], uint8_t input[]);
// void QQConv2D_Fused_BN_Optimized(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int8_t weight[], uint8_t input[]);
// void QQConv2D_Fused_BN_Optimized_PReLU(int out[], int out_ch, int out_dim, int in_ch, int in_dim, int ker_size, int stride, int padding, int bias[], int8_t weight[], float shift[], uint8_t input[]);
