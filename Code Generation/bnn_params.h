#define N_CONV1_WT 75
#define CONV1_IN_CH 3
#define CONV1_IN_DIM 32
#define CONV1_KER_SIZE 5
#define CONV1_OUT_CH 32
#define CONV1_OUT_DIM 32
#define CONV1_STRIDE 1
#define CONV1_PADDING 2
extern int conv1_wt[N_CONV1_WT];
#define N_SHIFT1 2400
extern float shift1[N_SHIFT1];

#define N_BN1_ALPHA1 2
#define N_BN1_ALPHA2 2
extern float bn1_alpha1[N_BN1_ALPHA1];
extern float bn1_alpha2[N_BN1_ALPHA2];

#define POOL1_KER_SIZE 2
#define POOL1_STRIDE 2
#define POOL1_PADDING 0
#define POOL1_IN_DIM 32
#define POOL1_OUT_DIM 16
#define POOL1_CH 32

#define N_CONV2_WT 800
#define CONV2_IN_CH 32
#define CONV2_IN_DIM 16
#define CONV2_KER_SIZE 5
#define CONV2_OUT_CH 32
#define CONV2_OUT_DIM 16
#define CONV2_STRIDE 1
#define CONV2_PADDING 2
extern int conv2_wt[N_CONV2_WT];
#define N_SHIFT2 25600
extern float shift2[N_SHIFT2];

#define N_BN2_WT 32
extern float bn2_wt[N_BN2_WT];

#define POOL2_KER_SIZE 2
#define POOL2_STRIDE 2
#define POOL2_PADDING 0
#define POOL2_IN_DIM 16
#define POOL2_OUT_DIM 8
#define POOL2_CH 32

#define N_CONV3_WT 1600
#define CONV3_IN_CH 32
#define CONV3_IN_DIM 8
#define CONV3_KER_SIZE 5
#define CONV3_OUT_CH 64
#define CONV3_OUT_DIM 8
#define CONV3_STRIDE 1
#define CONV3_PADDING 2
extern int conv3_wt[N_CONV3_WT];
#define N_SHIFT3 51200
extern float shift3[N_SHIFT3];

#define N_BN3_WT 64
extern float bn3_wt[N_BN3_WT];

#define POOL3_KER_SIZE 2
#define POOL3_STRIDE 2
#define POOL3_PADDING 0
#define POOL3_IN_DIM 8
#define POOL3_OUT_DIM 4
#define POOL3_CH 64

#define N_FC1_WT 320
#define FC1_IN_DIM 10
#define FC1_OUT_DIM 1024
extern int fc1_wt[N_FC1_WT];
#define N_SHIFT4 10240
extern float shift4[N_SHIFT4];

#define N_BN4_WT 10
extern float bn4_wt[N_BN4_WT];

#define N_CLASSES 10
#define N_BUFFER1 1024
#define N_BUFFER2 256
