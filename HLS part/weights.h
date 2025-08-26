#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "types.h"

extern const weight_t conv1_weight[4][1][3];
extern const weight_t conv1_bias[4];
extern const weight_t conv2_weight[8][4][3];
extern const weight_t conv2_bias[8];
extern const weight_t conv3_weight[16][8][3][3];
extern const weight_t conv3_bias[16];
extern const weight_t fc_weight[2][16];
extern const weight_t fc_bias[2];
extern const norm_param_t global_mean[6];
extern const norm_param_t global_std[6];

#endif // WEIGHTS_H
