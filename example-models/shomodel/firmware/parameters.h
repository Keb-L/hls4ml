#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_large.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_conv.h"
#include "nnet_utils/nnet_conv_large.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_large.h"
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_common.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 96
#define N_INPUT_2_1 2
#define N_OUTPUTS_2 82
#define N_FILT_2 129
#define N_OUTPUTS_4 78
#define N_FILT_4 65
#define N_LAYER_6 512
#define N_LAYER_8 512
#define N_LAYER_10 128
#define N_LAYER_12 128
#define N_LAYER_14 128
#define N_LAYER_16 128
#define N_LAYER_18 4

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_uint<16> model_weightdefault_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<16,6> layer18_t;
typedef ap_fixed<16,6> result_t;
typedef ap_uint<27> model_bigdefault_t;

