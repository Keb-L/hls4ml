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
#include "nnet_utils/nnet_upsampling2d.h"
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_common.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_INPUT_2_1 16
#define N_INPUT_3_1 4
#define OUT_HEIGHT_2 16
#define OUT_WIDTH_2 16
#define N_FILT_2 65
#define OUT_HEIGHT_6 8
#define OUT_WIDTH_6 8
#define N_FILT_6 65
#define OUT_HEIGHT_7 8
#define OUT_WIDTH_7 8
#define N_FILT_7 129
#define OUT_HEIGHT_11 8
#define OUT_WIDTH_11 8
#define N_FILT_11 129
#define OUT_HEIGHT_15 4
#define OUT_WIDTH_15 4
#define N_FILT_15 129
#define N_LAYER_16 9

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> model_weightdefault_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<16,6> result_t;
typedef ap_uint<27> model_bigdefault_t;

//hls-fpga-machine-learning insert layer-config
struct config2_relu : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_serial;
};

struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 75;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 75;
    static const unsigned merge_factor = 1;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef model_default_t weightmult_t;
};

struct config2 : nnet::conv2d_config {
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
    static const unsigned in_height = N_INPUT_1_1;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_chan = N_INPUT_3_1-1;
    static const unsigned n_chan_in = N_INPUT_3_1;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned n_filt = N_FILT_2-1;
    static const unsigned n_filt_in = N_FILT_2;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_2;
    static const unsigned out_width = OUT_WIDTH_2;
    static const unsigned reuse_factor = 75;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef config2_mult mult_config;
    typedef config2_relu relu_config;
};

struct config6 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_filt = N_FILT_6-1;
    static const unsigned n_chan = N_FILT_2-1;
    static const unsigned n_filt_in = N_FILT_6;
    static const unsigned n_chan_in = N_FILT_2;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned out_height = OUT_HEIGHT_6;
    static const unsigned out_width = OUT_WIDTH_6;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const unsigned reuse = 72;
};

struct config7_relu : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_serial;
};

struct config7_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 72;
    static const unsigned merge_factor = 1;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef model_default_t weightmult_t;
};

struct config7 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_6;
    static const unsigned in_width = OUT_WIDTH_6;
    static const unsigned n_chan = N_FILT_6-1;
    static const unsigned n_chan_in = N_FILT_6;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_7-1;
    static const unsigned n_filt_in = N_FILT_7;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_7;
    static const unsigned out_width = OUT_WIDTH_7;
    static const unsigned reuse_factor = 72;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef config7_mult mult_config;
    typedef config7_relu relu_config;
};

struct config11_relu : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_serial;
};

struct config11_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 72;
    static const unsigned merge_factor = 1;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef model_default_t weightmult_t;
};

struct config11 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_7;
    static const unsigned in_width = OUT_WIDTH_7;
    static const unsigned n_chan = N_FILT_7-1;
    static const unsigned n_chan_in = N_FILT_7;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_11-1;
    static const unsigned n_filt_in = N_FILT_11;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_11;
    static const unsigned out_width = OUT_WIDTH_11;
    static const unsigned reuse_factor = 72;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef config11_mult mult_config;
    typedef config11_relu relu_config;
};

struct config15 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_11;
    static const unsigned in_width = OUT_WIDTH_11;
    static const unsigned n_filt = N_FILT_15-1;
    static const unsigned n_chan = N_FILT_11-1;
    static const unsigned n_filt_in = N_FILT_15;
    static const unsigned n_chan_in = N_FILT_11;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned out_height = OUT_HEIGHT_15;
    static const unsigned out_width = OUT_WIDTH_15;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const unsigned reuse = 72;
};

struct config16 : nnet::dense_config {
    static const unsigned block_factor = 16.0;
    static const unsigned merge_factor = 1;
    static const unsigned n_input = N_FILT_15;
    static const unsigned n_output = N_LAYER_16;
    static const unsigned n_in = OUT_HEIGHT_15*OUT_WIDTH_15*(N_FILT_15-1);
    static const unsigned n_out = N_LAYER_16-1;
    static const unsigned io_type = nnet::io_serial;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 16384;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_weightdefault_t weight_t;
    typedef model_default_t weightmult_t;
    typedef ap_uint<1> index_t;
};

struct softmax_config17 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_serial;
};


#endif
