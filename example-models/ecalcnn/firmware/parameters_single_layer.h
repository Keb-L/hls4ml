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
#define N_INPUT_1_1_TRUE 56
#define N_INPUT_1_1 56
#define N_INPUT_2_1 55
#define N_INPUT_3_1 4
#define OUT_HEIGHT_102 56
#define OUT_WIDTH_102 55
#define N_CHANNEL_102 4
#define OUT_HEIGHT_2 56
#define OUT_WIDTH_2 55
#define N_FILT_2 16
#define OUT_HEIGHT_6 28
#define OUT_WIDTH_6 27
#define N_FILT_6 16
#define OUT_HEIGHT_7 28
#define OUT_WIDTH_7 27
#define N_FILT_7 32
#define OUT_HEIGHT_11 28
#define OUT_WIDTH_11 27
#define N_FILT_11 32
#define OUT_HEIGHT_15 14
#define OUT_WIDTH_15 13
#define N_FILT_15 32
#define OUT_HEIGHT_16 14
#define OUT_WIDTH_16 13
#define N_FILT_16 64
#define OUT_HEIGHT_20 14
#define OUT_WIDTH_20 13
#define N_FILT_20 64
#define OUT_HEIGHT_24 7
#define OUT_WIDTH_24 6
#define N_FILT_24 64
#define OUT_HEIGHT_25 7
#define OUT_WIDTH_25 6
#define N_FILT_25 128
#define OUT_HEIGHT_29 7
#define OUT_WIDTH_29 6
#define N_FILT_29 128
#define OUT_HEIGHT_33 3
#define OUT_WIDTH_33 3
#define N_FILT_33 128
#define OUT_HEIGHT_34 3
#define OUT_WIDTH_34 3
#define N_FILT_34 256
#define OUT_HEIGHT_38 3
#define OUT_WIDTH_38 3
#define N_FILT_38 256
#define N_LAYER_42 256
#define N_LAYER_46 256
#define N_LAYER_50 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer102_t;
typedef ap_fixed<16,6> layer103_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<16,6> layer16_t;
typedef ap_fixed<16,6> layer18_t;
typedef ap_fixed<16,6> layer19_t;
typedef ap_fixed<16,6> layer20_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<16,6> layer23_t;
typedef ap_fixed<16,6> layer24_t;
typedef ap_fixed<16,6> layer25_t;
typedef ap_fixed<16,6> layer27_t;
typedef ap_fixed<16,6> layer28_t;
typedef ap_fixed<16,6> layer29_t;
typedef ap_fixed<16,6> layer31_t;
typedef ap_fixed<16,6> layer32_t;
typedef ap_fixed<16,6> layer33_t;
typedef ap_fixed<16,6> layer34_t;
typedef ap_fixed<16,6> layer36_t;
typedef ap_fixed<16,6> layer37_t;
typedef ap_fixed<16,6> layer38_t;
typedef ap_fixed<16,6> layer40_t;
typedef ap_fixed<16,6> layer41_t;
typedef ap_fixed<16,6> layer42_t;
typedef ap_uint<1> bias42_t;
typedef ap_fixed<16,6> layer45_t;
typedef ap_fixed<16,6> layer46_t;
typedef ap_uint<1> bias46_t;
typedef ap_fixed<16,6> layer49_t;
typedef ap_fixed<16,6> layer50_t;
typedef ap_fixed<16,6> result_t;

//hls-fpga-machine-learning insert layer-config
struct config102 : nnet::upsampling2d_config {
    static const unsigned height_factor = 1;
    static const unsigned width_factor = 5;
    static const unsigned in_height = N_INPUT_1_1_TRUE;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned out_height = OUT_HEIGHT_102;
    static const unsigned out_width = OUT_WIDTH_102;
    static const unsigned n_channel = N_CHANNEL_102;
    static const nnet::Interp_Op interp_op = nnet::nearest;
};

struct config103 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_102*OUT_WIDTH_102*N_CHANNEL_102;
    static const unsigned n_filt = 4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 2;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};


struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 20;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};


struct config2_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_2;
    static const unsigned n_filt = 16;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config2_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config2 : nnet::conv2d_config {
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
    static const unsigned in_height = N_INPUT_1_1_TRUE;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_chan = N_INPUT_3_1;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned n_filt = N_FILT_2;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_2;
    static const unsigned out_width = OUT_WIDTH_2;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config2_mult mult_config;
    typedef config2_norm norm_config;
    typedef config2_relu relu_config;
};

struct config6 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_filt = N_FILT_6;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width =  2;
    static const unsigned out_height = OUT_HEIGHT_6;
    static const unsigned out_width = OUT_WIDTH_6;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const unsigned reuse = 64;
};

struct config7_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 72;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config7_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_7;
    static const unsigned n_filt = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config7_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_7;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config7 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_6;
    static const unsigned in_width = OUT_WIDTH_6;
    static const unsigned n_chan = N_FILT_6;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_7;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_7;
    static const unsigned out_width = OUT_WIDTH_7;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config7_mult mult_config;
    typedef config7_norm norm_config;
    typedef config7_relu relu_config;
};

struct config11_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 72;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config11_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_11;
    static const unsigned n_filt = 32;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config11_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_11;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config11 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_7;
    static const unsigned in_width = OUT_WIDTH_7;
    static const unsigned n_chan = N_FILT_7;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_11;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_11;
    static const unsigned out_width = OUT_WIDTH_11;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config11_mult mult_config;
    typedef config11_norm norm_config;
    typedef config11_relu relu_config;
};

struct config15 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_11;
    static const unsigned in_width = OUT_WIDTH_11;
    static const unsigned n_filt = N_FILT_15;
    static const unsigned n_chan = N_FILT_11;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width =  2;
    static const unsigned out_height = OUT_HEIGHT_15;
    static const unsigned out_width = OUT_WIDTH_15;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const unsigned reuse = 64;
};

struct config16_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 144;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config16_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_16;
    static const unsigned n_filt = 64;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config16_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config16 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_15;
    static const unsigned in_width = OUT_WIDTH_15;
    static const unsigned n_chan = N_FILT_15;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_16;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_16;
    static const unsigned out_width = OUT_WIDTH_16;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config16_mult mult_config;
    typedef config16_norm norm_config;
    typedef config16_relu relu_config;
};


struct config20_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 144;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config20_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_20;
    static const unsigned n_filt = 64;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config20_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_20;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config20 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_16;
    static const unsigned in_width = OUT_WIDTH_16;
    static const unsigned n_chan = N_FILT_16;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_20;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_20;
    static const unsigned out_width = OUT_WIDTH_20;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config20_mult mult_config;
    typedef config20_norm norm_config;
    typedef config20_relu relu_config;
};

struct config24 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_20;
    static const unsigned in_width = OUT_WIDTH_20;
    static const unsigned n_filt = N_FILT_24;
    static const unsigned n_chan = N_FILT_20;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width =  2;
    static const unsigned out_height = OUT_HEIGHT_24;
    static const unsigned out_width = OUT_WIDTH_24;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const unsigned reuse = 64;
};

struct config25_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 288;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config25_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_25;
    static const unsigned n_filt = 128;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config25_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_25;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config25 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_24;
    static const unsigned in_width = OUT_WIDTH_24;
    static const unsigned n_chan = N_FILT_24;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_25;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_25;
    static const unsigned out_width = OUT_WIDTH_25;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config25_mult mult_config;
    typedef config25_norm norm_config;
    typedef config25_relu relu_config;
};

struct config29_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 576;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};
struct config29_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_29;
    static const unsigned n_filt = 128;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config29_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_29;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config29 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_25;
    static const unsigned in_width = OUT_WIDTH_25;
    static const unsigned n_chan = N_FILT_25;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_29;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_29;
    static const unsigned out_width = OUT_WIDTH_29;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config29_mult mult_config;
    typedef config29_norm norm_config;
    typedef config29_relu relu_config;
};

struct config33 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_29;
    static const unsigned in_width = OUT_WIDTH_29;
    static const unsigned n_filt = N_FILT_33;
    static const unsigned n_chan = N_FILT_29;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width =  2;
    static const unsigned out_height = OUT_HEIGHT_33;
    static const unsigned out_width = OUT_WIDTH_33;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const unsigned reuse = 64;
};

struct config34_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 576;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config34_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_34;
    static const unsigned n_filt = 256;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config34_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_34;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config34 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_33;
    static const unsigned in_width = OUT_WIDTH_33;
    static const unsigned n_chan = N_FILT_33;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_34;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_34;
    static const unsigned out_width = OUT_WIDTH_34;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config34_mult mult_config;
    typedef config34_norm norm_config;
    typedef config34_relu relu_config;
};

struct config38_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 1152;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
};

struct config38_norm : nnet::batchnorm_config {
    static const unsigned n_in = N_FILT_38;
    static const unsigned n_filt = 256;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct config38_relu : nnet::activ_config {
    static const unsigned n_in = N_FILT_38;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config38 : nnet::conv2d_config {
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned in_height = OUT_HEIGHT_34;
    static const unsigned in_width = OUT_WIDTH_34;
    static const unsigned n_chan = N_FILT_34;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned n_filt = N_FILT_38;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_38;
    static const unsigned out_width = OUT_WIDTH_38;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config38_mult mult_config;
    typedef config38_norm norm_config;
    typedef config38_relu relu_config;
};

struct config42 : nnet::dense_config {
    static const unsigned n_in = OUT_HEIGHT_38*OUT_WIDTH_38*N_FILT_38;
    static const unsigned n_out = N_LAYER_42;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1152;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 589824;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias42_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct LeakyReLU_config45 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_42;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config46 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_42;
    static const unsigned n_out = N_LAYER_46;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias46_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct LeakyReLU_config49 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_46;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config50 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_46;
    static const unsigned n_out = N_LAYER_50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config52 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_50;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};


#endif
