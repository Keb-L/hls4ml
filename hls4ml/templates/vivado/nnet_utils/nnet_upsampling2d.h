#ifndef NNET_UPSAMPLING2D_H_
#define NNET_UPSAMPLING2D_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

enum Interp_Op { nearest, bilinear };
// template<typename T, Interp_Op op, typename CONFIG_T>
// T interp_op(T (&x)[N], float h, float w, int c){
// 	switch(op){
//         case Nearest: return pixel_nearest(x);
//         case Bilinear: return pixel_bilinear(x);
// 	}
// }

struct upsampling2d_config {
	static const unsigned height_factor = 1;
    static const unsigned width_factor = 1;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned out_height = 10;
    static const unsigned out_width = 10;
    static const unsigned n_channel = 3;
    static const Interp_Op interp_op = Nearest;
};

template<class data_T, class res_T, typename CONFIG_T>
void upsampling2d_cf(
    data_T data[CONFIG_T::in_height*CONFIG_T::in_width*CONFIG_T::n_channel],
    res_T  res[CONFIG_T::out_height*CONFIG_T::out_width*CONFIG_T::n_channel]
{

}

template<class data_T, class res_T, typename CONFIG_T>
void upsampling2d_cl(
    data_T data[CONFIG_T::in_height*CONFIG_T::in_width*CONFIG_T::n_channel],
    res_T  res[CONFIG_T::out_height*CONFIG_T::out_width*CONFIG_T::n_channel]
{
    // Channel Last: format = [None, Height, Width, Channel]
    InterpHeight: for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        InterpWidth: for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            InterpChan: for(int cc = 0; cc < CONFIG_T::n_channel; cc++){
                // Nearest Neighbor implementation
                // For each pixel in the output image, find the nearest neighbor in the input image
                // Index in the result image
                int res_index = oh * CONFIG_T::out_width*CONFIG_T::n_channel
                              + ow * CONFIG_T::n_channel
                              + cc;
                // Find index of nearest neighbor in data image
                int data_h = round((float)(oh + 0.5)/CONFIG_T::height_factor - 0.5);
                int data_w = round((float)(ow + 0.5)/CONFIG_T::width_factor - 0.5);

                int data_index = data_h * CONFIG_T::in_width*CONFIG_T::n_channel
                               + data_w * CONFIG_T::n_channel
                               + cc;
                res[res_index] = data[data_index];
                // res[res_index] = interp_op<data_T, CONFIG_T::Interp_Op, CONFIG_T>(data, data_h, data_w, cc);
            } // end by-channel loop
        } // end by-width loop
    } // end by-height loop

}


} // End nnet namespace

#endif