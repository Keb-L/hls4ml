//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"

//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/b7.h"
#include "weights/b11.h"
#include "weights/w16.h"
#include "weights/b16.h"

void myproject(
    hls::stream<input_t> input_6[N_INPUT_3_1],
    // hls::stream<result_t> layer17_out[N_LAYER_16],
    hls::stream<result_t> layer17_out[N_FILT_7],
    model_weightdefault_t w7[73728],
model_weightdefault_t w11[147456],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_6,layer17_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_16;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_weightdefault_t, 4800>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_weightdefault_t, 73728>(w7, "w7.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b7, "b7.txt");
        nnet::load_weights_from_txt<model_weightdefault_t, 147456>(w11, "w11.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b11, "b11.txt");
        nnet::load_weights_from_txt<model_weightdefault_t, 16384>(w16, "w16.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(b16, "b16.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer5_t> layer5_out[N_FILT_2];
    #pragma HLS STREAM variable=layer5_out depth=1000 dim=1
	for(int i0 = 0; i0 < 16*16; i0++) {
		nnet::conv_2d_large_cl<input_t, layer5_t, config2>(input_6, layer5_out, w2, b2);
	}

    hls::stream<layer6_t> layer6_out[N_FILT_6];
    #pragma HLS STREAM variable=layer6_out depth=1000 dim=1
	for(int i0 = 0; i0 < 16*16; i0++) {
		nnet::pooling2d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out);
	}

    hls::stream<layer10_t> layer10b_out[N_FILT_7];
    #pragma HLS STREAM variable=layer10b_out depth=1000 dim=1
	for(int i0 = 0; i0 < 8*8; i0++) {
		nnet::conv_2d_large_cl<layer6_t, layer10_t, config7>(layer6_out, layer10b_out, w7, b7);
	}

    // hls::stream<layer14_t> layer14_out[N_FILT_11];
    // #pragma HLS STREAM variable=layer14_out depth=64 dim=1
	// for(int i0 = 0; i0 < 8*8; i0++) {
	// 	nnet::conv_2d_large_cl2<layer10_t, layer14_t, config11>(layer10_out, layer14_out, w11, b11);
	// }

    // hls::stream<layer15_t> layer15_out[N_FILT_15];
    // #pragma HLS STREAM variable=layer15_out depth=64 dim=1
	// for(int i0 = 0; i0 < 8*8; i0++) {
	// 	nnet::pooling2d_cl<layer14_t, layer15_t, config15>(layer14_out, layer15_out);
	// }

    // hls::stream<layer16_t> layer16_out[N_LAYER_16];
    // #pragma HLS STREAM variable=layer16_out depth=1 dim=1
	// for(int i0 = 0; i0 < 4*4; i0++) {
	// 	nnet::dense_large_stream<layer15_t, layer16_t, config16>(layer15_out, layer16_out, w16, b16);
	// }

	// hls::stream<result_t>layer17b_out[N_LAYER_16];
	// #pragma HLS STREAM variable=layer17b_out depth=1 dim=1
	// nnet::softmax_stream<layer16_t, result_t, softmax_config17>(layer16_out, layer17b_out);

    for(int i1 = 0; i1 < 8*8; i1++) {
	for(int i0 = 0; i0 < N_FILT_7; i0++) { 
		#pragma HLS UNROLL
		result_t pTmp = (result_t) layer10b_out[i0].read();
		layer17_out[i0].write(pTmp);
	}
    }

}


