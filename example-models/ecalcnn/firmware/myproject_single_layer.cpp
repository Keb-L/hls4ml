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

#include "myproject_single_layer.h"

//hls-fpga-machine-learning insert weights
#include "weights/b103.h"
#include "weights/s103.h"
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/s4.h"
#include "weights/b4.h"
#include "weights/w7.h"
#include "weights/b7.h"
#include "weights/s9.h"
#include "weights/b9.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/s13.h"
#include "weights/b13.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/s18.h"
#include "weights/b18.h"
#include "weights/w20.h"
#include "weights/b20.h"
#include "weights/s22.h"
#include "weights/b22.h"
#include "weights/w25.h"
#include "weights/b25.h"
#include "weights/s27.h"
#include "weights/b27.h"
#include "weights/w29.h"
#include "weights/b29.h"
#include "weights/s31.h"
#include "weights/b31.h"
#include "weights/w34.h"
#include "weights/b34.h"
#include "weights/s36.h"
#include "weights/b36.h"
////#include "weights/w38.h"
#include "weights/b38.h"
#include "weights/s40.h"
#include "weights/b40.h"
////#include "weights/w42.h"
#include "weights/b42.h"
#include "weights/w46.h"
#include "weights/b46.h"
#include "weights/w50.h"
#include "weights/b50.h"

void myproject(
input_t em_barrel[N_INPUT_1_1][N_INPUT_2_1][N_INPUT_3_1],
// result_t layer52_out[N_LAYER_50],
result_t layer41_out[OUT_WIDTH_2*OUT_HEIGHT_2*N_FILT_2],

model_default_t w38[589824],
model_default_t w42[589824]                    
//unsigned short &const_size_in_1,
//unsigned short &const_size_out_1
) {

    #pragma HLS interface bram port=w38
    #pragma HLS interface bram port=w42
    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=layer41_out complete dim=0 
    #pragma HLS INTERFACE ap_vld port=em_barrel,layer41_out 
    #pragma HLS DATAFLOW 

    //const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    //const_size_out_1 = N_LAYER_50;

    #ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
    //hls-fpga-machine-learning insert load weights
    nnet::load_weights_from_txt<model_default_t, 589824>(w42, "w42.txt");
    nnet::load_weights_from_txt<bias42_t, 256>(b42, "b42.txt");
    nnet::load_weights_from_txt<model_default_t, 65536>(w46, "w46.txt");
    nnet::load_weights_from_txt<bias46_t, 256>(b46, "b46.txt");
    nnet::load_weights_from_txt<model_default_t, 256>(w50, "w50.txt");
    nnet::load_weights_from_txt<model_default_t, 1>(b50, "b50.txt");
    nnet::load_weights_from_txt<model_default_t, 4>(s103, "s103.txt");
    nnet::load_weights_from_txt<model_default_t, 4>(b103, "b103.txt");
    loaded_weights = true;
    }
    #endif


    // layer41_t layer41_out[OUT_WIDTH_38*OUT_HEIGHT_38*N_FILT_38];
    // #pragma HLS ARRAY_RESHAPE variable=layer41_out block factor=256
    unsigned index=0; 

    hls::stream<input_t>   sInput  [N_INPUT_3_1];
    hls::stream<result_t>  sOutput [N_FILT_2];
    #pragma HLS stream variable=sInput      depth=1
    #pragma HLS stream variable=sOutput     depth=1

    // layer102_t layer102_out[OUT_HEIGHT_102*OUT_WIDTH_102*N_CHANNEL_102];
    // #pragma HLS ARRAY_PARTITION variable=layer102_out complete dim=0
    // nnet::upsampling2d_cl<input_t, layer102_t, config102>(em_barrel, layer102_out);

    // layer103_t layer103_out[OUT_HEIGHT_102*OUT_WIDTH_102*N_CHANNEL_102];
    // #pragma HLS ARRAY_PARTITION variable=layer103_out complete dim=0
    // nnet::normalize<layer102_t, layer103_t, config103>(layer102_out, layer103_out, s103, b103);

    // layer103_t layer103_out_uf[OUT_HEIGHT_102][OUT_WIDTH_102][N_CHANNEL_102];
    // nnet::unflatten<layer103_t, OUT_HEIGHT_102, OUT_WIDTH_102, N_CHANNEL_102>(layer103_out, layer103_out_uf);

    bool lReset = true;
    for(unsigned iC = 0; iC < N_INPUT_1_1; iC++) { 
        //Read in the input image to bottom row of buffer
        LoopSubstream:
        for(unsigned i1 = 0; i1 < N_INPUT_2_1; i1++) {
            LoopInput:
            for(unsigned i2 = 0; i2 < N_INPUT_3_1; i2++) { 
            #pragma HLS UNROLL
            if(i1*iC < N_INPUT_1_1_TRUE*N_INPUT_2_1) {
                sInput[i2].write(em_barrel[iC][i1][i2]);
            } else { 
                input_t pVal = 0; 
                sInput[i2].write(pVal);
                }
            }

            subimage_stream(lReset,sInput,sOutput,w38);
            lReset = false;

            if(!sOutput[0].empty()) { 
            LoopOutput:
            std::cout << "---> " <<std::endl;
            for(unsigned iX = 0; iX < N_FILT_2; iX++) { 
                #pragma HLS UNROLL 
                layer41_out[index*N_FILT_2+iX] = (result_t)sOutput[iX].read();
            }
            index++; 
            // if(index > 9) break;
            std::cout << "---> " << index << std::endl;
                if(index == (OUT_WIDTH_2*OUT_HEIGHT_2)) { 
                    //dense layers

                    //     layer42_t layer42_out[N_LAYER_42];
                    //     #pragma HLS ARRAY_PARTITION variable=layer42_out complete dim=0
                    // nnet::dense_large<layer41_t, layer42_t, config42>(layer41_out, layer42_out, w42, b42);

                    // input_t alpha = 0.3;
                    // layer45_t layer45_out[N_LAYER_42];
                    //     #pragma HLS ARRAY_PARTITION variable=layer45_out complete dim=0
                    // nnet::leaky_relu<layer42_t, layer45_t, LeakyReLU_config45>(layer42_out,alpha, layer45_out);

                    // layer46_t layer46_out[N_LAYER_46];
                    //     #pragma HLS ARRAY_PARTITION variable=layer46_out complete dim=0
                    // nnet::dense_large<layer45_t, layer46_t, config46>(layer45_out, layer46_out, w46, b46);

                    // layer49_t layer49_out[N_LAYER_46];
                    //     #pragma HLS ARRAY_PARTITION variable=layer49_out complete dim=0
                    // nnet::leaky_relu<layer46_t, layer49_t, LeakyReLU_config49>(layer46_out,alpha, layer49_out);

                    // layer50_t layer50_out[N_LAYER_50];
                    //     #pragma HLS ARRAY_PARTITION variable=layer50_out complete dim=0
                    //     nnet::dense_large<layer49_t, layer50_t, config50>(layer49_out, layer50_out, w50, b50);

                    // nnet::relu<layer50_t, result_t, relu_config52>(layer50_out, layer52_out);
                    break;
                }
            }
        }
    }
}
void subimage_stream(bool iReset, 
hls::stream<input_t>  input[N_INPUT_3_1],
// hls::stream<result_t> output[N_FILT_38] ,
hls::stream<result_t> output[N_FILT_2] ,
model_default_t w38[589824]          
) { 

    #pragma HLS interface bram port=w38

    #ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
    //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 1600>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(s4, "s4.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 4608>(w7, "w7.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b7, "b7.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s9, "s9.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 9216>(w11, "w11.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b11, "b11.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s13, "s13.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b13, "b13.txt");
        nnet::load_weights_from_txt<model_default_t, 18432>(w16, "w16.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b16, "b16.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s18, "s18.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b18, "b18.txt");
        nnet::load_weights_from_txt<model_default_t, 36864>(w20, "w20.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b20, "b20.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s22, "s22.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b22, "b22.txt");
        nnet::load_weights_from_txt<model_default_t, 73728>(w25, "w25.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b25, "b25.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s27, "s27.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b27, "b27.txt");
        nnet::load_weights_from_txt<model_default_t, 147456>(w29, "w29.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b29, "b29.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s31, "s31.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b31, "b31.txt");
        nnet::load_weights_from_txt<model_default_t, 294912>(w34, "w34.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b34, "b34.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s36, "s36.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b36, "b36.txt");
        //nnet::load_weights_from_txt<model_default_t, 589824>(w38, "w38.txt");
        //nnet::load_weights_from_txt<model_default_t, 256>(b38, "b38.txt");
        //nnet::load_weights_from_txt<model_default_t, 256>(s40, "s40.txt");
        //nnet::load_weights_from_txt<model_default_t, 256>(b40, "b40.txt");
        loaded_weights = true;
    }
    #endif


    // static hls::stream<layer2_t> layer2_out[N_FILT_2];
    // #pragma HLS stream variable=layer2_out      depth=1
    input_t alpha = 0.3;
    // nnet::conv_2d_large_stream_norm_leaky<input_t,layer2_t,config2>(iReset,input,layer2_out,w2,b2,s4,b4,alpha);
    nnet::conv_2d_large_stream_norm_leaky<input_t,layer2_t,config2>(iReset,input,output,w2,b2,s4,b4,alpha);

    // static hls::stream<layer6_t> layer6_out[N_FILT_6];
    // #pragma HLS stream variable=layer6_out      depth=1
    // if(!layer2_out[0].empty()) nnet::pool_2d_large_stream<layer2_t,layer6_t,config6>(layer2_out,layer6_out);

    // static hls::stream<layer7_t> layer7_out[N_FILT_7];
    // #pragma HLS stream variable=layer7_out      depth=1
    // if(!layer6_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer6_t,layer2_t,config7>(iReset,layer6_out,layer7_out,w7,b7,s9,b9,alpha);

    // static hls::stream<layer11_t> layer11_out[N_FILT_11];
    // #pragma HLS stream variable=layer11_out      depth=1
    // if(!layer7_out[0].empty())  nnet::conv_2d_large_stream_norm_leaky<layer7_t,layer11_t,config11>(iReset,layer7_out,layer11_out,w11,b11,s13,b13,alpha);

    // static hls::stream<layer15_t> layer15_out[N_FILT_15];
    // #pragma HLS stream variable=layer15_out      depth=1
    // if(!layer11_out[0].empty()) nnet::pool_2d_large_stream<layer11_t,layer15_t,config15>(layer11_out,layer15_out);

    // static hls::stream<layer16_t> layer16_out[N_FILT_16];
    // #pragma HLS stream variable=layer16_out      depth=1
    // if(!layer15_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer15_t,layer16_t,config16>(iReset,layer15_out,layer16_out,w16,b16,s18,b18,alpha);

    // static hls::stream<layer20_t> layer20_out[N_FILT_20];
    // #pragma HLS stream variable=layer20_out      depth=1
    // //if(!layer16_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer16_t,layer20_t,config20>(iReset,layer16_out,output,w20,b20,s22,b22,alpha);
    // if(!layer16_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer16_t,layer20_t,config20>(iReset,layer16_out,layer20_out,w20,b20,s22,b22,alpha);

    // static hls::stream<layer24_t> layer24_out[N_FILT_24];
    // #pragma HLS stream variable=layer24_out      depth=1
    // if(!layer20_out[0].empty()) nnet::pool_2d_large_stream<layer20_t,layer24_t,config24>(layer20_out,layer24_out);

    // static hls::stream<layer25_t> layer25_out[N_FILT_25];
    // #pragma HLS stream variable=layer25_out      depth=1
    // if(!layer24_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer24_t,layer25_t,config25>(iReset,layer24_out,layer25_out,w25,b25,s27,b27,alpha);

    // static hls::stream<layer29_t> layer29_out[N_FILT_29];
    // #pragma HLS stream variable=layer29_out      depth=1
    // if(!layer25_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer25_t,layer29_t,config29>(iReset,layer25_out,layer29_out,w29,b29,s31,b31,alpha);

    // static hls::stream<layer33_t> layer33_out[N_FILT_33];
    // #pragma HLS stream variable=layer33_out      depth=1
    // if(!layer29_out[0].empty()) nnet::pool_2d_large_stream<layer29_t,layer33_t,config33>(layer29_out,layer33_out);

    // static hls::stream<layer34_t> layer34_out[N_FILT_34];
    // #pragma HLS stream variable=layer34_out      depth=1
    // if(!layer33_out[0].empty()) nnet::conv_2d_large_stream_norm_leaky<layer33_t,layer34_t,config34>(iReset,layer33_out,layer34_out,w34,b34,s36,b36,alpha);

    // //hls::stream<layer38_t> layer38_out[N_FILT_38];
    // //#pragma HLS stream variable=layer38_out      depth=1
    // if(!layer34_out[0].empty())  nnet::conv_2d_large_stream_norm_leaky<layer34_t,layer38_t,config38>(iReset,layer34_out,output,w38,b38,s40,b40,alpha);
}
