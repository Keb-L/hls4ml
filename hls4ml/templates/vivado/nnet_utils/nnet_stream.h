
#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {

struct copy_config
{
    static const unsigned n_chan = 0;
    static const unsigned n_elem = 0;
};


// template<class data_T, class res_T, typename CONFIG_T>
// void copy_stream(
//     hls::stream<data_T> data[CONFIG_T::n_chan], 
//     hls::stream<data_T> res[CONFIG_T::n_chan]) {
//     // Copy all values from input to output stream

//     CopyLoop: for(int i = 0; i < CONFIG_T::n_elem; i++) {
//         #pragma HLS PIPELINE

//         // Copy each channel over to the new stream
//         ChannelLoop: for(int j = 0; j < CONFIG_T::n_chan; j++) {
//             #pragma HLS UNROLL

//             // Read the data and cast to output data type
//             data_T data_in = data[j].read();
//             res[j].write(data_in);
//         }
//     }
// }

template<class data_T, class res_T, int N, int K>
void copy_stream(
    hls::stream<data_T> data[N], 
    hls::stream<data_T> res[N]) {
    // Copy all values from input to output stream

    CopyLoop: for(int i = 0; i < K; i++) {
        #pragma HLS PIPELINE

        // Copy each channel over to the new stream
        ChannelLoop: for(int j = 0; j < N; j++) {
            #pragma HLS UNROLL

            // Read the data and cast to output data type
            data_T data_in = data[j].read();
            res[j].write(data_in);
        }
    }
}


}

#endif
