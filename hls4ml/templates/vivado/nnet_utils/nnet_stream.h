
#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "hls_stream.h"

template<class data_T, class res_T, int N>
void copy_stream(
    hls::stream<data_T> data[CONFIG_T::n_chan], 
    hls::stream<data_T> res[CONFIG_T::n_chan]) {
    // Copy N values from input to output stream

    CopyLoop: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE

        // Copy each channel over to the new stream
        ChannelLoop: for(int j = 0; j < CONFIG_T::n_chan; j++) {
            #pragma HLS UNROLL

            // Read the data and cast to output data type
            data_T data_in = data.read();
            res.write(data_in);
        }
    }
}

#endif
