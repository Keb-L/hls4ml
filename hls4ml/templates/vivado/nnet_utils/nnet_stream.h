
#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "hls_stream.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void copy(
    hls::stream<data_T> data[CONFIG_T::n_chan], 
    hls::stream<data_T> res[CONFIG_T::n_chan]) {
    // Copy all values from input to output stream

    CopyLoop: for(int i = 0; i < CONFIG_T::n_elem; i++) {
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

}

#endif
