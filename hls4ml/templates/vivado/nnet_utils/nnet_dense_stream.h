#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"
#include "hls_stream.h"
#include <math.h>
#include <assert.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]
) {
    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense(
    hls::stream<data_T> data_stream[CONFIG_T::n_chan],
    hls::stream<res_T>  res_stream[CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out])
{
    data_T data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    res_T res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for(int i_in = 0; i_in < CONFIG_T::n_in / CONFIG_T::n_chan; i_in++) {
        if (CONFIG_T::n_in / CONFIG_T::n_chan > 1) {
            #pragma HLS PIPELINE
        }

        DataPack: for (int i_pack = 0; i_pack < CONFIG_T::n_chan; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * CONFIG_T::n_chan + i_pack] = data_stream[i_pack].read();
        }
    }

    dense_wrapper<data_T, res_T, CONFIG_T>(data, res, weights, biases);

    ResWrite: for(unsigned i_out = 0; i_out < CONFIG_T::n_out / CONFIG_T::n_filt; i_out++) {
        if (CONFIG_T::n_out / CONFIG_T::n_filt > 1) {
            #pragma HLS PIPELINE
        }
   
        ResPack: for (int i_pack = 0; i_pack < CONFIG_T::n_filt; i_pack++) {
            #pragma HLS UNROLL
            res_stream[i_pack].write(res[i_out * CONFIG_T::n_filt + i_pack]);
        }

    }
}

}

#endif
