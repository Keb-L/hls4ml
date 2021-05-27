#ifndef NNET_CONV_STREAM_H_
#define NNET_CONV_STREAM_H_

#include "ap_shift_reg.h"
#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

enum class conv_implementation { linebuffer=0, encoded=1};

// *************************************************
//       Encoded Implementation (Vlad's)
// *************************************************
template<unsigned K, unsigned S, unsigned W>
unsigned scale_index(const unsigned idx) {
    #pragma HLS INLINE

    if (idx < K - S) {
        return idx;
    }

    constexpr unsigned nW = ((W - K) / S) * S + K; // Nearest W without unused pixels on the right
    constexpr unsigned sW = (DIV_ROUNDUP(K, S) - 1) * S + K; // Scaled W that behaves like original W
    if (idx >= nW) {
        return sW;
    }

    const unsigned r = nW - idx;
    if (r <= K - S) {
        return sW - r;
    }

    return K - S + (idx - (K - S)) % S;
}

template<class data_T, class res_T, typename CONFIG_T>
void mult_buffer(
    hls::stream<data_T> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    res_T& res_pack,
    hls::stream<res_T>& res_stream,
    unsigned & outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    data_T data[CONFIG_T::kernel_size * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data complete
    res_T res[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res complete

    InitData: for (int id = 0; id < CONFIG_T::kernel_size * CONFIG_T::n_chan; id++) {
        #pragma HLS UNROLL
        data[id] = data_window[id].read();
    }

    #pragma HLS INLINE region
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<data_T, res_T, typename CONFIG_T::mult_config>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data, res, weights, biases);
    }

    CastLoop: for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        #pragma HLS UNROLL
        if (res_T::size / CONFIG_T::n_filt == 1) {
            res_pack[jj] = res[jj];
        } else {
            res_pack[outputs_ready * CONFIG_T::n_filt + jj] = res[jj];
        }
    }

    if (res_T::size / CONFIG_T::n_filt == 1) {
        res_stream.write(res_pack);
    } else {
        if (outputs_ready == (res_T::size / CONFIG_T::n_filt) - 1) {
            res_stream.write(res_pack);
            outputs_ready = 0;
        } else {
            outputs_ready++;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_encoded_output(
    const data_T& in_elem,
    hls::stream<data_T> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt],
    ap_uint<CONFIG_T::kernel_size> *pixel_idx
) {
    #pragma HLS INLINE

    MultLoop: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        CopyDataFilt: for (unsigned f = 0; f < CONFIG_T::kernel_size; f++) {
            #pragma HLS UNROLL
            CopyDataChan: for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                if (pixel_idx[p][f]) data_window[f * CONFIG_T::n_chan + c].write(in_elem[p * CONFIG_T::n_chan + c]);
            }
        }
        if (pixel_idx[p][CONFIG_T::kernel_size - 1]) {
            mult_buffer<data_T, res_T, CONFIG_T>(data_window, res_pack, res, outputs_ready, weights, biases);
        }
    }
}



// *************************************************
//       Line Buffer Implementation (Phil's)
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void kernel_shift_1d(
    const data_T& in_elem,
    res_T kernel_window[CONFIG_T::filt_width * CONFIG_T::n_chan]
) {
    #pragma HLS inline
    #pragma HLS PIPELINE 
    
    // Shift kernel_window by one step to the left (manual shift operation)
    static const int filt_width = CONFIG_T::filt_width - 1;
    KernelShiftWidth: for (int i_iw = 0; i_iw < filt_width; i_iw++) {
        KernelShiftChannel: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
            #pragma HLS UNROLL
            // Shift every element in kernel_window to the left
            kernel_window[i_iw * CONFIG_T::n_chan + i_ic] = kernel_window[(i_iw + 1) * CONFIG_T::n_chan + i_ic];
        }
    }

    // Insert shift_buffer column into right-most column of kernel
    static const int lastheight = (CONFIG_T::filt_width - 1) * CONFIG_T::n_chan;
    KernelPushChannel: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        #pragma HLS UNROLL
        kernel_window[lastheight + i_ic] = in_elem[i_ic];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void kernel_shift_2d(
    data_T shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan],
    res_T kernel_window[CONFIG_T::filt_width * CONFIG_T::filt_height * CONFIG_T::n_chan]
) {
    #pragma HLS inline
    #pragma HLS PIPELINE 

    // Shift kernel_window by one step to the left (manual shift operation)
    static const int filt_width = CONFIG_T::filt_width - 1;
    KernelShiftWidth: for (int i_iw = 0; i_iw < filt_width; i_iw++) {
        KernelShiftHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::filt_height; i_ih++) {
            KernelShiftChannel: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
                #pragma HLS UNROLL
                // Shift every element in kernel_window to the left
                kernel_window[i_ih * CONFIG_T::filt_width * CONFIG_T::n_chan + i_iw * CONFIG_T::n_chan + i_ic] = kernel_window[i_ih * CONFIG_T::filt_width * CONFIG_T::n_chan + (i_iw + 1) * CONFIG_T::n_chan + i_ic];
            }
        }
    }

    // Insert shift_buffer column into right-most column of kernel
    static const int lastheight = (CONFIG_T::filt_width - 1) * CONFIG_T::n_chan;
    KernelPushHeight: for (int i_ih = 0; i_ih < CONFIG_T::filt_height; i_ih++) {
        #pragma HLS UNROLL
        KernelPushChannel: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
            kernel_window[lastheight + i_ih * CONFIG_T::filt_width * CONFIG_T::n_chan + i_ic] = shift_buffer[i_ih][i_ic];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void shift_line_buffer(data_T data_in[CONFIG_T::n_chan], 
                    ap_shift_reg<data_T, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1][CONFIG_T::n_chan],
                    data_T kernel_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan]
) {
    
    #pragma HLS PIPELINE

    // Temporary buffer for popped (shifted) elements
    data_T shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable = shift_buffer complete dim = 0

    UpdateBuffer: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        #pragma HLS UNROLL

        // Insert pixel(s) at end of shift buffer
        shift_buffer[CONFIG_T::filt_height - 1][i_ic] = data_in[i_ic];
    }

    LineBufferDataIn: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        // Shift the shift buffer into the line buffer
        LineBufferShift: for (unsigned i_ih = 1; i_ih < CONFIG_T::filt_height; i_ih++) {
            #pragma HLS UNROLL
            data_T pop_elem = line_buffer[i_ih - 1][i_ic].shift(shift_buffer[CONFIG_T::filt_height - i_ih][i_ic]); // Shift the line buffer, return the popped pixel
            shift_buffer[CONFIG_T::filt_height - i_ih - 1][i_ic] = pop_elem; // Popped element placed back into shift_buffer, one row up.
        }
    }
    kernel_shift_2d<data_T, res_T, CONFIG_T>(shift_buffer, kernel_window);
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_output_buffer_2d(
    hls::stream<data_T> data_stream[CONFIG_T::n_chan],
    ap_shift_reg<data_T, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1][CONFIG_T::n_chan],
    hls::stream<res_T> res_stream[CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    // Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    // Counters
    static int pX = 0; // Pixel X
    static int pY = 0; // Pixel Y

    static int sX = 0; // Stride X
    static int sY = 0; // Stride Y

    static data_T kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=kernel_data complete

    res_T res_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=res_out complete dim = 0

    data_T data_in[CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=data_in complete

    for(int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        data_in[i_ic] = data_stream[i_ic].read();
    }

    // Add pixel to buffer
    nnet::shift_line_buffer<data_T, res_T, CONFIG_T>(data_in, line_buffer, kernel_data);

    // Check to see if we have a full kernel
    if ( (sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
        
        // Dense multiply
        #pragma HLS INLINE region
        if (CONFIG_T::strategy == nnet::latency) {
            dense_latency<data_T, res_T, typename CONFIG_T::mult_config>(kernel_data, res_out, weights, biases);
        } else {
            dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(kernel_data, res_out, weights, biases);
        }

        // Pack output
        CastLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
            #pragma HLS UNROLL
            res_stream[i_ic].write(res_out[i_ic]);
        }
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width)  // Includes padding, end of line (padded)
    {
        pX = 0; 
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) {  // Reached bottom of image
            pY = 0; 
            sY = 0;
        } else {
            pY = pY + 1;
            // Update stride (threshold) ? subtract stride : increment stride
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1; 
        }
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1; 
    }
}

// Conv 1D compute output
template<class data_T, class res_T, typename CONFIG_T>
void compute_output_buffer_1d(
    const data_T& in_elem,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    // Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;

    // Counters
    static int pX = 0; // pixel counter
    static int sX = 0; // stride counter

    static data_T kernel_data[CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=kernel_data complete

    res_T res_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res_out complete dim = 0

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack

    // Add pixel to buffer
    nnet::kernel_shift_1d<data_T, res_T, CONFIG_T>(in_elem, kernel_data);

    // Check to see if we have a full kernel
    if ( (sX - lShiftX) == 0 && pX > lShiftX - 1 ) {
        
        // Dense multiply
        #pragma HLS INLINE region
        if (CONFIG_T::strategy == nnet::latency) {
            dense_latency<data_T, res_T, typename CONFIG_T::mult_config>(kernel_data, res_out, weights, biases);
        } else {
            dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(kernel_data, res_out, weights, biases);
        }

        // Pack output
        CastLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
            #pragma HLS UNROLL
            res_pack[i_ic] = res_out[i_ic];
        }

        // Write output to stream when output ready
        res_stream.write(res_pack);
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width)  // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1; 
    }
}

}
#endif
