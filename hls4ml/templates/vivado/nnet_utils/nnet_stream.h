
#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {

struct copy_config {
  static const unsigned n_chan = 0;
  static const unsigned n_elem = 0;
};

template <class data_T, class res_T, int N, int K>
void copy_stream(
    hls::stream<data_T> data[N],
    hls::stream<data_T> res[N]) {
  // Copy all values from input to output stream

CopyLoop:
  for (int i = 0; i < K; i++) {
#pragma HLS PIPELINE

  // Copy each channel over to the new stream
  ChannelLoop:
    for (int j = 0; j < N; j++) {
#pragma HLS UNROLL

      // Read the data and cast to output data type
      data_T data_in = data[j].read();
      res[j].write(data_in);
    }
  }
}

// TODO: This really should be in its own nnet file, but this is an experimental branch.
struct padding2d_config {
    static const unsigned n_chan = 10;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned out_height = 10;
    static const unsigned out_width = 10;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
};

template <class res_T, typename CONFIG_T>
void fill_zero(hls::stream<res_T> res[CONFIG_T::n_chan]) {
  #pragma HLS INLINE
  ChannelFillZero:
  for (int c = 0; c < CONFIG_T::n_chan; c++) {
    #pragma HLS UNROLL
    res[c].write(0);
  }
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(
    hls::stream<data_T> data[CONFIG_T::n_chan],
    hls::stream<res_T> res[CONFIG_T::n_chan]) {
PadTop:
  for (int i = 0; i < CONFIG_T::pad_top; i++) {
  PadTopWidth:
    for (int j = 0; j < CONFIG_T::out_width; j++) {
      fill_zero<res_T, CONFIG_T>(res);
    }
  }

PadMain:
  for (int i = 0; i < CONFIG_T::in_height; i++) {
  PadLeft:
    for (int j = 0; j < CONFIG_T::pad_left; j++) {
      fill_zero<res_T, CONFIG_T>(res);
    }
  CopyMain:
    for (int j = 0; j < CONFIG_T::in_width; j++) {
    CopyChannel:
      for (int c = 0; c < CONFIG_T::n_chan; c++) {
        res[c].write(data[c].read());
      }
    }
  PadRight:
    for (int j = 0; j < CONFIG_T::pad_right; j++) {
      fill_zero<res_T, CONFIG_T>(res);
    }
  }

PadBottom:
  for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
  PadBottomWidth:
    for (int j = 0; j < CONFIG_T::out_width; j++) {
      fill_zero<res_T, CONFIG_T>(res);
    }
  }
}

}  // namespace nnet

#endif
