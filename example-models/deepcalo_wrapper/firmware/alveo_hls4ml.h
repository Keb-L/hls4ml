#ifndef ALVEO_HLS4ML_H_
#define ALVEO_HLS4ML_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "kernel_params.h"

  extern "C" void alveo_hls4ml(
      const bigdata_t *in, // Read-Only Vector
      const model_default_t *in_weights1,
      const model_default_t *in_weights2,
      const model_default_t *in_weights3,
      const model_default_t *in_weights4,
      bigdata_t *out // Output Result
  );

#endif