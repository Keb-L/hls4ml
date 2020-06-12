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

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters_full_layer.h"

// Prototype of top level function for C-synthesis
void myproject_full_layer(
  input_t em_barrel[N_INPUT_1_1][N_INPUT_2_1][N_INPUT_3_1]
//   input_t em_barrel[IN_HEIGHT_102*IN_WIDTH_102*IN_CHANNEL_102]
, result_t layer52_out[N_LAYER_50]
// ,    result_t preds[N_RES*N_SIZE] // For synth!
, model_default_t w38[589824]
, model_default_t w42[589824]
);


void subimage_stream(bool iReset
, hls::stream<input_t>  input[N_INPUT_3_1]
, hls::stream<result_t> output[N_RES]
, model_default_t w38[589824]
);

#endif
