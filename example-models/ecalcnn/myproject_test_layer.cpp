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
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters_full.h"
#include "firmware/myproject_layer.h"


#define CHECKPOINT 10

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif


  std::string iline;
  std::string pline;
  int e = 0;

  bool iReset=true;
  //hls::stream<input_t> inputstream[N_FILT_88];//88];//18];//164];
  input_t inputstream[N_FILT_88];//88];//18];//164];
  hls::stream<result_t> outputstream[N_FILT_92];//92];//21];//167];
  for(int i0 = 0; i0 < N_INPUT_3_1; i0++) { 
    inputstream[i0].write(i0);
  }
  //model_default_t w167[config21::mult_config::n_in*config21::mult_config::n_out];
  //model_default_t w167[config167::mult_config::n_in*config167::mult_config::n_out];
  model_default_t w167[config92::mult_config::n_in*config92::mult_config::n_out];
  subimage_stream(iReset,inputstream,outputstream,w167);
  
  //hls-fpga-machine-learning insert output
  // for(int i = 0; i < N_LAYER_OUT_3; i++) {
  //    std::cout << layer176_out[i].read() << " ";
  //  }
  //  std::cout << std::endl;

  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
