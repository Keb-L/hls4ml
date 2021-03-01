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

#include "firmware/parameters.h"
#include "firmware/myproject.h"

#define CHECKPOINT 5000

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
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      e++;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
model_weightdefault_t w7[73728];
model_weightdefault_t w11[147456];
      for(int iX = 0; iX < 11; iX++){
unsigned index=0;
      hls::stream<input_t> input_6[N_INPUT_3_1];
      for(int i0 = 0; i0 < 16*16; i0++) {
      for(int i1 = 0; i1 < 16*3; i1++) {
      for(int i2 = 0; i2 < 3+1; i2++) {
        input_6[i2].write(in[index]);index++;
      }
      }
      }
      hls::stream<result_t> layer17_out[N_LAYER_16];

      //hls-fpga-machine-learning insert top-level-function
      unsigned short size_in1,size_out1;
      myproject(input_6,layer17_out,w7, w11,size_in1,size_out1);

      //hls-fpga-machine-learning insert tb-output
      for(int i0 = 0; i0 < 9; i0++) {
        fout << layer17_out[i0].read() << " ";
      }
      fout << std::endl;
      }

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_16; i++) {
          std::cout << pr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
      }
    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
    //hls-fpga-machine-learning insert zero
model_weightdefault_t w7[73728];
model_weightdefault_t w11[147456];
    for(int iX = 0; iX < 11; iX++){
    input_t pTest = 0;
    hls::stream<input_t> input_6[N_INPUT_3_1];
    for(int i0 = 0; i0 < 16*16; i0++) {
    for(int i2 = 0; i2 < 3+1; i2++) {
    if (i2 == 0) input_6[i2].write(pTest);
    if (i2 > 0 && i0 < 1) input_6[i2].write(iX + 1);
    if (i2 > 0 && i0 > 0) input_6[i2].write(iX + 32);
    if (pTest == 0) pTest = 1;
    }
    }
    hls::stream<result_t> layer17_out[N_FILT_7];

    //hls-fpga-machine-learning insert top-level-function
    unsigned short size_in1,size_out1;
    myproject(input_6,layer17_out,w7, w11,size_in1,size_out1);
    std::printf("Iteration %d\n", iX);
    //hls-fpga-machine-learning insert output

    //hls-fpga-machine-learning insert tb-output
    for(int i1 = 0; i1 < 64; i1++){
    for(int i0 = 0; i0 < N_FILT_7; i0++) {
      fout << layer17_out[i0].read() << " ";
    }
    }
    fout << std::endl;
   
    for(int i0 = 0; i0 < N_INPUT_3_1; i0++) {
      std::cout << input_6[i0].empty() << " ";
    }
    std::cout << std::endl;

    }
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
