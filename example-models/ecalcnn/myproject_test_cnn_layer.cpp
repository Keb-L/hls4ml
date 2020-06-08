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

#include "firmware/parameters_cnn_layer.h"
#include "firmware/myproject_cnn_layer.h"


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
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  int e = 0;

  static const unsigned output_size = OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6;
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
        input_t em_barrel[N_INPUT_1_1][N_INPUT_2_1][N_INPUT_3_1];
        for(int i1=0; i1<N_INPUT_1_1; i1++){
          for(int i2=0; i2<N_INPUT_2_1; i2++){
            for(int i3=0; i3<N_INPUT_3_1; i3++){
              em_barrel[i1][i2][i3] = (input_t)in[i1*N_INPUT_2_1*N_INPUT_3_1+i2*N_INPUT_3_1+i3];
            }
          }
        } 
      result_t  preds[output_size];

      myproject_cnn_layer(em_barrel, preds);
      
      //hls-fpga-machine-learning insert tb-output
      for(int i = 0; i < output_size; i++) {
        fout << preds[i] << " ";
      }
      fout << std::endl;

    }
    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
    //hls-fpga-machine-learning insert zero
    input_t em_barrel[N_INPUT_1_1][N_INPUT_2_1][N_INPUT_3_1] = {1};

    result_t  preds[output_size];

    //hls-fpga-machine-learning insert top-level-function
    unsigned short size_in1,size_out1;
    //myproject(em_barrel,layer52_out,size_in1,size_out1);
    myproject_cnn_layer(em_barrel,preds);

    //hls-fpga-machine-learning insert output
    for(int i = 0; i < output_size; i++) {
      std::cout << preds[i] << " ";
    }
    std::cout << std::endl;

    //hls-fpga-machine-learning insert tb-output
    for(int i = 0; i < output_size; i++) {
      fout << preds[i] << " ";
    }
    fout << std::endl;
  }
  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}

  // input_t   gpu_0_data_0[N_INPUT_1_1][N_INPUT_2_1][N_INPUT_3_1];
  // //result_t  preds[N_LAYER_50];
  // result_t  preds[OUT_WIDTH_38*OUT_HEIGHT_38*N_FILT_38];

  // model_default_t w38[config38::mult_config::n_in*config38::mult_config::n_out];
  // model_default_t w42[config42::n_in*config42::n_out];
  // //hls-fpga-machine-learning insert top-level-function
  // //bool iReset = true;
  // //  subimage(iReset, gpu_0_data_0,layer176_out);
  // myproject(gpu_0_data_0,preds,w38,w42);
  
  // //hls-fpga-machine-learning insert output
  // for(int i = 0; i < N_LAYER_50; i++) {
  //   fout << preds[i] << " ";
  // }
  // fout << std::endl;

