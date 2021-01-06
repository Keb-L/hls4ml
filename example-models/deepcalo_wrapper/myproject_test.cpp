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
// #include "firmware/myproject.h"
#include "firmware/alveo_hls4ml.h"
#include "firmware/weights/w31.h"
#include "firmware/weights/w36.h"
#include "firmware/weights/w40.h"
#include "firmware/weights/w44.h"
#include "firmware/parameters.h"

#define CHECKPOINT 5000
#define ITERATIONS 3

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

  std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
  //hls-fpga-machine-learning insert zero

  hls::stream<input_t> em_barrel[STREAMSIZE][N_INPUT_3_1];
  hls::stream<result_t> layer54_out[STREAMSIZE][DATA_SIZE_OUT + 1];
  //  #pragma HLS STREAM variable = layer54_out depth = 5000 dim = 1

  bigdata_t embarrel_bigbuf[BIGSTREAMSIZE_IN * STREAMSIZE];
  bigdata_t output_bigbuf[BIGSTREAMSIZE_OUT * STREAMSIZE];

  // Repeat for three iterations
  for (int k = 0; k < ITERATIONS; k++)
  {

    // Pre-generate dataset
    for (int iX = 0; iX < STREAMSIZE; iX++)
    {
      input_t pTest = 0;

      for (int i0 = 0; i0 < 11 * 56; i0++)
      {
        for (int i2 = 0; i2 < DATA_SIZE_IN; i2++) // N_INPUT_3_1
        {
          // if (i2 == 0)
          //   em_barrel[iX][i2].write(pTest);
          if (i2 >= 0 && i0 < 1)
            em_barrel[iX][i2].write(iX + 1);
          if (i2 >= 0 && i0 > 0)
            em_barrel[iX][i2].write(iX + 1);
          // if (pTest == 0)
          //   pTest = 1;
        }
      }
    }

    // stream to bigdata buffer
    for (int i = 0; i < STREAMSIZE; i++)
    {
      // Write from streams to bigdata buffer
      bigdata_t tmp;
      for (int i0 = 0; i0 < IN_STREAM_LEN; i0++)
      {
        for (int i1 = 0; i1 < DATA_SIZE_IN; i1++)
        {
#pragma HLS UNROLL
          // subindex in tmp
          int ib = (i0 * DATA_SIZE_IN + i1) % COMPRESSION;
          // index in the output buffer
          int ia = i * BIGSTREAMSIZE_IN + ((i0 * DATA_SIZE_IN + i1) / COMPRESSION);
          // std::cout << "reading from ch[" << i1 << "]  ib:" << ib << " and ia:" << ia << std::endl;

          input_t tmp_in = em_barrel[i][i1].read();

          tmp((ib + 1) * 16 - 1, (ib)*16) = tmp_in.range(15, 0);
          if (((i0 * DATA_SIZE_IN + i1) % COMPRESSION == COMPRESSION - 1) || (i0 == OUT_STREAM_LEN - 1 && i1 == DATA_SIZE_IN - 1))
            embarrel_bigbuf[ia] = tmp;
        }
      }

      for (int i1 = 0; i1 < DATA_SIZE_IN; i1++)
        std::cout << "em_barrel [" << i << "][" << i1 << "] empty =" << em_barrel[i][i1].empty() << std::endl;
    }

    std::cout << "bigbuf complete" << std::endl;

    const bigdata_t *embarrel_ptr = embarrel_bigbuf;
    const model_default_t *in_weights_1 = w31;
    const model_default_t *in_weights_2 = w36;
    const model_default_t *in_weights_3 = w40;
    const model_default_t *in_weights_4 = w44;
    // bigdata_t *output_ptr;

    // unsigned short size_in1, size_out1;
    // myproject(em_barrel, layer54_out, w31, w36, w40, w44);
    alveo_hls4ml(embarrel_ptr, in_weights_1, in_weights_2, in_weights_3, in_weights_4, output_bigbuf);

    // Big buffer to stream
    std::cout << "post wrapper" << std::endl;
    for (int i = 0; i < STREAMSIZE; i++)
    {
      for (int i0 = 0; i0 < OUT_STREAM_LEN; i0++)
      {
        for (int i1 = 0; i1 < DATA_SIZE_OUT; i1++)
        {
#pragma HLS UNROLL
          int ib = (i0 * DATA_SIZE_OUT + i1) % COMPRESSION;
          int ia = i * BIGSTREAMSIZE_OUT + ((i0 * DATA_SIZE_OUT + i1) / COMPRESSION);
          result_t tmp;
          tmp.range(15, 0) = output_bigbuf[ia].range(16 * (ib + 1) - 1, 16 * ib);
          int chan = ((i0 * DATA_SIZE_OUT + i1) % DATA_SIZE_OUT) + 1;

          // std::cout << "writing to ch[" << chan << "]  ib:" << ib << " and ia:" << ia << std::endl;

          if (chan == 1)
            layer54_out[i][0].write((result_t)(i0 == 0 ? 0 : 1));
          // TODO: Add conditional for writing tmp
          layer54_out[i][chan].write(tmp);
        }
      }
    }

    for (int i = 0; i < STREAMSIZE; i++)
    {
      for (int i1 = 0; i1 < OUT_STREAM_LEN; i1++)
      {
        for (int i0 = 0; i0 < DATA_SIZE_OUT + 1; i0++)
        {
          fout << layer54_out[i][i0].read() << " ";
        }
        fout << std::endl;
      }

      // std::cout << "input1 ";
      // for (int i0 = 0; i0 < 5; i0++)
      // {
      //   std::cout << " " << em_barrel[i][i0].empty();
      //   if (!em_barrel[i][i0].empty())
      //     std::cout << "--> " << em_barrel[i][i0].read();
      // }
      // std::cout << std::endl;

      std::cout << "Layer Output ";
      for (int i0 = 0; i0 < DATA_SIZE_OUT; i0++)
      {
        std::cout << " " << layer54_out[i][i0].empty();
      }
      std::cout << std::endl;
      std::cout << "----> Done " << std::endl;
    }
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
