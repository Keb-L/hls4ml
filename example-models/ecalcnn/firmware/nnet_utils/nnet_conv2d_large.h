#ifndef NNET_CONV2D_LARGE_H_
#define NNET_CONV2D_LARGE_H_

#include "ap_shift_reg.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_batchnorm.h"
#include "nnet_conv2d.h"
#include "nnet_dense_large.h"
#include "nnet_dense_large_stream.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void im2col_2d(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width])
{
    const int output_h = (CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom -
        (CONFIG_T::dilation_height * (CONFIG_T::filt_height - 1) + 1)) / CONFIG_T::stride_height + 1;
    const int output_w = (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right -
        (CONFIG_T::dilation_width * (CONFIG_T::filt_width - 1) + 1)) / CONFIG_T::stride_width + 1;
    const int channel_size = CONFIG_T::in_height * CONFIG_T::in_width;

    for (int channel = CONFIG_T::n_chan; channel--; data += channel_size) {
        for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
            for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
                int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (input_row < 0 || input_row > CONFIG_T::in_height) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                                *(data_col++) = data[input_row * CONFIG_T::in_width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += CONFIG_T::stride_width;
                        }
                    }
                    input_row += CONFIG_T::stride_height;
                }
            }
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_full(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    data_T data_conv[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width];
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    //#pragma HLS ARRAY_PARTITION variable=data_conv complete
    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete
    im2col_2d<data_T, CONFIG_T>(data, data_conv);
    for (int i = 0; i < CONFIG_T::out_height * CONFIG_T::out_width; i++) {
        for (int j = 0; j < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; j++) {
            data_col[j] = data[j * CONFIG_T::out_height * CONFIG_T::out_width + i];
        }
        dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            //res[i * CONFIG_T::n_filt + j] = res_col[j];
            res[j * CONFIG_T::out_height * CONFIG_T::out_width + i] = res_col[j]; // Transposed order
        }
    }
}

template<class data_T, typename CONFIG_T>
void im2col_2d_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    data_T data_col[CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width],
    const int row,
    const int col)
{
    const int channel_size = CONFIG_T::in_height * CONFIG_T::in_width;
    int index = 0;
    for (int channel = CONFIG_T::n_chan; channel--; data += channel_size) {
        #pragma HLS UNROLL
        for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
            int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;
            for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
                if (input_row < 0 || input_row > CONFIG_T::in_height) {
                    data_col[index++] = 0;
                } else {
                    int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;
                    if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                        //*(data_col++) = data[input_row * CONFIG_T::in_width + input_col];
                        data_col[index++] = data[input_row * CONFIG_T::in_width + input_col];
                    } else {
                        //*(data_col++) = 0;
                        data_col[index++] = 0;
                    }
                    input_col += CONFIG_T::stride_width;
                }
            }
            input_row += CONFIG_T::stride_height;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);

    //#pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE         variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    //#pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete
    
    HeightLoop:
    for (int i = 0; i < CONFIG_T::out_height; i++) {
        WidthLoop:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            im2col_2d_cf<data_T, CONFIG_T>(data, data_col, i, j);
            dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
            FiltLoop:
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                //res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
                res[k * CONFIG_T::out_height * CONFIG_T::out_width + i * CONFIG_T::out_width + j] = res_col[k]; // Transposed order
            }
        }
    }
}
template<class data_T, typename CONFIG_T>
void im2col_2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    const int row,
    const int col)
{
  //#pragma HLS inline REGION
  static const int nchan  = CONFIG_T::n_chan;
  static const int filth  = CONFIG_T::filt_height-CONFIG_T::pad_top;
  static const int filtw  = CONFIG_T::filt_width;
  static const int padt   = -CONFIG_T::pad_top; 
  static const int padl   = -CONFIG_T::pad_left; 
  static const int height = CONFIG_T::in_height; 
  static const int width  = CONFIG_T::in_width; 
  int shifth = row * CONFIG_T::stride_height;
  int shiftw = col * CONFIG_T::stride_width;
  unsigned index = 0;
  data_T temp    = 0; 
  int input_row  = 0;
  int input_col  = 0;
  int dataindex  = 0;

  //for (int channel = CONFIG_T::n_chan; channel--; data++) {  
  for (int channel = 0; channel < nchan; channel++) {  
    #pragma HLS UNROLL
    for (int kernel_row =  0; kernel_row < filth; kernel_row++) {//add dilation
      #pragma HLS UNROLL
      input_row = padt+kernel_row+shifth;
      for (int kernel_col = 0; kernel_col < filtw; kernel_col++) {
        #pragma HLS UNROLL
	input_col = padl + kernel_col + shiftw;
	if (input_row < 0 || input_row >= height || input_col < 0 || input_col >= width) {
	  temp=0; 
	} else {
	  dataindex=input_row * width * nchan + input_col * nchan + channel;
	  temp =  data[dataindex];
	}
	data_col[index] = temp;
	index++;
      }
    }
  }
}
//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image(unsigned iShiftX,unsigned iShiftY,
		  data_T input[CONFIG_T::n_filt],
  		  res_T  data [CONFIG_T::out_height*CONFIG_T::out_width * CONFIG_T::n_chan]) { 
    #pragma HLS PIPELINE
    const unsigned lShift = (iShiftY/CONFIG_T::stride_height)*CONFIG_T::out_width*CONFIG_T::n_chan+(iShiftX/CONFIG_T::stride_width)*CONFIG_T::n_chan;
    for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
      data[lShift+i2] = input[i2];
    }
}
//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image_2d(unsigned iShiftX,unsigned iShiftY,
		     data_T input[CONFIG_T::n_filt],
		     res_T  data [CONFIG_T::out_width][CONFIG_T::filt_height][CONFIG_T::n_filt]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  //unsigned lX = iShiftX*CONFIG_T::stride_width;
  //unsigned lY = iShiftY*CONFIG_T::stride_height;
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
    data[iShiftX][iShiftY][i2] = input[i2];
  }
}
//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image_2dFull(unsigned iShiftX,unsigned iShiftY,
		     data_T input[CONFIG_T::n_filt],
		     res_T  data [CONFIG_T::out_width][CONFIG_T::out_height][CONFIG_T::n_filt]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  //unsigned lX = iShiftX*CONFIG_T::stride_width;
  //unsigned lY = iShiftY*CONFIG_T::stride_height;
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
    data[iShiftX][iShiftY][i2] = input[i2];
  }
}
//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image_2d2(unsigned iShiftX,unsigned iShiftY,
		     data_T input[CONFIG_T::n_filt],
		     res_T  data [CONFIG_T::out_height][CONFIG_T::out_width][CONFIG_T::n_filt2]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  //unsigned lX = iShiftX*CONFIG_T::stride_width;
  //unsigned lY = iShiftY*CONFIG_T::stride_height;
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
    data[iShiftX][iShiftY][i2] = input[i2];
  }
}
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image_2dS(unsigned iShiftX,
		     data_T input[CONFIG_T::n_filt],
		     res_T  data [CONFIG_T::out_width][CONFIG_T::n_filt]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  //unsigned lX = iShiftX*CONFIG_T::stride_width;
  //unsigned lY = iShiftY*CONFIG_T::stride_height;
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
    data[iShiftX][i2] = input[i2];
  }
}
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image_2dS1(
		       data_T input[CONFIG_T::n_filt],
		       hls::stream<res_T>  data [CONFIG_T::n_filt]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
   #pragma HLS UNROLL
    data[i2].write(input[i2]);
  }
}
template<class data_T, class res_T, typename CONFIG_T>
  void fill_image_2dS2(
		       data_T input[CONFIG_T::n_filt],
		       hls::stream<res_T>  data[1]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
   data[0].write(input[i2]);
  }
}
template<class data_T, class res_T, typename CONFIG_T>
  void clone(	       data_T input[CONFIG_T::n_in],
		       res_T  data [CONFIG_T::n_out]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  static unsigned min = MIN(CONFIG_T::n_in,CONFIG_T::n_out);
  for(unsigned i2 = 0; i2 < min; i2++) {
   #pragma HLS UNROLL
    data[i2] = input[i2]+i2;
  }
}

template<class data_T, class res_T, typename CONFIG_T>
  void clone_db(	       data_T input[2*CONFIG_T::n_in],
			       res_T  data [CONFIG_T::n_out]) { //CONFIG_T::n_filt2
  #pragma HLS PIPELINE
  static int id = 0; 
  static unsigned min = MIN(CONFIG_T::n_in,CONFIG_T::n_out);
  for(unsigned i2 = 0; i2 < min; i2++) {
   #pragma HLS UNROLL
    data[i2] = input[id*CONFIG_T::n_in+i2]+i2;
  }
  id++;
  if(id == 2) id = 0; 
}

//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
void fill_entry(unsigned iShift,
		data_T input[CONFIG_T::n_chan],
		res_T  data [CONFIG_T::filt_width   * CONFIG_T::out_width * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  static const unsigned lShift = iShift*CONFIG_T::filt_height*CONFIG_T::n_chan+(CONFIG_T::filt_height-1)*CONFIG_T::n_chan;
  for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) {
    data[lShift+i2] = input[i2];
  }
}

//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
void fill_entry_2d(unsigned iShift,
		   data_T input[CONFIG_T::n_chan],
		   res_T  data [CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  static const unsigned lChan = CONFIG_T::filt_height-1; 
  for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) {
    if(iShift < CONFIG_T::out_width) data[iShift][lChan][i2] = input[i2];
  }
}
// //Fills the temporary array to be fed in the CNN 
// template<class data_T, class res_T, typename CONFIG_T>
// void shift_right_small(//To be fixed with stride
// 		 data_T input[CONFIG_T::filt_height * CONFIG_T::n_chan],
//       res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
//   //Shift register by image height
//   for(int i0 = 0; i0 < CONFIG_T::filt_width-1; i0++) { 
//     #pragma HLS PIPELINE II=1
//     for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
//       for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
// 	data[i0*CONFIG_T::filt_height*CONFIG_T::n_chan+i1*CONFIG_T::n_chan+i2] = data[(i0+1)*CONFIG_T::filt_height*CONFIG_T::n_chan+i1*CONFIG_T::n_chan+i2];
//       }
//     }
//   }
//   static const unsigned lastheight=(CONFIG_T::filt_width-1)*(CONFIG_T::filt_height)*CONFIG_T::n_chan;
//   for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
//     #pragma HLS UNROLL
//     for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
//       data[lastheight+i1*CONFIG_T::n_chan+i2] = input[i1*CONFIG_T::n_chan+i2];
//     }
//   }
// }

//with stride
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_small(//To be fixed with stride
			      data_T input[CONFIG_T::filt_height][CONFIG_T::n_chan],
			      res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  
  //Shift register by image height
  static const int filt_width = CONFIG_T::filt_width-1;
  for(int i0 = 0; i0 < filt_width; i0++) { 
    #pragma HLS PIPELINE II=1
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i0*CONFIG_T::n_chan+i2] = data[i1*CONFIG_T::filt_width*CONFIG_T::n_chan+(i0+1)*CONFIG_T::n_chan+i2];
      }
    }
  }
  static const int lastheight=(CONFIG_T::filt_width-1)*CONFIG_T::n_chan;
  for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
   #pragma HLS UNROLL
    for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
     data[lastheight+i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i2] = input[i1][i2];
    }
  }
}


//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_small_2d(//To be fixed with stride
			  data_T input[CONFIG_T::filt_height][CONFIG_T::n_chan],
			  res_T  data[CONFIG_T::filt_width  ][CONFIG_T::filt_height][CONFIG_T::n_chan]) { 
  //Shift register by image height
  for(int i0 = 0; i0 < CONFIG_T::filt_width-1; i0++) { 
    #pragma HLS PIPELINE II=1
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[i0][i1][i2] = data[(i0+1)][i1][i2];
      }
    }
  }
  for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
    #pragma HLS UNROLL
    for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
      data[(CONFIG_T::filt_width-1)][i1][i2] = input[i1][i2];
    }
  }
}
//Fills the temporary array to be fed in the CNN 
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_small_2dX(//To be fixed with stride
			   data_T input[CONFIG_T::filt_height][CONFIG_T::n_chan],
			   res_T  data [CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan]) { 
  static const int lW=CONFIG_T::n_chan;
  static const int lH=CONFIG_T::filt_height*CONFIG_T::n_chan;
  static const int lLast=CONFIG_T::filt_width-1;
  static const int lWLast=lLast*lW;
  //Shift register by image height
  for(int i0 = 0; i0 < lLast; i0++) { 
    #pragma HLS PIPELINE II=1
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) {
	data[i1*lH+i0*lW+i2] = data[i1*lH+(i0+1)*lW+i2]; 
      }
    }
  }
  for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
    #pragma HLS UNROLL
    for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
      data[i1*lH+lW*lWLast+i2] = input[i1][i2];
    }
  }
}
template<class data_T, class res_T, typename CONFIG_T>
void shift_right(unsigned iShift,
		 data_T input[CONFIG_T::out_height*CONFIG_T::filt_height * CONFIG_T::n_chan],
                 res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) {
  #pragma HLS PIPELINE
  const unsigned shift=iShift*CONFIG_T::filt_height * CONFIG_T::n_chan;
  data_T tmpinput[CONFIG_T::filt_height * CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(int i0 = 0; i0 < CONFIG_T::filt_height; i0++) { 
   for(int i1 = 0; i1 < CONFIG_T::n_chan; i1++) { 
      tmpinput[i0*CONFIG_T::n_chan+i1] = input[shift+i0*CONFIG_T::filt_height+i1];
    }
  } 
  shift_right_small<data_T,res_T,CONFIG_T>(tmpinput,data);
}
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_2d(unsigned iShift,
		    data_T input[CONFIG_T::out_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
                    res_T  data[CONFIG_T::filt_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) {
  #pragma HLS PIPELINE
  data_T tmpinput[CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(int i0 = 0; i0 < CONFIG_T::filt_height; i0++) { 
   for(int i1 = 0; i1 < CONFIG_T::n_chan; i1++) { 
     if(i0+iShift < CONFIG_T::out_width) { 
       tmpinput[i0][i1] = input[iShift][i0][i1];
     } else { 
       tmpinput[i0][i1] = 0;
     }
    }
  } 
  shift_right_small_2d<data_T,res_T,CONFIG_T>(tmpinput,data);
}
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_2dX(unsigned iShift,
		     data_T input[CONFIG_T::out_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
		     res_T  data[CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan]) {
 #pragma HLS PIPELINE
  data_T tmpinput[CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(int i0 = 0; i0 < CONFIG_T::filt_height; i0++) { 
   for(int i1 = 0; i1 < CONFIG_T::n_chan; i1++) { 
     if(i0+iShift < CONFIG_T::out_width) { 
       tmpinput[i0][i1] = input[iShift][i0][i1];
     } else { 
       tmpinput[i0][i1] = 0;
     }
    }
  } 
  shift_right_small_2dX<data_T,res_T,CONFIG_T>(tmpinput,data);
}
template<class data_T,unsigned iArr>
void zeros(data_T input[iArr]) { 
  for(unsigned i0 = 0; i0 < iArr; i0++) { 
    #pragma HLS UNROLL
    input[i0] = 0; 
  }
}
template<class data_T,unsigned iArr0,unsigned iArr1,unsigned iArr2>
void zeros(data_T input[iArr0][iArr1][iArr2]) { 
  #pragma HLS PIPELINE
  for(unsigned i0 = 0; i0 < iArr0; i0++) { 
    for(unsigned i1 = 0; i1 < iArr1; i1++) { 
      for(unsigned i2 = 0; i2 < iArr2; i2++) { 
       #pragma HLS UNROLL
       input[i0][i1][i2] = 0; 
      }
    }
  }
}

template<class data_T, class res_T, typename CONFIG_T>
void shift_left_small_KL(
            data_T input[CONFIG_T::filt_height][CONFIG_T::n_chan],
            res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]
) { 
  static const unsigned offsetH = CONFIG_T::filt_width*CONFIG_T::n_chan;
  static const unsigned offsetW = CONFIG_T::n_chan;

  for(unsigned i0 = 0; i0 < CONFIG_T::filt_width-1; i0++) { 
    #pragma HLS PIPELINE II=1
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
        data[i1*offsetH + i0*offsetW + i2] = data[i1*offsetH + (i0+1)*offsetW + i2];
      }
    }
  }

  // Shift in the new values
  static const unsigned lastwidth = CONFIG_T::filt_width-1;
  for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
   #pragma HLS UNROLL
   for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
     data[i1*offsetH + lastwidth*offsetW + i2] = input[i1][i2];
   }
  }
}


//with stride
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_small_stride(//To be fixed with stride
			      data_T input[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
			      res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  
  //Shift register by image height
  static const int filt_width = CONFIG_T::filt_width-CONFIG_T::stride_width;
  for(int i0 = 0; i0 < filt_width; i0++) { 
    #pragma HLS PIPELINE II=1
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	      data[i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i0*CONFIG_T::n_chan+i2] = data[i1*CONFIG_T::filt_width*CONFIG_T::n_chan+(i0+1)*CONFIG_T::n_chan+i2];
      }
    }
  }
  static const int lastheight=(CONFIG_T::filt_width-CONFIG_T::stride_width)*CONFIG_T::n_chan;
  for(int i0 = 0; i0 < CONFIG_T::stride_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
     #pragma HLS UNROLL
     for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       data[lastheight+i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i0*CONFIG_T::n_chan+i2] = input[i0][i1][i2];
     }
    }
  }
}

//with stride
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_small_stride_db(//To be fixed with stride
			      data_T input[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
			      res_T  data[2*CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  
  //Shift register by image height
  static const int tmp = 0;
  static const int filt_width = SUB(CONFIG_T::filt_width,2*CONFIG_T::stride_width);
  static unsigned id = 0;
  int lShift = id*CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan;
  for(int i0 = 0; i0 < filt_width; i0++) { 
    #pragma HLS PIPELINE II=1
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[lShift+i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i0*CONFIG_T::n_chan+i2] = data[lShift+i1*CONFIG_T::filt_width*CONFIG_T::n_chan+(i0+1)*CONFIG_T::n_chan+i2];
      }
    }
  }
  static const int lastheight= SUB(CONFIG_T::filt_width,2*CONFIG_T::stride_width)*CONFIG_T::n_chan;
  for(unsigned i0 = 0; i0 < CONFIG_T::stride_width; i0++) { 
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
     #pragma HLS UNROLL
     for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       data[lShift+lastheight+i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i0*CONFIG_T::n_chan+i2] = input[i0][i1][i2];
     }
    }
  }
  id++;
  if(id == 2) id = 0; 
}
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_stride(unsigned iShift,
			data_T input[CONFIG_T::out_height*CONFIG_T::filt_height * CONFIG_T::n_chan],
			res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) {
  #pragma HLS PIPELINE
  const unsigned shift=(iShift-1)*CONFIG_T::filt_height * CONFIG_T::n_chan*CONFIG_T::stride_width;
  data_T tmpinput[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan]; //note this breaks for stride larger than filter size
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  unsigned index = 0;
  for(int i0 = 0; i0 < CONFIG_T::stride_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::n_chan; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::filt_height; i2++) { 
	tmpinput[i0][i2][i1] = input[index];
	index++;
      }
    } 
  }
  shift_right_small_stride<data_T,res_T,CONFIG_T>(tmpinput,data);
}

template<class data_T, class res_T, typename CONFIG_T>
void shift_right_stride_2d(unsigned iShift,
			   data_T input[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan],
			   res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  unsigned lShift = iShift*CONFIG_T::stride_width;
  //static const unsigned lX = CONFIG_T::filt_height*CONFIG_T::n_chan;
  //static const unsigned lY = CONFIG_T::n_chan;
  data_T tmpinput[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(unsigned i0 = 0; i0 < CONFIG_T::stride_width;  i0++) {
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;    i2++) { 
        if(i0+lShift < CONFIG_T::out_width) { 
          tmpinput[i0][i1][i2] = input[lShift+i0][i1][i2];//*lX+i1*lY+i2];
        } else { 
          tmpinput[i0][i1][i2] = 0;
        }
      }
    } 
  }
  shift_right_small_stride<data_T,res_T,CONFIG_T>(tmpinput,data);
}

template<class data_T, class res_T, typename CONFIG_T>
void shift_right_stride_2dNew(unsigned iShiftX,unsigned iShiftY,
			      data_T input[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan],
			      res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  unsigned lShiftX = iShiftX*CONFIG_T::stride_width;
  unsigned lShiftY = iShiftY*CONFIG_T::stride_height-CONFIG_T::filt_height+1;
  //static const unsigned lX = CONFIG_T::filt_height*CONFIG_T::n_chan;
  //static const unsigned lY = CONFIG_T::n_chan;
  static const unsigned minwidth  = CONFIG_T::pad_left;
  static const unsigned maxwidth  = CONFIG_T::pad_left+CONFIG_T::in_width;
  static const unsigned minheight = CONFIG_T::pad_top;
  static const unsigned maxheight = CONFIG_T::pad_top+CONFIG_T::in_height;
  data_T tmpinput[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(unsigned i0 = 0; i0 < CONFIG_T::stride_width;  i0++) {
    int pX = i0+lShiftX;
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      int pY  = i1+lShiftY;
      unsigned pYC = pY % CONFIG_T::filt_height; 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;    i2++) { 
        if(pX >= minwidth && pX < maxwidth && pY >= minheight && pY < maxheight) { 
          tmpinput[i0][i1][i2] = input[pX][pYC][i2];
        } else { 
          tmpinput[i0][i1][i2] = 0;
        }
      }
    } 
  }

  shift_right_small_stride<data_T,res_T,CONFIG_T>(tmpinput,data);
}

template<class data_T, class res_T, typename CONFIG_T>
void shift_left_2d_KL(unsigned iX,unsigned iY,
            data_T input[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan],
            res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE

  static const unsigned minwidth  = CONFIG_T::pad_left;
  static const unsigned maxwidth  = CONFIG_T::pad_left+CONFIG_T::in_width;
  static const unsigned minheight = CONFIG_T::pad_top;
  static const unsigned maxheight = CONFIG_T::pad_top+CONFIG_T::in_height;

  // Shift register in one column of data.
  data_T tmpinput[CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0

  unsigned tX = iX + CONFIG_T::pad_left; // True X (including padding)
  unsigned tY = iY + CONFIG_T::pad_top;  // True Y (include padding)
  // For each element in the column
  for(unsigned ih = 0; ih < CONFIG_T::filt_height; ih++) { 
    unsigned pY  = tY + ih + 1;                  // True Y + offset pointer + 1 (The +1 is to compensate for the off-by-one issue with modulo)
    unsigned pYC = pY % CONFIG_T::filt_height;  // Modulo for position in line buffer (layer_in_row)

    // For each channel
    for(unsigned iw = 0; iw < CONFIG_T::n_chan;    iw++) { 
      // Bounds check (this should)
      // if(tX >= minwidth && tX < maxwidth && pY >= minheight && pY < maxheight) { 
      tmpinput[ih][iw] = input[tX][pYC][iw];
      // } else { // Out-of-bounds!
        // tmpinput[ih][iw] = 0;
      // }

    }
  }
  // Shift the tmpinput into the data array
  shift_left_small_KL<data_T,res_T,CONFIG_T>(tmpinput,data);
}

template<class data_T, class res_T, typename CONFIG_T>
void shift_right_stride_2dNewResnet(unsigned iShiftX,unsigned iShiftY,
				    data_T input[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan],
				    res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  unsigned lShiftX = iShiftX*CONFIG_T::stride_width;
  unsigned lShiftY = iShiftY*CONFIG_T::stride_height-CONFIG_T::filt_height+1;
  //static const unsigned lX = CONFIG_T::filt_height*CONFIG_T::n_chan;
  //static const unsigned lY = CONFIG_T::n_chan;
  static const unsigned minwidth  = CONFIG_T::pad_left;
  static const unsigned maxwidth  = CONFIG_T::pad_left+CONFIG_T::in_width;
  static const unsigned minheight = CONFIG_T::pad_top;
  static const unsigned maxheight = CONFIG_T::pad_top+CONFIG_T::in_height;
  data_T tmpinput[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(unsigned i0 = 0; i0 < CONFIG_T::stride_width;  i0++) {
    int pX = i0+lShiftX;
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      int pY  = i1+lShiftY;
      unsigned pYC = pY % CONFIG_T::filt_height; 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;    i2++) { 
	if(pX >= minwidth && pX < maxwidth && pY >= minheight && pY < maxheight) { 
	  tmpinput[i0][i1][i2] = input[pX][pYC][i2];
	} else { 
	  tmpinput[i0][i1][i2] = 0;
	}
      }
    } 
  }
  ///Weird resnet layer with stride 2 that is only one pixel (so takes bottom right)
  for(int i0 = 0; i0 < CONFIG_T::filt_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
     #pragma HLS UNROLL
     for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       data[i1*CONFIG_T::filt_width*CONFIG_T::n_chan+i0*CONFIG_T::n_chan+i2] = input[CONFIG_T::stride_width-1][i1][i2];
     }
    }
  }
}
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_stride_2dNew_db(unsigned iShiftX,unsigned iShiftY,
				 data_T input[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan],
				 res_T  data[2*CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  unsigned lShiftX = iShiftX*CONFIG_T::stride_width;
  unsigned lShiftY = iShiftY*CONFIG_T::stride_height-CONFIG_T::filt_height+1;
  //static const unsigned lX = CONFIG_T::filt_height*CONFIG_T::n_chan;
  //static const unsigned lY = CONFIG_T::n_chan;
  static const unsigned minwidth  = CONFIG_T::pad_left;
  static const unsigned maxwidth  = CONFIG_T::pad_left+CONFIG_T::in_width;
  static const unsigned minheight = CONFIG_T::pad_top;
  static const unsigned maxheight = CONFIG_T::pad_top+CONFIG_T::in_height;
  data_T tmpinput[CONFIG_T::stride_width][CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(unsigned i0 = 0; i0 < CONFIG_T::stride_width;  i0++) {
    int pX = i0+lShiftX;
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      int pY  = i1+lShiftY;
      unsigned pYC = pY % CONFIG_T::filt_height; 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;    i2++) { 
	if(pX >= minwidth && pX < maxwidth && pY >= minheight && pY < maxheight) { 
	  tmpinput[i0][i1][i2] = input[pX][pYC][i2];
	} else { 
	  tmpinput[i0][i1][i2] = 0;
	}
      }
    } 
  }
  shift_right_small_stride_db<data_T,res_T,CONFIG_T>(tmpinput,data);
}
template<class data_T, class res_T, typename CONFIG_T>
void shift_right_2dNew(unsigned iShiftX,unsigned iShiftY,
		       data_T input[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan],
		       res_T  data[CONFIG_T::filt_width   * CONFIG_T::filt_height * CONFIG_T::n_chan]) { 
  #pragma HLS PIPELINE
  static const unsigned minwidth  = CONFIG_T::pad_left-1;
  static const unsigned maxwidth  = CONFIG_T::pad_left+CONFIG_T::in_width;
  static const unsigned minheight = CONFIG_T::pad_top-1;
  static const unsigned maxheight = CONFIG_T::pad_top+CONFIG_T::in_height;
  data_T tmpinput[CONFIG_T::filt_height][CONFIG_T::n_chan];
  #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
  for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
    unsigned pY  = i1+iShiftY;
    unsigned pYC = pY % CONFIG_T::filt_height; 
    for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;    i2++) { 
      #pragma HLS UNROLL
      if(iShiftX > minwidth && iShiftX < maxwidth && pY > minheight && pY < maxheight) { 
        tmpinput[i1][i2] = input[iShiftX][pYC][i2];
      } else { 
        tmpinput[i1][i2] = 0;
      }
    }
  } 
  shift_right_small_2dX<data_T,res_T,CONFIG_T>(tmpinput,data);
}
//Fills the temporary array to be fed in the CNN
template<class data_T, class res_T, typename CONFIG_T>
void reset_down(//To be fixed with stride
  		 data_T input[CONFIG_T::out_width  * CONFIG_T::filt_height* CONFIG_T::n_chan],
                 res_T  data [CONFIG_T::filt_width * CONFIG_T::filt_height* CONFIG_T::n_chan]) { 
  //Shift register by image height
  #pragma HLS PIPELINE
  for(int i0 = 0; i0 < CONFIG_T::filt_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[i0*CONFIG_T::filt_height*CONFIG_T::n_chan+i1*CONFIG_T::n_chan+i2] = input[i0*CONFIG_T::filt_height*CONFIG_T::n_chan+i1*CONFIG_T::n_chan+i2];
      }
    }
  }
}
//Fills the temporary array to be fed in the CNN
template<class data_T, class res_T, typename CONFIG_T>
void reset_down_2d(//To be fixed with stride
                   data_T input[CONFIG_T::out_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
                   res_T  data [CONFIG_T::filt_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) { 
  //Shift register by image height
  #pragma HLS PIPELINE
  for(int i0 = 0; i0 < CONFIG_T::filt_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       data[i0][i1][i2] = input[i0][i1][i2];
      }
    }
  }
}
//Fills the temporary array to be fed in the CNN
template<class data_T, class res_T, typename CONFIG_T>
void reset_down_2dX(//To be fixed with stride
                   data_T input[CONFIG_T::out_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
                   res_T  data [CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan]) { 
  static const unsigned lW = CONFIG_T::filt_height*CONFIG_T::n_chan;
  static const unsigned lH = CONFIG_T::n_chan;
  
  //Shift register by image height
  #pragma HLS PIPELINE
  for(int i0 = 0; i0 < CONFIG_T::filt_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       data[i0*lW+i1*lH+i2] = input[i0][i1][i2];
      }
    }
  }
}
//Fills the temporary array to be fed in the CNN
template<class data_T, class res_T, typename CONFIG_T>
  void reset_down_2dXNew(unsigned iY,
			 data_T input[CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
			 res_T  data [CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan]) { 
  static const unsigned lW = CONFIG_T::n_chan;
  static const unsigned lH = CONFIG_T::filt_width*CONFIG_T::n_chan;
  unsigned lY              = iY-(CONFIG_T::filt_height-1);//*CONFIG_T::stride_height;

  //Shift register by 1 (stride_height)
  #pragma HLS PIPELINE
  for(int i0 = CONFIG_T::pad_left+1; i0 < CONFIG_T::filt_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      unsigned pYC = (i1+lY) % CONFIG_T::filt_height;
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	      data[i1*lH+i0*lW+i2] = input[i0-1][pYC][i2];
      }
    }
  }
  for(int i0 = 0; i0 < CONFIG_T::pad_left+1; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	      data[i1*lH+i0*lW+i2] = 0;
      }
    }
  }
}

//Fills the temporary array to be fed in the CNN
template<class data_T, class res_T, typename CONFIG_T>
  void reset_down_2dXNew_db(unsigned iY,
			 data_T input[CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan],
			 res_T  data [2*CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan]) { 
  static const unsigned lW = CONFIG_T::n_chan;
  static const unsigned lH = CONFIG_T::filt_width*CONFIG_T::n_chan;
  unsigned lY              = iY-(CONFIG_T::filt_height-1);//*CONFIG_T::stride_height;

  //Shift register by image height
  #pragma HLS PIPELINE
  for(int id = 0; id < 2; id++) { 
   for(int i0 = CONFIG_T::pad_left+1; i0 < CONFIG_T::filt_width; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      unsigned pYC = (i1+lY) % CONFIG_T::filt_height;
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[id*CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan+i1*lH+i0*lW+i2] = input[i0-1][pYC][i2];
      }
    }
   }
  }
  for(int id = 0; id < 2; id++) { 
   for(int i0 = 0; i0 < CONFIG_T::pad_left+1; i0++) { 
    for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[id*CONFIG_T::filt_width*CONFIG_T::filt_height*CONFIG_T::n_chan+i1*lH+i0*lW+i2] = 0;
      }
    }
   }
  }
}

//Shifts whole row
template<class data_T, class res_T, typename CONFIG_T>
void shift_down(//To be fixed with stride
  		 data_T input[CONFIG_T::out_width * CONFIG_T::n_chan],
                 res_T  data [CONFIG_T::filt_height* CONFIG_T::out_width * CONFIG_T::n_chan]) { 
  //Shift register by image height
  #pragma HLS PIPELINE 
  for(int i1 = 0; i1 < CONFIG_T::out_width; i1++) { 
    for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
      for(int i0 = 0; i0 < CONFIG_T::filt_height-1; i0++) { 
	data[i1*CONFIG_T::n_chan*CONFIG_T::filt_height+i2*CONFIG_T::filt_height+i0] = data[i1*CONFIG_T::n_chan*CONFIG_T::filt_height+i2*CONFIG_T::filt_height + i0+1];
      }
    }
  }
  for(int i1 = 0; i1 < CONFIG_T::out_width; i1++) {
    for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) {
      data[i1*CONFIG_T::n_chan*CONFIG_T::filt_height+i2*CONFIG_T::filt_height+CONFIG_T::filt_height-1] = input[i1*CONFIG_T::n_chan+i2];
    }
  }
}

//Shifts whole row
template<class data_T, typename CONFIG_T>
void shift_down_small(//To be fixed with stride
		      data_T  data [CONFIG_T::filt_height* CONFIG_T::out_width * CONFIG_T::n_chan]) {
  //Shift register by image height
  #pragma HLS PIPELINE 
  for(int i0 = 0; i0 < CONFIG_T::out_width; i0++) { 
    for(int i1 = 0; i1 < (CONFIG_T::filt_height-1); i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       data[i0*CONFIG_T::n_chan*CONFIG_T::filt_height+i1*CONFIG_T::filt_height+i2] = data[i0*CONFIG_T::n_chan*CONFIG_T::filt_height+(i1+1)*CONFIG_T::filt_height+i2];
      }
    }
  }
}
//Shifts whole row
template<class data_T, typename CONFIG_T>
void shift_down_small_2d(//To be fixed with stride
		      data_T  data [CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) {
  #pragma HLS PIPELINE 
  //Shift register by image height
  for(int i0 = 0; i0 < CONFIG_T::in_width; i0++) { 
   for(int i1 = 0; i1 < (CONFIG_T::filt_height-1); i1++) { 
     for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
      #pragma HLS UNROLL
      data[i0][i1][i2] = data[i0][i1+1][i2];
    }
   }
  }
}


//Shifts whole row
template<class data_T, typename CONFIG_T>
void shift_down_small_2d_test(//To be fixed with stride
		      data_T  data [CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) {
  //Shift register by image height
  for(int i0 = 0; i0 < CONFIG_T::in_width; i0++) { 
   for(int i1 = 0; i1 < (CONFIG_T::filt_height-1); i1++) { 
     for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
       #pragma HLS UNROLL
       data_T tmp =  data[i0][i1+1][i2];
      data[i0][i1][i2] = tmp;
    }
   }
  }
}

//Shifts whole row
template<class data_T, typename CONFIG_T>
void shift_down_small_stride(//To be fixed with stride
			     data_T  data [CONFIG_T::filt_height* CONFIG_T::out_width * CONFIG_T::n_chan]) {
  //Shift register by image height
  #pragma HLS PIPELINE 
  for(unsigned i0 = 0; i0 < CONFIG_T::out_width; i0++) { 
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height-CONFIG_T::stride_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[i0*CONFIG_T::n_chan*CONFIG_T::filt_height+i1*CONFIG_T::n_chan+i2] = data[i0*CONFIG_T::n_chan*CONFIG_T::filt_height+(i1+CONFIG_T::stride_height)*CONFIG_T::n_chan+i2];
      }
    }
  }
}
//Shifts whole row
template<class data_T, typename CONFIG_T>
void shift_down_small_stride_2d(//To be fixed with stride
				data_T  data [CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) {
  //Shift register by image height
  #pragma HLS PIPELINE 
  for(unsigned i0 = 0; i0 < CONFIG_T::in_width; i0++) { 
    for(unsigned i1 = 0; i1 < CONFIG_T::filt_height-CONFIG_T::stride_height; i1++) { 
      for(unsigned i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data[i0][i1][i2] = data[i0][i1+CONFIG_T::stride_height][i2];
      }
    }
  }
}


template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_cl(
    data_T data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);
    static const int nfilt  =  CONFIG_T::n_filt;
    static const int owidth =  CONFIG_T::out_width;
    static const int oheight =  CONFIG_T::out_height;

    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=data_col complete
    unsigned index=0;
    unsigned dataindex=0;
    for(int i0 = 0; i0 < CONFIG_T::filt_width; i0++) {  
     //This needs to be reuse factor and needs to be II of previous layer
     #pragma HLS PIPELINE II=1
     dataindex=0;
     for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
        data_col[index] = data[dataindex];
	index++;
	dataindex++;
      }
      res_T res_col[CONFIG_T::n_filt];
      #pragma HLS ARRAY_RESHAPE variable=res_col complete
      dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
      FiltLoop:
      for (int k = 0; k < CONFIG_T::n_filt; k++) {
       #pragma HLS UNROLL
       res[i1*CONFIG_T::n_chan+k] = res_col[k];
      }
     }
    }
}
template<class data_T, typename CONFIG_T>
void reset_zero_vert(
		     data_T  data [CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) {  
    for(unsigned i0  = 0; i0 < CONFIG_T::pad_left; i0++) { 
     #pragma HLS UNROLL
      for(unsigned i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
	for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;    i2++) { 
	  data[i0]                  [i1][i2]=0;
	  data[CONFIG_T::in_width-CONFIG_T::pad_left+i0][i1][i2]=0; //assumes left and right pad are the same
	}
      }
    }
}
template<class data_T, typename CONFIG_T>    
void reset_zero_horiz(
		      data_T  data [CONFIG_T::in_width][CONFIG_T::filt_height][CONFIG_T::n_chan]) {  

    for(unsigned i0 = 0; i0 < CONFIG_T::in_width; i0++) { 
      for(unsigned i1  = 0; i1 < CONFIG_T::filt_height; i1++) { 
        #pragma HLS PIPELINE II=1
	for(unsigned i2 = 0; i2 < CONFIG_T::n_chan;   i2++) { 
	 data[i0][i1][i2]     =  0;
        }
      }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_cl_stream(
		      hls::stream<data_T> data[CONFIG_T::filt_height * CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      //res_T  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
		      typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);
    static const int nfilt  =  CONFIG_T::n_filt;
    static const int owidth =  CONFIG_T::out_width;
    static const int oheight =  CONFIG_T::out_height;

    //Im2col right here with streams => Missing Stride and Padding
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=data_col complete
    unsigned index=0;
    unsigned dataindex=0;
    for(int i0 = 0; i0 < CONFIG_T::filt_width; i0++) {  
     //This needs to be reuse factor and needs to be II of previous layer
     #pragma HLS PIPELINE II=1
     dataindex=0;
     for(int i1 = 0; i1 < CONFIG_T::filt_height; i1++) { 
      for(int i2 = 0; i2 < CONFIG_T::n_chan; i2++) { 
	data_col[index] = data[dataindex].read();
	index++;
	dataindex++;
      }
      res_T res_col[CONFIG_T::n_filt];
      #pragma HLS ARRAY_RESHAPE variable=res_col complete
      dense_large<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
      FiltLoop:
      for (int k = 0; k < CONFIG_T::n_filt; k++) {
       #pragma HLS UNROLL
       res[i1*CONFIG_T::n_chan+k].write(res_col[k]);
      }
     }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream(
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      //res_T  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
		      typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
  
    #pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;  

    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static unsigned pX=0; 
    static unsigned pY=0;
    static bool pPass = false;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    }
    if(pX == lShiftX && pPass) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pY,layer_in_row,layer_in);
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//add padding
      nnet::dense_large<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights,biases);
      nnet::relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_out, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    } 
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm(bool iReset,
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      //res_T  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
		      typename CONFIG_T::bias_t   biases [CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt]
		      ) {

  //#pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static int pX=0; 
    static int pY=0;
    if(iReset) { 
      pX = 0; 
      pY = 0; 
    }
    static bool pPass = false;    
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 
    if(pX == lShiftX && pPass) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pY,layer_in_row,layer_in);
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//add padding
      nnet::dense_large<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights,biases);
      nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      nnet::relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    }
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm_nobias(bool iReset,
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt]
		      ) {

  //#pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static int pX=0; 
    static int pY=0;
    if(iReset) { 
      pX = 0; 
      pY = 0; 
    }
    static bool pPass = false;    
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 
    if(pX == lShiftX && pPass) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pY,layer_in_row,layer_in);
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//add padding
      nnet::dense_large_nobias<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights);
      nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      nnet::relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    }
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
}
template<class data_T, class res_T, typename CONFIG_T>
  void cnnshift(hls::stream<data_T> data[CONFIG_T::n_chan_in],
		ap_shift_reg<data_T, (CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right)> layer_in_row[(CONFIG_T::filt_height)-1][CONFIG_T::n_chan],
		data_T output[(CONFIG_T::filt_height*CONFIG_T::filt_width)*(CONFIG_T::n_chan)]) { 

    #pragma HLS PIPELINE
    const static int rowsize = (CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right);
    
    static const unsigned nchan = CONFIG_T::n_chan;
    data_T tmpinput[CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
    
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      data_T base = data[i0].read(); // +1 for reset case
      tmpinput[CONFIG_T::filt_height-1][i0] = base;
      for(unsigned i1 = 1; i1 < CONFIG_T::filt_height; i1++) {
        #pragma HLS UNROLL
        data_T tmp1      = tmpinput[CONFIG_T::filt_height-i1][i0];
        data_T tmp       = layer_in_row[i1-1][i0].shift(tmp1);
        tmpinput[CONFIG_T::filt_height-i1-1][i0] = tmp;
      }
    }
    shift_right_small<data_T,res_T,CONFIG_T>(tmpinput,output);
}

template<class data_T, class res_T, typename CONFIG_T>
  void cnnshiftzero(
		    ap_shift_reg<data_T, (CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right)> layer_in_row[(CONFIG_T::filt_height)-1][CONFIG_T::n_chan],
		    data_T output[(CONFIG_T::filt_height*CONFIG_T::filt_width)*(CONFIG_T::n_chan)]) { 

    #pragma HLS PIPELINE
    const static int rowsize = (CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right);
    
    static const unsigned nchan = CONFIG_T::n_chan;
    data_T tmpinput[CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=tmpinput complete dim=0
    
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      data_T base = 0;
      tmpinput[CONFIG_T::filt_height-1][i0] = base;
      for(unsigned i1 = 1; i1 < CONFIG_T::filt_height; i1++) {
        #pragma HLS UNROLL
        data_T tmp1      = tmpinput[CONFIG_T::filt_height-i1][i0];
        data_T tmp       = layer_in_row[i1-1][i0].shift(tmp1);
        tmpinput[CONFIG_T::filt_height-i1-1][i0] = tmp;
      }
    }
    shift_right_small<data_T,res_T,CONFIG_T>(tmpinput,output);
}

template<class data_T, class res_T, typename CONFIG_T>
  void fill_image(
		  data_T input[CONFIG_T::n_filt],
		  res_T  pPixId,
		  hls::stream<res_T>  data [CONFIG_T::n_filt]) { //CONFIG_T::n_filt2
      #pragma HLS PIPELINE
      for(unsigned i2 = 0; i2 < CONFIG_T::n_filt; i2++) {
      #pragma HLS UNROLL
      // if(i2 == 0) { 
      //   data[i2].write(pPixId);
      // } else { 
        data[i2].write(input[i2]); // -1 
      // }  
  }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_cl_sr(bool iReset,
            hls::stream<data_T> data[CONFIG_T::n_chan],
            hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
            typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
            typename CONFIG_T::bias_t   biases [CONFIG_T::n_filt],
            typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
            typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt],
            data_T alpha
		      ) {
  
    //#pragma HLS inline
    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;

    static ap_shift_reg<data_T, (CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right)> layer_in_row[(CONFIG_T::filt_height)-1][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=2
    
    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static int pX=0; 
    static int pY=0;
    // data_T iReset = data[0].read();
    // if(iReset==0) { 
    if(iReset) {
      pX = 0; 
      pY = 0; 
      for(int i0 = 0; i0 < CONFIG_T::pad_left; i0++) nnet::cnnshiftzero<data_T,res_T,CONFIG_T>(layer_in_row,layer_in);
    }

    static bool pPass = false;    
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
    nnet::cnnshift<data_T,res_T,CONFIG_T>(data,layer_in_row,layer_in);

    unsigned pLoop = 1;
    if(pX == CONFIG_T::in_width-1) pLoop = CONFIG_T::pad_right+1;
    for(int i0 = 0; i0 < pLoop; i0++) { 
      if(i0 > 0) nnet::cnnshiftzero<data_T,res_T,CONFIG_T>(layer_in_row,layer_in); 
      
      for(int i = 0; i < CONFIG_T::filt_height; i++) {
        for(int j = 0; j < CONFIG_T::filt_width; j++) {
          std::cout << layer_in[i*CONFIG_T::filt_width*CONFIG_T::n_chan + j * CONFIG_T::n_chan] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;


      if((i0+pX-lShiftX) % CONFIG_T::stride_width == 0 && (i0+pY-lShiftY) % CONFIG_T::stride_height == 0 && pPass) { 
        nnet::dense_large_nobias<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights);
        nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
        nnet::leaky_relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout,alpha, layer_reluout);

        for(int i = 0; i < CONFIG_T::n_filt; i++) {
          std::cout << layer_reluout[i] << " ";
        }
        std::cout << std::endl;

        res_T pPixId = 0;
        if(pX > 0 || pY > 0) pPixId = 1;
        nnet::fill_image<data_T,data_T,CONFIG_T>(layer_reluout,pPixId,res);
      }
     }
    pX = pX+1;
    if(pX == CONFIG_T::in_width) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
      for(int i0 = 0; i0 < CONFIG_T::pad_left; i0++) nnet::cnnshiftzero<data_T,res_T,CONFIG_T>(layer_in_row,layer_in);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm_leaky(bool iReset,
				     hls::stream<data_T> data[CONFIG_T::n_chan],
				     hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
				     typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
				     typename CONFIG_T::bias_t   biases [CONFIG_T::n_filt],
				     typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
				     typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt],
				     data_T alpha
		      ) {

  //#pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    // Write pointers
    static int pX=0; 
    static int pY=0;

    if(iReset) { 
      pX = 0; 
      pY = 0; 
    }

    static bool pPass = false;    
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true; // pPass = True when we have enough elements
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 

    // printf("Pointer (%d, %d)\n", pX, pY);
    // KL: ASSUMES 0 padding! Need to implement non-zero padding initialization
    // Perform left shift. This will always occur when pY >= filt_height (i.e. we have enough vertical values to evaluate)
    if (pY > lShiftY-1) nnet::shift_left_2d_KL<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in); //add padding

    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      nnet::dense_large_nobias<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights);
      nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      nnet::leaky_relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout,alpha, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    }
    
    // Update pointers
    pX = pX+1;
    if(pX == CONFIG_T::in_width) {  // Reached the end of the line. Reset and increment Y
      pX = 0;
      pY = pY+1;

      pPass = false;

      // KL: Reset the filter buffer at the EOL
      // ASSUMES 0 padding! Need to implement non-zero padding initialization
      nnet::zeros<data_T, CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan>(layer_in);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm_leaky_EOL_fix(bool iReset,
				     hls::stream<data_T> data[CONFIG_T::n_chan],
				     hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
				     typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
				     typename CONFIG_T::bias_t   biases [CONFIG_T::n_filt],
				     typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
				     typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt],
				     data_T alpha
		      ) {

  //#pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    // Write pointers
    static unsigned wX=0; 
    static unsigned wY=0;

    // Eval pointers
    static unsigned eX = 0;
    static unsigned eY = 0;

    // Eval flag
    static bool eval_en = false;   

    if(iReset) { 
      wX = 0; 
      wY = 0; 
      eX = 0;
      eY = 0;

      eval_en = false;
    }
    
    //===============================
    //  Writing to the line buffer
    //===============================
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[wX+CONFIG_T::pad_left][(CONFIG_T::pad_top+wY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 
    wX = wX + 1;

    //===============================
    //  Writing to the filter buffer
    //===============================
    // When the write pointer exceeds the lShiftY, there are enough values in line buffer column to shift into layer_in
    if (wY > lShiftY-1) nnet::shift_left_2d_KL<data_T,data_T,CONFIG_T>(eX,eY,layer_in_row,layer_in); 

    //===============================
    //    Evaluating the filter
    //===============================
    if (wY > lShiftY-1 && wX == lShiftX) eval_en = true;
    if(eval_en) { // TODO: implement stride
      printf("Evaluating convolution for (%d, %d)!\n", eX, eY);
      nnet::dense_large_nobias<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights);
      nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      nnet::leaky_relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout,alpha, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);

      eX = eX + 1;
    }
    
    //===============================
    //        Pointer update
    //===============================
    // Reached EOL for writer
    if (wX == CONFIG_T::in_width) {
      wX = 0;
      wY = wY + 1;
    }

    // Reached the EOL for evaluation
    if (eX == CONFIG_T::in_width) {
      eX = 0;
      eY = eY + 1;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm_nobias_stream(bool iReset,
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      hls::stream<typename CONFIG_T::weight_t> weights[CONFIG_T::block_factor],
		      typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt]
		      ) {

  //#pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static int pX=0; 
    static int pY=0;
    if(iReset) { 
      pX = 0; 
      pY = 0; 
    }
    static bool pPass = false;    
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 
    if(pX == lShiftX && pPass) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pY,layer_in_row,layer_in);
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//add padding
      nnet::dense_large_nobias_stream<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights);
      nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      nnet::relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    }
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
}

//Weird resnet layer that breaks all the rules
template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm_resnet(bool iReset,
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt], //Filt Width clocks to read output
		      typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::scale_t  scale  [CONFIG_T::n_filt],
		      typename CONFIG_T::norm_config::bias_t   sbiases[CONFIG_T::n_filt]
		      ) {

  //#pragma HLS inline

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static int pX=0; 
    static int pY=0;
    if(iReset) { 
      pX = 0; 
      pY = 0; 
    }
    static bool pPass = false;    
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 
    if(pX == lShiftX && pPass) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pY,layer_in_row,layer_in);
    if((pX+1) % CONFIG_T::stride_width == 0 && (pY+1) % CONFIG_T::stride_height == 0 && pPass) { 
      nnet::shift_right_stride_2dNewResnet<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//add padding
      nnet::dense_large_nobias<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights);
      nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      nnet::relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    }
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_large_stream_norm_test(
		      hls::stream<data_T> data[CONFIG_T::n_chan],
		      hls::stream<res_T>  res [CONFIG_T::n_filt]) {  //Filt Width clocks to read output

    const static int lShiftX = CONFIG_T::filt_width-CONFIG_T::pad_left-1;
    const static int lShiftY = CONFIG_T::filt_height-CONFIG_T::pad_top-1;
  
    static data_T layer_in_row[CONFIG_T::in_width+CONFIG_T::pad_left+CONFIG_T::pad_right][CONFIG_T::filt_height][CONFIG_T::n_chan];
    #pragma HLS ARRAY_RESHAPE variable=layer_in_row complete dim=3

    //static data_T layer_in[2*CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    static data_T layer_in[CONFIG_T::filt_height*CONFIG_T::filt_width*CONFIG_T::n_chan];
    //#pragma HLS ARRAY_RESHAPE variable=layer_in block factor=CONFIG_T::n_chan dim=0
    #pragma HLS ARRAY_RESHAPE variable=layer_in complete dim=0

    static res_T layer_normout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_normout complete dim=0

    static res_T layer_reluout[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_reluout complete dim=0

    static res_T layer_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_RESHAPE variable=layer_out complete dim=0

    static int pX=0; 
    static int pY=0;
    static bool pPass = false;
    if(pY > lShiftY-1 && pX == lShiftX) pPass = true;
    for(int i0 = 0; i0 < CONFIG_T::n_chan; i0++) {
      #pragma HLS UNROLL
      layer_in_row[pX+CONFIG_T::pad_left][(CONFIG_T::pad_top+pY) % CONFIG_T::filt_height][i0] =  data[i0].read();
    } 
    if(pX == lShiftX && pPass) nnet::reset_down_2dXNew<data_T,data_T,CONFIG_T>(pY,layer_in_row,layer_in);
    //std::cout << "===> " << (pX+1) % CONFIG_T::stride_width << " -- " << (pY+1) % CONFIG_T::stride_width  << " -- > " << pX << " --" << pY << " -- " << CONFIG_T::stride_width << " -- " << CONFIG_T::in_height << " -- " << CONFIG_T::n_filt << " -- " << pPass << " -- " << lShiftY << " -- " << lShiftX << " -- " << CONFIG_T::filt_height << std::endl;
    if(((pX+1) % CONFIG_T::stride_width) == 0 && ((pY+1) % CONFIG_T::stride_height) == 0 && pPass) { 
      nnet::shift_right_stride_2dNew<data_T,data_T,CONFIG_T>(pX,pY,layer_in_row,layer_in);//add padding
      //nnet::dense_large_db<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_out,weights,biases);
      nnet::clone<data_T,res_T,typename CONFIG_T::mult_config>(layer_in,layer_reluout);
      //nnet::normalize2<res_T, res_T,typename CONFIG_T::norm_config>(layer_out, layer_normout,scale,sbiases);
      //nnet::relu<res_T,res_T,typename CONFIG_T::relu_config>(layer_normout, layer_reluout);
      nnet::fill_image_2dS1<data_T,data_T,CONFIG_T>(layer_reluout,res);
    }
    pX = pX+1;
    if(pX == CONFIG_T::in_height) { 
      pX = 0;
      pY = pY+1;
      pPass = false;
    }
}

}
#endif
