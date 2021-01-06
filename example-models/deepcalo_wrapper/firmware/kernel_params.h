#include "ap_fixed.h"
#include "parameters.h"

//how many consecutive sets of inputs to run over per kernel execution
#define COMPRESSION 32
#define STREAMSIZE 1
#define BIGSTREAMSIZE_IN  77    // 56 * 11 * 4 / 32 (Compression)
#define BIGSTREAMSIZE_OUT 182   // 14 * 13 * 32 / 32 (Compression)
 
#define IN_STREAM_LEN  (N_INPUT_1_1*N_INPUT_2_1)
#define OUT_STREAM_LEN  OUT_HEIGHT_17*OUT_WIDTH_17

#define DATA_SIZE_IN  (N_INPUT_3_1-1)
#define DATA_SIZE_OUT  N_FILT_17

typedef ap_fixed<16,6> data_t;
typedef ap_uint<512>    bigdata_t;

#define NW1 147456
#define NW2 294912
#define NW3 589824
#define NW4 589824

