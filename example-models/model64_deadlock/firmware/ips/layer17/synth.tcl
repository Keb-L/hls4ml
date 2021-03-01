set arg_0 "-I . -DN_1=N_LAYER_16 -DN_2=N_LAYER_16"
set arg_1 "-DCONFIG=softmax_config17"
set arg_2 "-DINPUT_T=layer16_t -DLAYER_T=result_t"
set args "$arg_0 $arg_1 $arg_2"
set layer_type softmax_stream


source ../common/build.tcl

