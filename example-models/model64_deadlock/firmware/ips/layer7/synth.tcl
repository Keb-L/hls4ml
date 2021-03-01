set arg_0 "-I . -DN_INPUT=N_FILT_6 -DN_OUTPUT=N_FILT_7"
set arg_1 "-DCONFIG=config7"
set arg_2 "-DINPUT_T=layer6_t -DLAYER_T=layer10_t"
set arg_3 "-DN_WEIGHTS=73728 -DWEIGHTS=w7  -DBIASES=b7"
set args "$arg_0 $arg_1 $arg_2 $arg_3"
set layer_type conv_2d_large_channels_last_port


source ../common/build.tcl

