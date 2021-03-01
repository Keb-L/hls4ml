set arg_0 "-I . -DN_1=2048 -DN_2=8"
set arg_1 "-DCONFIG=config16"
set arg_2 "-DINPUT_T=layer15_t -DLAYER_T=layer16_t"
set arg_3 "-DN_WEIGHTS=16384 -DWEIGHTS=w16  -DBIASES=b16"
set args "$arg_0 $arg_1 $arg_2 $arg_3"
set layer_type dense_large


source ../common/build.tcl

