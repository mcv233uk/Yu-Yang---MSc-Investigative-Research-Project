// types.h
#ifndef TYPES_H
#define TYPES_H

#include <ap_fixed.h>
#include <ap_int.h>  // 添加 ap_int 头文件

typedef ap_fixed<32, 11>  data_t;
typedef ap_fixed<32, 11>  weight_t;
typedef ap_fixed<32, 16> accum_t;
typedef ap_fixed<32, 8> norm_param_t;

// 添加 AXI-Stream 接口结构体
struct axis_stream {
    data_t data;
    ap_uint<1> last;  // TLAST 信号
};

#endif
