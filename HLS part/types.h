// types.h
#ifndef TYPES_H
#define TYPES_H

#include <ap_fixed.h>
#include <ap_int.h>  // ��� ap_int ͷ�ļ�

typedef ap_fixed<32, 11>  data_t;
typedef ap_fixed<32, 11>  weight_t;
typedef ap_fixed<32, 16> accum_t;
typedef ap_fixed<32, 8> norm_param_t;

// ��� AXI-Stream �ӿڽṹ��
struct axis_stream {
    data_t data;
    ap_uint<1> last;  // TLAST �ź�
};

#endif
