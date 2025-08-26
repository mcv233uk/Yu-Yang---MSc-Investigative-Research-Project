#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "cnn_stream_top.h"
#include "types.h"

template<int IN_FEAT, int OUT_FEAT>
void fc_layer(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const weight_t weights[OUT_FEAT][IN_FEAT],
    const data_t bias[OUT_FEAT])
{
#pragma HLS INLINE off

    // 输入特征缓存
    data_t input_features[IN_FEAT];
#pragma HLS ARRAY_PARTITION variable=input_features complete dim=1

    // 读取输入特征
    for (int i = 0; i < IN_FEAT; i++) {
//#pragma HLS PIPELINE II=1
        input_features[i] = in.read();
    }

    // 计算每个输出节点
    for (int out_idx = 0; out_idx < OUT_FEAT; out_idx++) {
//#pragma HLS PIPELINE II=1

        accum_t acc = bias[out_idx];

        for (int in_idx = 0; in_idx < IN_FEAT; in_idx++) {
//#pragma HLS UNROLL
            acc += input_features[in_idx] * weights[out_idx][in_idx];
        }

        out.write((data_t)acc);
    }
}

#endif
