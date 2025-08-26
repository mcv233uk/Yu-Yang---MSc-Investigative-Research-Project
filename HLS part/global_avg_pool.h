#ifndef GLOBAL_AVG_POOL_H
#define GLOBAL_AVG_POOL_H

#include "cnn_stream_top.h"
#include "types.h"

template<int CH>
void global_avg_pool(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const int rows,
    const int cols)
{
#pragma HLS INLINE off

    // 用于累加每个通道总和的寄存器
    accum_t channel_sums[CH];
#pragma HLS ARRAY_PARTITION variable=channel_sums complete dim=1

    // 初始化累加器为0
    InitLoop: for (int i = 0; i < CH; i++) {
#pragma HLS UNROLL
        channel_sums[i] = 0;
    }

    const int total_pixels = rows * cols;

    // 遍历所有输入像素并进行累加
    // 循环次数为 rows * cols，每次读取一个像素（包含所有通道）
    AccumLoop: for (int i = 0; i < total_pixels; i++) {
//#pragma HLS PIPELINE II=1
        // 一次性读取一个像素的所有通道数据
        for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
            data_t pixel_val = in.read();
            channel_sums[ch] += pixel_val;
        }
    }

    // 所有像素处理完毕后，计算平均值并输出
    AvgLoop: for (int ch = 0; ch < CH; ch++) {
//#pragma HLS PIPELINE II=1
        // 计算平均值
        accum_t avg_val = channel_sums[ch] / total_pixels;
        // 写入输出流
        out.write((data_t)avg_val);
    }
}

#endif
