#ifndef MAX_POOLING1D_H
#define MAX_POOLING1D_H

#include "cnn_stream_top.h"

template<int CH, int POOL_SIZE, int COLS>
void max_pooling1d(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const int rows,
    const int cols)
{
#pragma HLS INLINE off

    // 行缓冲存储前一行数据
	data_t line_buffer[COLS][CH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2

    // 处理每两行（一个池化窗口）
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=1

            // 读取当前列的所有通道
            data_t current_col[CH];
            for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
                current_col[ch] = in.read();
            }

            // 偶数行：存储数据到行缓冲
            if (r % POOL_SIZE == 0) {
                for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
                    line_buffer[c][ch] = current_col[ch];
                }
            }
            // 奇数行：与缓冲数据比较并输出最大值
            else {
                for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
                    data_t max_val = line_buffer[c][ch];

                    // 比较并获取最大值
                    if (current_col[ch] > max_val) {
                        max_val = current_col[ch];
                    }

                    // 输出池化结果
                    out.write(max_val);
                }
            }
        }
    }
}

#endif
