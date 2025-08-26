#ifndef CONV1D_HORIZONTAL_LAYER_H
#define CONV1D_HORIZONTAL_LAYER_H

#include "cnn_stream_top.h"

template<int IN_CH, int OUT_CH, int KERNEL_SIZE, int IN_ROWS>
void conv1d_horizontal_layer(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const weight_t weights[OUT_CH][IN_CH][KERNEL_SIZE],
    const data_t bias[OUT_CH],
    const int rows,
    const int cols)
{
#pragma HLS INLINE off

    // 用于存储每行输入数据的滑动窗口缓冲
    static data_t window_buffer[1][KERNEL_SIZE][IN_CH];
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=3
#pragma HLS RESOURCE variable=window_buffer core=RAM_2P_BRAM

    // 对权重和偏置进行分区，以实现完全并行访问
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    // 独立处理每一行
    RowLoop: for (int r = 0; r < rows; r++) {
        // 遍历列。我们多循环一次 (c <= cols) 来刷新流水线并计算最后一列的结果
        ColLoop: for (int c = 0; c <= cols; c++) {
//#pragma HLS PIPELINE II=16

            // 1. 将内部滑动窗口向左移动
            for (int ch = 0; ch < IN_CH; ch++) {
                for (int i = 0; i < KERNEL_SIZE - 1; i++) {
                    window_buffer[0][i][ch] = window_buffer[0][i + 1][ch];
                }
            }

            // 2. 将新数据读入窗口的最右侧
            // 如果 c < cols，从输入流读取
            // 如果 c == cols，送入0以进行右侧填充
            if (c < cols) {
                for (int ch = 0; ch < IN_CH; ch++) {
                    window_buffer[0][KERNEL_SIZE - 1][ch] = in.read();
                }
            } else { // 这是最后一次迭代，用于右侧填充
                for (int ch = 0; ch < IN_CH; ch++) {
                    window_buffer[0][KERNEL_SIZE - 1][ch] = 0;
                }
            }

            // 3.有足够的数据时 (c > 0)，计算输出
            // 输出列 'c_out' 是在迭代 'c = c_out + 1' 时计算的
            if (c > 0) {
                data_t calc_window[KERNEL_SIZE][IN_CH];
#pragma HLS ARRAY_PARTITION variable=calc_window complete dim=0

                // 4. 正确地构建用于卷积的窗口，并处理左侧填充
                for (int ch = 0; ch < IN_CH; ch++) {
                    // 对于第一个输出（列0），窗口最左边的元素是0填充
                    if (c == 1) {
                        calc_window[0][ch] = 0; // 左侧填充
                        calc_window[1][ch] = window_buffer[0][KERNEL_SIZE - 2][ch];
                        calc_window[2][ch] = window_buffer[0][KERNEL_SIZE - 1][ch];
                    } else { // 对于所有后续的列
                        calc_window[0][ch] = window_buffer[0][0][ch];
                        calc_window[1][ch] = window_buffer[0][1][ch];
                        calc_window[2][ch] = window_buffer[0][2][ch];
                    }
                }

                // 5. 为所有输出通道执行卷积
                for (int oc = 0; oc < OUT_CH; oc++) {
//#pragma HLS PIPELINE II=1
                    accum_t acc = bias[oc];
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        for (int ch = 0; ch < IN_CH; ch++) {
#pragma HLS UNROLL
                            acc += calc_window[kw][ch] * weights[oc][ch][kw];
                        }
                    }

                    // ReLU 激活函数
                    if (acc < 0) acc = 0;

                    // 将结果写入输出流
                    out.write((data_t)acc);
                }
            }
        }
    }
}

#endif
