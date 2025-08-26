#ifndef CONV1D_VERTICAL_LAYER_H
#define CONV1D_VERTICAL_LAYER_H

#include "cnn_stream_top.h"
#include "types.h"

template<int IN_CH, int OUT_CH, int KERNEL_SIZE, int IN_COLS>
void conv1d_vertical_layer(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const weight_t weights[OUT_CH][IN_CH][KERNEL_SIZE],
    const data_t bias[OUT_CH],
    const int rows,
    const int cols)
{
#pragma HLS INLINE off

    // 行缓冲，需要存储 KERNEL_SIZE 行数据才能进行中心对齐的计算
    static data_t line_buffer[KERNEL_SIZE][IN_COLS][IN_CH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=3
#pragma HLS RESOURCE variable=line_buffer core=RAM_2P_BRAM

    // 权重和偏置分区优化
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    // 预先填充行缓冲的前 KERNEL_SIZE-1 行，用0填充第一行以处理上边界
    for (int i = 0; i < KERNEL_SIZE - 1; i++) {
        for (int c = 0; c < cols; c++) {
            for (int ch = 0; ch < IN_CH; ch++) {
                line_buffer[i][c][ch] = 0; // 顶部填充
            }
        }
    }

    // 主处理循环，遍历所有输入行
    RowLoop: for (int r = 0; r < rows; r++) {
        // 将行缓冲向上移动
        for (int i = 0; i < KERNEL_SIZE - 1; i++) {
//#pragma HLS PIPELINE
            for (int c = 0; c < cols; c++) {
                for (int ch = 0; ch < IN_CH; ch++) {
                    line_buffer[i][c][ch] = line_buffer[i + 1][c][ch];
                }
            }
        }

        // 从输入流读取新行到缓冲区的最后一行
        for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
            for (int ch = 0; ch < IN_CH; ch++) {
                 line_buffer[KERNEL_SIZE - 1][c][ch] = in.read();
            }
        }

        // 当r>=1时，line_buffer[0]对应r-1行，line_buffer[1]对应r行
        // 这时才开始计算r-1行的输出
        if (r >= KERNEL_SIZE - 2) { // KERNEL_SIZE=3时，即 r>=1
            // 对当前窗口数据进行卷积计算
            ColLoop: for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=8
                // 计算每个输出通道
                for (int oc = 0; oc < OUT_CH; oc++) {
//#pragma HLS UNROLL
                    accum_t acc = bias[oc];
                    ConvKernelLoop: for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int ch = 0; ch < IN_CH; ch++) {
#pragma HLS UNROLL
                             acc += line_buffer[kh][c][ch] * weights[oc][ch][kh];
                        }
                    }
                    // ReLU激活
                    if (acc < 0) acc = 0;
                    // 输出结果
                    out.write((data_t)acc);
                }
            }
        }
    }

    // 处理最后一行数据的输出，需要底部补零
    // 将行缓冲向上移动，并将最后一行填充为0
    for (int i = 0; i < KERNEL_SIZE - 1; i++) {
//#pragma HLS PIPELINE
        for (int c = 0; c < cols; c++) {
            for (int ch = 0; ch < IN_CH; ch++) {
                line_buffer[i][c][ch] = line_buffer[i + 1][c][ch];
            }
        }
    }
    for (int c = 0; c < cols; c++) {
        for (int ch = 0; ch < IN_CH; ch++) {
            line_buffer[KERNEL_SIZE - 1][c][ch] = 0; // 底部填充
        }
    }

    // 计算最后一行（r=rows-1）的输出
    FinalColLoop: for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=1
        for (int oc = 0; oc < OUT_CH; oc++) {
//#pragma HLS UNROLL
            accum_t acc = bias[oc];
            FinalConvKernelLoop: for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                for (int ch = 0; ch < IN_CH; ch++) {
                    acc += line_buffer[kh][c][ch] * weights[oc][ch][kh];
                }
            }
            if (acc < 0) acc = 0;
            out.write((data_t)acc);
        }
    }
}

#endif
