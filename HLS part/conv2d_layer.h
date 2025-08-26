#ifndef CONV2D_LAYER_H
#define CONV2D_LAYER_H

#include "cnn_stream_top.h"

template<int IN_CH, int OUT_CH, int KERNEL_H, int KERNEL_W, int COLS>
void conv2d_layer(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const weight_t weights[OUT_CH][IN_CH][KERNEL_H][KERNEL_W],
    const data_t bias[OUT_CH],
    const int rows,
    const int cols)
{
#pragma HLS INLINE off

    // 填充大小
    const int PAD_TOP = KERNEL_H / 2;
    const int PAD_LEFT = KERNEL_W / 2;

    // 行缓冲，用于缓存进行卷积所需的行
    static data_t line_buffer[KERNEL_H][COLS][IN_CH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=3
#pragma HLS RESOURCE variable=line_buffer core=RAM_2P_BRAM

    // 卷积窗口
    static data_t window[KERNEL_H][KERNEL_W][IN_CH];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // 使用0填充来初始化行缓冲区的顶部
    for(int i = 0; i < PAD_TOP; i++) {
        for(int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
            for(int ch = 0; ch < IN_CH; ch++) {
                line_buffer[i][c][ch] = 0;
            }
        }
    }

    // 从输入流读取数据，填满行缓冲区的剩余部分
    for(int i = PAD_TOP; i < KERNEL_H; i++) {
        for(int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
            for(int ch = 0; ch < IN_CH; ch++) {
                line_buffer[i][c][ch] = in.read();
            }
        }
    }

    // 主处理循环
    RowLoop: for (int r = 0; r < rows; r++) {
        ColLoop: for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=8192
            // 1. 将行缓冲中的数据加载到卷积窗口的第一列，实现窗口的初始化和垂直滑动
            for (int kh = 0; kh < KERNEL_H; kh++) {
                for(int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS PIPELINE II=1
                	for (int ch = 0; ch < IN_CH; ch++) {
                    // 当c=0时，填充左侧边界
                        if (c - PAD_LEFT + kw < 0 || c - PAD_LEFT + kw >= cols) {
                            window[kh][kw][ch] = 0;
                        } else {
                            window[kh][kw][ch] = line_buffer[kh][c - PAD_LEFT + kw][ch];
                        }
                    }
                }
            }

            // 2. 计算输出
            for (int oc = 0; oc < OUT_CH; oc++) {
//#pragma HLS UNROLL// factor=4
                accum_t acc = bias[oc];
                ConvKernelLoop_H: for (int kh = 0; kh < KERNEL_H; kh++) {
#pragma HLS UNROLL
                    ConvKernelLoop_W: for (int kw = 0; kw < KERNEL_W; kw++) {
#pragma HLS UNROLL
                        for (int ch = 0; ch < IN_CH; ch++) {
//#pragma HLS UNROLL factor=8
#pragma HLS PIPELINE II=1
                            acc += window[kh][kw][ch] * weights[oc][ch][kh][kw];
                        }
                    }
                }
                if (acc < 0) acc = 0; // ReLU
                out.write((data_t)acc);
            }
        }

        // 3. 行缓冲向上移位
        for(int i = 0; i < KERNEL_H - 1; i++) {
//#pragma HLS UNROLL
            for(int c = 0; c < cols; c++) {
                for(int ch = 0; ch < IN_CH; ch++) {
                    line_buffer[i][c][ch] = line_buffer[i+1][c][ch];
                }
            }
        }

        // 4. 读取新行到行缓冲区的底部 (如果还有数据)
        // r < rows-PAD_TOP-1 是为了确保不会读取超过输入行数的数据
        if (r < rows - PAD_TOP -1) {
             for (int c = 0; c < cols; c++) {
#pragma HLS UNROLL
                for(int ch = 0; ch < IN_CH; ch++) {
#pragma HLS UNROLL
                    line_buffer[KERNEL_H-1][c][ch] = in.read();
                }
            }
        } else { // 填充底部边界
             for (int c = 0; c < cols; c++) {
#pragma HLS UNROLL
                for(int ch = 0; ch < IN_CH; ch++) {
#pragma HLS UNROLL
                    line_buffer[KERNEL_H-1][c][ch] = 0;
                }
            }
        }
    }
}
#endif
