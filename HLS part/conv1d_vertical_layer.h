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

    // �л��壬��Ҫ�洢 KERNEL_SIZE �����ݲ��ܽ������Ķ���ļ���
    static data_t line_buffer[KERNEL_SIZE][IN_COLS][IN_CH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=3
#pragma HLS RESOURCE variable=line_buffer core=RAM_2P_BRAM

    // Ȩ�غ�ƫ�÷����Ż�
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    // Ԥ������л����ǰ KERNEL_SIZE-1 �У���0����һ���Դ����ϱ߽�
    for (int i = 0; i < KERNEL_SIZE - 1; i++) {
        for (int c = 0; c < cols; c++) {
            for (int ch = 0; ch < IN_CH; ch++) {
                line_buffer[i][c][ch] = 0; // �������
            }
        }
    }

    // ������ѭ������������������
    RowLoop: for (int r = 0; r < rows; r++) {
        // ���л��������ƶ�
        for (int i = 0; i < KERNEL_SIZE - 1; i++) {
//#pragma HLS PIPELINE
            for (int c = 0; c < cols; c++) {
                for (int ch = 0; ch < IN_CH; ch++) {
                    line_buffer[i][c][ch] = line_buffer[i + 1][c][ch];
                }
            }
        }

        // ����������ȡ���е������������һ��
        for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE
            for (int ch = 0; ch < IN_CH; ch++) {
                 line_buffer[KERNEL_SIZE - 1][c][ch] = in.read();
            }
        }

        // ��r>=1ʱ��line_buffer[0]��Ӧr-1�У�line_buffer[1]��Ӧr��
        // ��ʱ�ſ�ʼ����r-1�е����
        if (r >= KERNEL_SIZE - 2) { // KERNEL_SIZE=3ʱ���� r>=1
            // �Ե�ǰ�������ݽ��о������
            ColLoop: for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=8
                // ����ÿ�����ͨ��
                for (int oc = 0; oc < OUT_CH; oc++) {
//#pragma HLS UNROLL
                    accum_t acc = bias[oc];
                    ConvKernelLoop: for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                        for (int ch = 0; ch < IN_CH; ch++) {
#pragma HLS UNROLL
                             acc += line_buffer[kh][c][ch] * weights[oc][ch][kh];
                        }
                    }
                    // ReLU����
                    if (acc < 0) acc = 0;
                    // ������
                    out.write((data_t)acc);
                }
            }
        }
    }

    // �������һ�����ݵ��������Ҫ�ײ�����
    // ���л��������ƶ����������һ�����Ϊ0
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
            line_buffer[KERNEL_SIZE - 1][c][ch] = 0; // �ײ����
        }
    }

    // �������һ�У�r=rows-1�������
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
