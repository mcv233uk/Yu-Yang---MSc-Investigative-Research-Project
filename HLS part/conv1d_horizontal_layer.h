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

    // ���ڴ洢ÿ���������ݵĻ������ڻ���
    static data_t window_buffer[1][KERNEL_SIZE][IN_CH];
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=3
#pragma HLS RESOURCE variable=window_buffer core=RAM_2P_BRAM

    // ��Ȩ�غ�ƫ�ý��з�������ʵ����ȫ���з���
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1

    // ��������ÿһ��
    RowLoop: for (int r = 0; r < rows; r++) {
        // �����С����Ƕ�ѭ��һ�� (c <= cols) ��ˢ����ˮ�߲��������һ�еĽ��
        ColLoop: for (int c = 0; c <= cols; c++) {
//#pragma HLS PIPELINE II=16

            // 1. ���ڲ��������������ƶ�
            for (int ch = 0; ch < IN_CH; ch++) {
                for (int i = 0; i < KERNEL_SIZE - 1; i++) {
                    window_buffer[0][i][ch] = window_buffer[0][i + 1][ch];
                }
            }

            // 2. �������ݶ��봰�ڵ����Ҳ�
            // ��� c < cols������������ȡ
            // ��� c == cols������0�Խ����Ҳ����
            if (c < cols) {
                for (int ch = 0; ch < IN_CH; ch++) {
                    window_buffer[0][KERNEL_SIZE - 1][ch] = in.read();
                }
            } else { // �������һ�ε����������Ҳ����
                for (int ch = 0; ch < IN_CH; ch++) {
                    window_buffer[0][KERNEL_SIZE - 1][ch] = 0;
                }
            }

            // 3.���㹻������ʱ (c > 0)���������
            // ����� 'c_out' ���ڵ��� 'c = c_out + 1' ʱ�����
            if (c > 0) {
                data_t calc_window[KERNEL_SIZE][IN_CH];
#pragma HLS ARRAY_PARTITION variable=calc_window complete dim=0

                // 4. ��ȷ�ع������ھ���Ĵ��ڣ�������������
                for (int ch = 0; ch < IN_CH; ch++) {
                    // ���ڵ�һ���������0������������ߵ�Ԫ����0���
                    if (c == 1) {
                        calc_window[0][ch] = 0; // ������
                        calc_window[1][ch] = window_buffer[0][KERNEL_SIZE - 2][ch];
                        calc_window[2][ch] = window_buffer[0][KERNEL_SIZE - 1][ch];
                    } else { // �������к�������
                        calc_window[0][ch] = window_buffer[0][0][ch];
                        calc_window[1][ch] = window_buffer[0][1][ch];
                        calc_window[2][ch] = window_buffer[0][2][ch];
                    }
                }

                // 5. Ϊ�������ͨ��ִ�о��
                for (int oc = 0; oc < OUT_CH; oc++) {
//#pragma HLS PIPELINE II=1
                    accum_t acc = bias[oc];
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        for (int ch = 0; ch < IN_CH; ch++) {
#pragma HLS UNROLL
                            acc += calc_window[kw][ch] * weights[oc][ch][kw];
                        }
                    }

                    // ReLU �����
                    if (acc < 0) acc = 0;

                    // �����д�������
                    out.write((data_t)acc);
                }
            }
        }
    }
}

#endif
