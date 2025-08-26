#ifndef MAX_POOLING2D_H
#define MAX_POOLING2D_H

#include "cnn_stream_top.h"

template<int CH, int POOL_H, int POOL_W, int COLS>
void max_pooling2d(
    hls::stream<data_t> &in,
    hls::stream<data_t> &out,
    const int rows,
    const int cols)
{
#pragma HLS INLINE off

    // ���� POOL_H=2, POOL_W=2, Stride=2
    // ����ߴ�: rows x cols x CH
    // ����ߴ�: (rows/2) x (cols/2) x CH

    // ������������
    data_t line_buffer[2][COLS][CH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=3

    // ÿ�δ�������
    RowLoop: for (int r = 0; r < rows; r += 2) {

        // ��ȡ���е��л���
        for (int i = 0; i < 2; i++) {
            for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=1
                for (int ch = 0; ch < CH; ch++) {
                    line_buffer[i][c][ch] = in.read();
                }
            }
        }

        // ���������Ͻ��гػ���ÿ�δ�������
        ColLoop: for (int c = 0; c < cols; c += 2) {
//#pragma HLS PIPELINE II=1
            for (int ch = 0; ch < CH; ch++) {
//#pragma HLS UNROLL
                // �ҵ�2x2���ڵ����ֵ
                data_t max_val = line_buffer[0][c][ch];

                if (line_buffer[0][c+1][ch] > max_val) {
                    max_val = line_buffer[0][c+1][ch];
                }
                if (line_buffer[1][c][ch] > max_val) {
                    max_val = line_buffer[1][c][ch];
                }
                if (line_buffer[1][c+1][ch] > max_val) {
                    max_val = line_buffer[1][c+1][ch];
                }

                out.write(max_val);
            }
        }
    }
}
#endif
