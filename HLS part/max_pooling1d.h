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

    // �л���洢ǰһ������
	data_t line_buffer[COLS][CH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2

    // ����ÿ���У�һ���ػ����ڣ�
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
//#pragma HLS PIPELINE II=1

            // ��ȡ��ǰ�е�����ͨ��
            data_t current_col[CH];
            for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
                current_col[ch] = in.read();
            }

            // ż���У��洢���ݵ��л���
            if (r % POOL_SIZE == 0) {
                for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
                    line_buffer[c][ch] = current_col[ch];
                }
            }
            // �����У��뻺�����ݱȽϲ�������ֵ
            else {
                for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
                    data_t max_val = line_buffer[c][ch];

                    // �Ƚϲ���ȡ���ֵ
                    if (current_col[ch] > max_val) {
                        max_val = current_col[ch];
                    }

                    // ����ػ����
                    out.write(max_val);
                }
            }
        }
    }
}

#endif
