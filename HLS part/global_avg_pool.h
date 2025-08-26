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

    // �����ۼ�ÿ��ͨ���ܺ͵ļĴ���
    accum_t channel_sums[CH];
#pragma HLS ARRAY_PARTITION variable=channel_sums complete dim=1

    // ��ʼ���ۼ���Ϊ0
    InitLoop: for (int i = 0; i < CH; i++) {
#pragma HLS UNROLL
        channel_sums[i] = 0;
    }

    const int total_pixels = rows * cols;

    // ���������������ز������ۼ�
    // ѭ������Ϊ rows * cols��ÿ�ζ�ȡһ�����أ���������ͨ����
    AccumLoop: for (int i = 0; i < total_pixels; i++) {
//#pragma HLS PIPELINE II=1
        // һ���Զ�ȡһ�����ص�����ͨ������
        for (int ch = 0; ch < CH; ch++) {
#pragma HLS UNROLL
            data_t pixel_val = in.read();
            channel_sums[ch] += pixel_val;
        }
    }

    // �������ش�����Ϻ󣬼���ƽ��ֵ�����
    AvgLoop: for (int ch = 0; ch < CH; ch++) {
//#pragma HLS PIPELINE II=1
        // ����ƽ��ֵ
        accum_t avg_val = channel_sums[ch] / total_pixels;
        // д�������
        out.write((data_t)avg_val);
    }
}

#endif
