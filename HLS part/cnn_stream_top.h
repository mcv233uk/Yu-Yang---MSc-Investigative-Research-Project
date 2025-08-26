#ifndef CNN_STREAM_TOP_H
#define CNN_STREAM_TOP_H

#include "types.h"
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_int.h>
#include "conv1d_vertical_layer.h"
#include "conv1d_horizontal_layer.h"
#include "max_pooling1d.h"
#include "conv2d_layer.h"
#include "max_pooling2d.h"
#include "global_avg_pool.h"
#include "fc_layer.h"
#include "weights.h"
//#include "hls_video.h"

// ����AXI4-Stream�ӿڽṹ��
struct axis_data {
    data_t data;
    ap_uint<1> last;  // ���TLAST�ź�
};

// �������
static const int IN_ROWS   = 500;    // ��������
static const int IN_COLS   = 6;      // ��������
static const int IN_CH     = 1;      // ����ͨ����

static const int CONV1_OUT = 4;      // ��һ��������ͨ��
static const int CONV2_OUT = 8;      // �ڶ���������ͨ��
static const int CONV3_OUT = 16;     // ������������ͨ��
static const int OUT_CH    = 2;      // ���������

//------------------------------------------------------------------
// ���㺯������
//------------------------------------------------------------------
void cnn_stream_top(
	hls::stream<axis_stream>  &din,
	hls::stream<axis_stream>  &dout  // �޸�Ϊ����AXI-Stream�ӿ�
);

#endif
