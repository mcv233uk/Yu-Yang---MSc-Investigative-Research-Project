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

// 定义AXI4-Stream接口结构体
struct axis_data {
    data_t data;
    ap_uint<1> last;  // 添加TLAST信号
};

// 网络参数
static const int IN_ROWS   = 500;    // 输入行数
static const int IN_COLS   = 6;      // 输入列数
static const int IN_CH     = 1;      // 输入通道数

static const int CONV1_OUT = 4;      // 第一卷积层输出通道
static const int CONV2_OUT = 8;      // 第二卷积层输出通道
static const int CONV3_OUT = 16;     // 第三卷积层输出通道
static const int OUT_CH    = 2;      // 输出分类数

//------------------------------------------------------------------
// 顶层函数声明
//------------------------------------------------------------------
void cnn_stream_top(
	hls::stream<axis_stream>  &din,
	hls::stream<axis_stream>  &dout  // 修改为完整AXI-Stream接口
);

#endif
