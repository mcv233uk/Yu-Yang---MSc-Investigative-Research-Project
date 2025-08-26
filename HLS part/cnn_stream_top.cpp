#include "cnn_stream_top.h"

void cnn_stream_top(
	hls::stream<axis_stream>  &din,
	hls::stream<axis_stream>  &dout)
{
#pragma HLS INTERFACE axis port=din
#pragma HLS INTERFACE axis port=dout
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
//#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW

    // ���������
	static hls::stream<data_t> internal_din("internal_din");
	static hls::stream<data_t> conv1_out("conv1_out");
	static hls::stream<data_t> conv2_out("conv2_out");
	static hls::stream<data_t> pool2_out("pool2_out");
	static hls::stream<data_t> conv3_out("conv3_out");
	static hls::stream<data_t> pool3_out("pool3_out");
	static hls::stream<data_t> gap_out("gap_out");
	static hls::stream<data_t> fc_out("fc_out");

	// �����
	#pragma HLS STREAM variable=internal_din depth=64
	#pragma HLS STREAM variable=conv1_out depth=64
	#pragma HLS STREAM variable=conv2_out depth=64
	#pragma HLS STREAM variable=pool2_out depth=64
	#pragma HLS STREAM variable=conv3_out depth=64
	#pragma HLS STREAM variable=pool3_out depth=32
	#pragma HLS STREAM variable=gap_out depth=16

	// ������ת�� (AXI-Stream -> ������)
	const int TOTAL_IN_SIZE = IN_ROWS * IN_COLS * IN_CH;
	for(int i = 0; i < TOTAL_IN_SIZE; i++) {
	    #pragma HLS PIPELINE II=1
	    axis_stream in_val = din.read();
	    internal_din.write(in_val.data);
	}

    //---------------------------------------------
    // ��һ��: Conv1 (3x1��ֱ���-ʱ��������ȡ)
    // ����: 500x6x1, ���: 500x6x4
    //---------------------------------------------
    conv1d_vertical_layer<IN_CH, CONV1_OUT, 3, IN_COLS>(
    	internal_din, conv1_out,
        conv1_weight, conv1_bias,
        IN_ROWS, IN_COLS
    );

    //---------------------------------------------
    // �ڶ���: Conv2 (1x3ˮƽ���-�ռ�������ȡ)
    // ����: 500x6x4, ���: 500x6x8
    //---------------------------------------------
    conv1d_horizontal_layer<CONV1_OUT, CONV2_OUT, 3, IN_ROWS>(
        conv1_out, conv2_out,
        conv2_weight, conv2_bias,
        IN_ROWS, IN_COLS
    );

    //---------------------------------------------
    // �ػ���1: 2x1���ػ� (ʱ��ά��)
    // ����: 500x6x8, ���: 250x6x8
    //---------------------------------------------
    max_pooling1d<CONV2_OUT, 2, IN_COLS>(
        conv2_out, pool2_out,
        IN_ROWS, IN_COLS
    );

    //---------------------------------------------
    // ������: Conv3 (3x3���-���������ȡ)
    // ����: 250x6x8, ���: 250x6x16
    //---------------------------------------------
    conv2d_layer<CONV2_OUT, CONV3_OUT, 3, 3, IN_COLS>(
        pool2_out, conv3_out,
        conv3_weight, conv3_bias,
        IN_ROWS/2, IN_COLS
    );

    //---------------------------------------------
    // �ػ���2: 2x2���ػ�
    // ����: 250x6x16, ���: 125x3x16
    //---------------------------------------------
    max_pooling2d<CONV3_OUT, 2, 2, IN_COLS>(
        conv3_out, pool3_out,
        IN_ROWS/2, IN_COLS
    );

    //---------------------------------------------
    // ȫ��ƽ���ػ�
    // ����: 125x3x16, ���: 1x1x16
    //---------------------------------------------
    global_avg_pool<CONV3_OUT>(
        pool3_out, gap_out,
        IN_ROWS/4, IN_COLS/2  // �ػ���ߴ�
    );

    //---------------------------------------------
    // ȫ���Ӳ� (�������)
    // ����: 16, ���: 2
    //---------------------------------------------
    fc_layer<CONV3_OUT, OUT_CH>(
            gap_out, fc_out,  // ������м���
            fc_weight, fc_bias
        );

    // ��������� (���TLAST)
        for(int i = 0; i < OUT_CH; i++) {
            #pragma HLS PIPELINE II=1
            axis_stream out_val;
            out_val.data = fc_out.read();
            out_val.last = (i == OUT_CH - 1) ? 1 : 0;  // ���һ����������TLAST=1
            //std::cout << "���last�ź�" << std::endl;
            dout.write(out_val);
        }
}
