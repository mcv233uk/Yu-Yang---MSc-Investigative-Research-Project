#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "cnn_stream_top.h" // ����ģ��ͷ�ļ�
#include "weights.h"        // ����Ȩ�غ͹�һ������

// ��������ߴ磨������ cnn_stream_top.h һ�£�
#define IN_ROWS 500
#define IN_COLS 6
#define OUT_CH  2

int main() {
    // --- 1. ׼������������ ---
	hls::stream<axis_stream> in_stream;
	hls::stream<axis_stream> out_stream;

    // --- 2. ���ļ���ȡ�������� ---
    std::cout << ">> ��ʼ���ļ����ز�������..." << std::endl;

    // �ļ�·�� (�滻Ϊ����ʵ��·��)
    const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataU1_Faulty_group109.txt";
    //const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataH1_Healthy_group023.txt";
    //const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataU3_Faulty_group043.txt";
    //const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataU2_Faulty_group043.txt";

    std::ifstream ifs(in_file_path);
    if (!ifs.is_open()) {
        std::cerr << "����: �޷������������ļ�: " << in_file_path << std::endl;
        return 1;
    }

    // �����ݶ�����ʱvector
    std::vector<float> temp_data;
    float val;
    while (ifs >> val) {
        temp_data.push_back(val);
    }
    ifs.close();

    // ����������Ƿ���ȷ
    if (temp_data.size() != IN_ROWS * IN_COLS) {
        std::cerr << "����: �������ݳߴ粻ƥ�䣡" << std::endl;
        std::cerr << "����: " << IN_ROWS * IN_COLS << "��ֵ, ʵ�ʶ�ȡ: " << temp_data.size() << "��ֵ��" << std::endl;
        return 1;
    }

    // --- 3. ʹ��ȫ�ֹ�һ��������һ������ ---
        std::cout << ">> ʹ��ȫ�ֹ�һ�����������������..." << std::endl;

        // Ӧ��ȫ�ֹ�һ��: (value - global_mean) / global_std
        for (int r = 0; r < IN_ROWS; ++r) {
            for (int c = 0; c < IN_COLS; ++c) {
                int index = r * IN_COLS + c;
                float mean_val = global_mean[c].to_float();
                float std_val = global_std[c].to_float();
                temp_data[index] = (temp_data[index] - mean_val) / std_val;
            }
        }

        // --- 4. ������д��HLS������ ---
        for (int r = 0; r < IN_ROWS; ++r) {
            for (int c = 0; c < IN_COLS; ++c) {
                axis_stream in_val;
                in_val.data = (data_t)temp_data[r * IN_COLS + c];
                in_val.last = 0;  // ����TLAST������Ϊ0
                in_stream.write(in_val);
            }
        }
    std::cout << ">> �������ݼ��ز���һ�����, �� " << temp_data.size() << " �����ݵ㡣" << std::endl;

    // --- 5. ���ö��㺯�� (Device Under Test) ---
    std::cout << ">> ����HLS����ģ�� cnn_stream_top..." << std::endl;
    cnn_stream_top(in_stream, out_stream);
    std::cout << ">> HLSģ��ִ����ϡ�" << std::endl;

    // --- 6. ��ȡ����֤������ ---
        std::cout << ">> ��ȡ������..." << std::endl;

        int output_count = 0;
        float final_scores[OUT_CH];
        int score_idx = 0;
        bool last_signal_found = false;  // ���TLAST����־

        while (!out_stream.empty() && score_idx < OUT_CH) {
            axis_stream out_val = out_stream.read();

            // ��ӡ���������Ϣ
            std::cout << "���[" << score_idx << "]: "
                      << "���� = " << out_val.data.to_float()
                      << ", TLAST = " << out_val.last
                      << std::endl;

            // �������
            final_scores[score_idx] = out_val.data.to_float();

            // ���TLAST�ź�
            if (out_val.last == 1) {
                last_signal_found = true;
                std::cout << "--> ��⵽TLAST�ź� (λ��: " << score_idx << ")" << std::endl;

                // ��֤�Ƿ������һ�����
                if (score_idx != OUT_CH - 1) {
                    std::cerr << "����: TLAST�����ڷ����һ�����λ��!" << std::endl;
                }
            }
            // ��֤�����һ�����ݲ�Ӧ��TLAST
            else if (score_idx == OUT_CH - 1) {
                std::cerr << "����: ���һ�����ȱ��TLAST�ź�!" << std::endl;
            }

            score_idx++;
            output_count++;
        }

        // ����Ƿ��⵽TLAST
        if (last_signal_found) {
            std::cout << ">> TLAST�ź���֤�ɹ�" << std::endl;
        } else if (output_count > 0) {
            std::cerr << "����: δ��⵽�κ�TLAST�ź�!" << std::endl;
        }

        // ����Ƿ��ж�������
        while (!out_stream.empty()) {
            axis_stream extra_val = out_stream.read();
            std::cerr << "����: ��⵽�������! "
                      << "���� = " << extra_val.data.to_float()
                      << ", TLAST = " << extra_val.last
                      << std::endl;
            output_count++;
        }

    // --- 7. ���ս�� ---
    std::cout << "\n-----------------------------------" << std::endl;
    std::cout << "---           �������           ---" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return 0;
}
