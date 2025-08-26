#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include "cnn_stream_top.h" // 顶层模块头文件
#include "weights.h"        // 包含权重和归一化参数

// 定义输入尺寸（必须与 cnn_stream_top.h 一致）
#define IN_ROWS 500
#define IN_COLS 6
#define OUT_CH  2

int main() {
    // --- 1. 准备输入和输出流 ---
	hls::stream<axis_stream> in_stream;
	hls::stream<axis_stream> out_stream;

    // --- 2. 从文件读取测试数据 ---
    std::cout << ">> 开始从文件加载测试数据..." << std::endl;

    // 文件路径 (替换为您的实际路径)
    const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataU1_Faulty_group109.txt";
    //const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataH1_Healthy_group023.txt";
    //const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataU3_Faulty_group043.txt";
    //const char* in_file_path = "E:/SOFTWARE2/Xilinx/Vivado/project/CNN_HLS2/CNN_HLS2/data/hls_data2/TraindataU2_Faulty_group043.txt";

    std::ifstream ifs(in_file_path);
    if (!ifs.is_open()) {
        std::cerr << "错误: 无法打开输入数据文件: " << in_file_path << std::endl;
        return 1;
    }

    // 将数据读入临时vector
    std::vector<float> temp_data;
    float val;
    while (ifs >> val) {
        temp_data.push_back(val);
    }
    ifs.close();

    // 检查数据量是否正确
    if (temp_data.size() != IN_ROWS * IN_COLS) {
        std::cerr << "错误: 输入数据尺寸不匹配！" << std::endl;
        std::cerr << "期望: " << IN_ROWS * IN_COLS << "个值, 实际读取: " << temp_data.size() << "个值。" << std::endl;
        return 1;
    }

    // --- 3. 使用全局归一化参数归一化数据 ---
        std::cout << ">> 使用全局归一化参数处理测试数据..." << std::endl;

        // 应用全局归一化: (value - global_mean) / global_std
        for (int r = 0; r < IN_ROWS; ++r) {
            for (int c = 0; c < IN_COLS; ++c) {
                int index = r * IN_COLS + c;
                float mean_val = global_mean[c].to_float();
                float std_val = global_std[c].to_float();
                temp_data[index] = (temp_data[index] - mean_val) / std_val;
            }
        }

        // --- 4. 将数据写入HLS输入流 ---
        for (int r = 0; r < IN_ROWS; ++r) {
            for (int c = 0; c < IN_COLS; ++c) {
                axis_stream in_val;
                in_val.data = (data_t)temp_data[r * IN_COLS + c];
                in_val.last = 0;  // 输入TLAST可以设为0
                in_stream.write(in_val);
            }
        }
    std::cout << ">> 测试数据加载并归一化完成, 共 " << temp_data.size() << " 个数据点。" << std::endl;

    // --- 5. 调用顶层函数 (Device Under Test) ---
    std::cout << ">> 调用HLS顶层模块 cnn_stream_top..." << std::endl;
    cnn_stream_top(in_stream, out_stream);
    std::cout << ">> HLS模块执行完毕。" << std::endl;

    // --- 6. 读取并验证输出结果 ---
        std::cout << ">> 读取输出结果..." << std::endl;

        int output_count = 0;
        float final_scores[OUT_CH];
        int score_idx = 0;
        bool last_signal_found = false;  // 添加TLAST检测标志

        while (!out_stream.empty() && score_idx < OUT_CH) {
            axis_stream out_val = out_stream.read();

            // 打印完整输出信息
            std::cout << "输出[" << score_idx << "]: "
                      << "数据 = " << out_val.data.to_float()
                      << ", TLAST = " << out_val.last
                      << std::endl;

            // 保存分数
            final_scores[score_idx] = out_val.data.to_float();

            // 检查TLAST信号
            if (out_val.last == 1) {
                last_signal_found = true;
                std::cout << "--> 检测到TLAST信号 (位置: " << score_idx << ")" << std::endl;

                // 验证是否是最后一个输出
                if (score_idx != OUT_CH - 1) {
                    std::cerr << "错误: TLAST出现在非最后一个输出位置!" << std::endl;
                }
            }
            // 验证非最后一个数据不应有TLAST
            else if (score_idx == OUT_CH - 1) {
                std::cerr << "错误: 最后一个输出缺少TLAST信号!" << std::endl;
            }

            score_idx++;
            output_count++;
        }

        // 检查是否检测到TLAST
        if (last_signal_found) {
            std::cout << ">> TLAST信号验证成功" << std::endl;
        } else if (output_count > 0) {
            std::cerr << "警告: 未检测到任何TLAST信号!" << std::endl;
        }

        // 检查是否有额外数据
        while (!out_stream.empty()) {
            axis_stream extra_val = out_stream.read();
            std::cerr << "错误: 检测到额外输出! "
                      << "数据 = " << extra_val.data.to_float()
                      << ", TLAST = " << extra_val.last
                      << std::endl;
            output_count++;
        }

    // --- 7. 最终结果 ---
    std::cout << "\n-----------------------------------" << std::endl;
    std::cout << "---           仿真完成           ---" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return 0;
}
