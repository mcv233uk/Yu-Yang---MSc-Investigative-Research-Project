import os
import functools
import operator
import numpy as np

# ==============================================================================
# 用户配置部分
# ==============================================================================

# 1. 输入目录：MATLAB 脚本导出的参数文件 (.txt) 所在位置
INPUT_DIR = 'E:\\FILE2\\sheffeild\\毕设\\fpga_weights_python'

# 2. 输出目录：生成的 weights.h 和 weights.cpp 的保存位置
OUTPUT_DIR = 'E:\\FILE2\\sheffeild\\毕设\\fpga_weights_cpp'

# ==============================================================================
# 参数映射
# ==============================================================================

PARAMS_MAP = {
    # MATLAB 文件名前缀    : C++ 变量名
    'conv1_weights'     : 'conv1_weight',
    'conv1_bias'        : 'conv1_bias',
    'conv2_weights'     : 'conv2_weight',
    'conv2_bias'        : 'conv2_bias',
    'conv3_weights'     : 'conv3_weight',
    'conv3_bias'        : 'conv3_bias',
    'fc_weights'        : 'fc_weight',
    'fc_bias'           : 'fc_bias',
    'global_mean'       : 'global_mean',
    'global_std'        : 'global_std'
}

# ==============================================================================
# 辅助函数
# ==============================================================================

def calculate_product(dims):
    """计算维度列表的总元素数量。"""
    if not dims: return 0
    return functools.reduce(operator.mul, dims, 1)

def read_data(file_path):
    """读取每行一个浮点数的数据文件。"""
    if not os.path.exists(file_path):
        print(f"警告: 数据文件未找到: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            return [float(line.strip()) for line in f.replace('\n', ',').split(',') if line.strip()]
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 失败: {e}")
        return None

def generate_cpp_initializer(dims, data):
    """递归生成C++的多维初始化列表字符串。"""
    data_iter = iter(data)
    def _recursive_formatter(sub_dims):
        if not sub_dims: return str(next(data_iter))
        parts = [_recursive_formatter(sub_dims[1:]) for _ in range(sub_dims[0])]
        return f"{{ {', '.join(parts)} }}"
    return _recursive_formatter(dims)

def format_dims_to_c_array(dims):
    """将维度列表 [4, 3] 转换为C++数组声明字符串 "[4][3]" """
    return "".join([f"[{d}]" for d in dims])

def save_dimension_info(output_dir, basename, dims):
    """保存维度信息到单独的文件"""
    dim_file = os.path.join(output_dir, f"{basename}_dims.txt")
    with open(dim_file, 'w') as f:
        f.write(",".join(map(str, dims)))
    print(f"已保存维度信息: {dim_file}")

# ==============================================================================
# 主逻辑
# ==============================================================================

def main():
    print("开始生成 weights.h 和 weights.cpp...")
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    h_file_lines, cpp_file_parts = [], []
    
    for basename, var_name in PARAMS_MAP.items():
        print(f"\n处理参数: {basename} -> {var_name}")
        
        data_file = os.path.join(INPUT_DIR, f"{basename}.txt")
        data = read_data(data_file)
        
        if data is None:
            print(f"  -> 跳过 {var_name}。")
            continue
        
        # 确定参数维度
        if var_name == 'conv1_weight':
            # conv1: [4, 1, 3]
            dims = [4, 1, 3]
        elif var_name == 'conv1_bias':
            # conv1_bias: [4]
            dims = [4]
        elif var_name == 'conv2_weight':
            # conv2: [8, 4, 3]
            dims = [8, 4, 3]
        elif var_name == 'conv2_bias':
            # conv2_bias: [8]
            dims = [8]
        elif var_name == 'conv3_weight':
            # conv3: [16, 8, 3, 3]
            dims = [16, 8, 3, 3]
        elif var_name == 'conv3_bias':
            # conv3_bias: [16]
            dims = [16]
        elif var_name == 'fc_weight':
            # fc_weight: [2, 16]
            dims = [2, 16]
        elif var_name == 'fc_bias':
            # fc_bias: [2]
            dims = [2]
        elif var_name == 'global_mean' or var_name == 'global_std':
            # 全局参数: [6]
            dims = [6]
        
        # 保存维度信息
        save_dimension_info(OUTPUT_DIR, basename, dims)
        
        # 特殊处理：全连接层权重转置
        if var_name == 'fc_weight':
            print("  -> 检测到 fc_weight，执行转置操作")
            # 将数据重塑为 [16, 2] 然后转置为 [2, 16]
            data = np.array(data).reshape(16, 2).T.flatten().tolist()
        
        # 检查数据长度
        expected_count = calculate_product(dims)
        if len(data) != expected_count:
            print(f"  -> 警告: 数据量 ({len(data)}) 与维度 {dims} 不匹配 (期望 {expected_count})")
        
        # 准备 .h 文件内容 (extern 声明)
        dims_str = format_dims_to_c_array(dims)
        h_file_lines.append(f"extern const weight_t {var_name}{dims_str};")
        
        # 准备 .cpp 文件内容 (定义和初始化)
        cpp_definition = f"const weight_t {var_name}{dims_str} = "
        initializer = generate_cpp_initializer(dims, data)
        cpp_file_parts.append(f"{cpp_definition}{initializer};\n")
        
        # 打印前几个值用于验证
        print(f"  -> 前3个值: {data[:3]}")
    
    # 组装并写入 weights.h
    h_content = ('#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n#include "types.h"\n\n' +
                 "\n".join(h_file_lines) + '\n\n#endif // WEIGHTS_H\n')
    h_file = os.path.join(OUTPUT_DIR, 'weights.h')
    with open(h_file, 'w', encoding='utf-8') as f: 
        f.write(h_content)
    print(f"\n成功写入: {h_file}")
    
    # 组装并写入 weights.cpp
    cpp_content = ('#include "weights.h"\n\n' + "\n".join(cpp_file_parts))
    cpp_file = os.path.join(OUTPUT_DIR, 'weights.cpp')
    with open(cpp_file, 'w', encoding='utf-8') as f: 
        f.write(cpp_content)
    print(f"成功写入: {cpp_file}")
    
    print("\n所有文件生成完毕！")

if __name__ == '__main__':
    main()