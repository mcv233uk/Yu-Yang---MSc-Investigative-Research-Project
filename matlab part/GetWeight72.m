%% 参数提取脚本
clear; clc;

% 加载训练好的模型
modelFile = 'motor_cnn_model7.mat';
load(modelFile, 'net', 'modelInfo');

% 创建输出目录
outDir = 'E:\FILE2\sheffeild\毕设\fpga_weights_python';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% 提取并保存全局归一化参数
writematrix(modelInfo.global_mean', fullfile(outDir, 'global_mean.txt'));
writematrix(modelInfo.global_std', fullfile(outDir, 'global_std.txt'));

% 函数：保存数据到文件
save_data = @(data, filename) dlmwrite(fullfile(outDir, filename), data, 'precision', '%.10f');

% 遍历网络层并提取参数
for i = 1:numel(net.Layers)
    layer = net.Layers(i);
    
    switch class(layer)
        case 'nnet.cnn.layer.Convolution2DLayer'
            % 处理卷积层
            layerName = layer.Name;
            
            % 提取权重和偏置
            weights = layer.Weights;
            bias = layer.Bias;
            
            % 重塑权重维度以匹配HLS实现
            if contains(layerName, 'time')  % 垂直卷积 (3x1)
                % 原始维度: [3,1,1,4] -> HLS格式: [4,1,3]
                weights = permute(weights, [4, 3, 1, 2]);
                weights = reshape(weights, [4, 1, 3]);
                
            elseif contains(layerName, 'spatial')  % 水平卷积 (1x3)
                % 原始维度: [1,3,4,8] -> HLS格式: [8,4,3]
                weights = permute(weights, [4, 3, 2, 1]);
                weights = reshape(weights, [8, 4, 3]);
                
            elseif contains(layerName, 'mixed')  % 2D卷积 (3x3)
                % 原始维度: [3,3,8,16] -> HLS格式: [16,8,3,3]
                weights = permute(weights, [4, 3, 1, 2]);
            end
            
            % 保存参数
            save_data(weights(:), [layerName '_weights.txt']);
            save_data(bias(:), [layerName '_bias.txt']);
            
        case 'nnet.cnn.layer.FullyConnectedLayer'
            % 处理全连接层
            layerName = layer.Name;
            
            % 提取权重和偏置
            weights = layer.Weights;
            bias = layer.Bias;
            
            % 保存参数 (维度: [2,16])
            save_data(weights(:), [layerName '_weights.txt']);
            save_data(bias(:), [layerName '_bias.txt']);
    end
end

disp('所有权重和偏置已成功导出！');