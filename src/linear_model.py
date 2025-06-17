import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, time_steps, features_per_step):
        super(LinearModel, self).__init__()
        # 输入维度将是时间步数乘以每个时间步的特征数
        self.flatten_dim = time_steps * features_per_step
        self.linear = nn.Linear(self.flatten_dim, 1) # 定义线性层，输出维度为1

    def forward(self, x):
        # x 期望的形状: (batch_size, time_steps, features_per_step)
        # batch_size 在此上下文中通常是 stock_num (股票数量)
        
        # 展平输入以便线性层处理
        # 将形状为 [N, T, F] 的输入张量 x 转换为形状为 [N, T*F]
        # x.size(0) 是批次大小 (N)
        x = x.reshape(x.size(0), -1) 
        
        # 应用线性层
        x = self.linear(x)
        # 输出形状: (batch_size, 1)
        return x 