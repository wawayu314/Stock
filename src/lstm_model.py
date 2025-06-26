import torch
import torch.nn as nn

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout_prob=0.1):
        """
        LSTM模型用于股票预测。
        参数:
            input_dim (int): 输入特征的维度 (channels)
            hidden_dim (int): LSTM隐藏层的维度
            num_layers (int): LSTM层数
            output_dim (int): 输出维度 (通常为1，预测单个值)
            dropout_prob (float): LSTM层后的dropout概率 (如果num_layers > 1)
        """
        super(LSTMStockPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层
        # batch_first=True 表示输入和输出张量的批次大小维度在前 (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # 全连接层，将LSTM的最后一个时间步的输出映射到预测值
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 的预期形状: (batch_size/stock_num, time_steps, input_dim/channels)
        
        # 初始化隐藏状态和细胞状态
        # h0 和 c0 的形状: (num_layers, batch_size/stock_num, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM前向传播
        # out 的形状: (batch_size/stock_num, time_steps, hidden_dim)
        # _ (h_n, c_n) 代表最后一个时间步的隐藏状态和细胞状态
        out, _ = self.lstm(x, (h0, c0))

        # 我们只取序列中最后一个时间步的输出进行预测
        # out[:, -1, :] 的形状: (batch_size/stock_num, hidden_dim)
        last_time_step_out = out[:, -1, :]
        
        #通过全连接层
        prediction = self.fc(last_time_step_out) # 形状: (batch_size/stock_num, output_dim)
        return prediction 