import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu' # 如果CUDA可用则使用GPU，否则使用CPU

data_path = '../dataset' # 数据路径
market_name = 'NASDAQ' # 设置市场为NASDAQ
relation_name = 'wikidata' # 关系名称 (似乎未使用)
stock_num = 1026 # NASDAQ的股票数量
lookback_length = 16 # 回溯期长度
epochs = 100 # 训练轮数
valid_index = 756 # 论文中NASDAQ的训练集天数，即验证集起始索引
test_index = 1008 # 论文中NASDAQ的训练+验证集天数，即测试集起始索引
fea_num = 5 # 特征数量
market_num = 20 # 市场相关参数 (似乎用于NoGraphMixer的hidden_dim，但StockMixer初始化时未使用此变量名)
steps = 1 # 预测步长
learning_rate = 0.001 # 学习率
alpha = 0.1 # 损失函数中排序损失的权重
scale_factor = 3 # StockMixer的尺度因子
activation = 'GELU' # 激活函数名称 (当前未在StockMixer初始化中直接使用，但可以作为参考或未来扩展)

dataset_path = '../dataset/' + market_name # 数据集完整路径
if market_name == "SP500":
    # SP500数据集的特殊加载和预处理逻辑
    data = np.load('../dataset/SP500/SP500.npy')
    data = data[:, 915:, :] # 数据切片
    stock_num = data.shape[0] # 更新 stock_num 为 SP500 数据集中的实际股票数量
    print(f"市场为 SP500，实际加载股票数量更新为: {stock_num}")
    price_data = data[:, :, -1] # 价格数据
    mask_data = np.ones((data.shape[0], data.shape[1])) # 掩码数据
    eod_data = data # EOD数据
    gt_data = np.zeros((data.shape[0], data.shape[1])) # 真实收益率数据初始化
    # 计算真实收益率
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   data[ticket][row - steps][-1]
else:
    # 其他市场 (如NASDAQ) 的数据加载逻辑
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f) # 加载EOD数据
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f) # 加载掩码数据
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f) # 加载真实收益率数据
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f) # 加载价格数据

trade_dates = mask_data.shape[1] # 交易日数
# 初始化StockMixer模型
model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    no_graph_mixer_hidden_dim=market_num, # 确保使用正确的参数名和值
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 定义优化器
best_valid_loss = np.inf # 记录最佳验证损失
best_valid_perf = None # 记录最佳验证集性能
best_test_perf = None # 记录最佳验证损失对应的测试集性能
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int) # 用于训练时打乱数据批次的偏移量


def validate(start_index, end_index):
    # 在验证集或测试集上评估模型性能
    with torch.no_grad(): # 关闭梯度计算
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float) # 预测值初始化
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float) # 真实值初始化
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float) # 掩码初始化
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        # 遍历指定时间段的数据
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            # 获取一个批次的数据 (不再包含volatility_batch)
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch) # 模型前向传播
            # 计算损失 (不再传递volatility_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            # 存储预测结果、真实值和掩码
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index) # 计算平均损失
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask) # 评估性能指标
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_batch(offset=None):
    # 获取一个批次的数据用于训练或验证
    if offset is None:
        offset = random.randrange(0, valid_index) # 如果未指定偏移量，则随机选择
    seq_len = lookback_length
    # 截取EOD数据、掩码数据、价格数据和真实收益率数据
    eod_data_batch = eod_data[:, offset:offset + seq_len, :]
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1) # 确保整个序列窗口内数据有效
    
    # # 计算历史波动率 (移除相关代码)
    # if fea_num > 0: # 确保有特征数据可以用来计算波动率
    #     close_prices_seq = eod_data_batch[:, :, -1] # 取最后一个特征作为收盘价序列
    #     if seq_len > 1:
    #         # 计算日收益率 ((T_i - T_{i-1}) / T_{i-1})
    #         daily_returns = (close_prices_seq[:, 1:] - close_prices_seq[:, :-1]) / (close_prices_seq[:, :-1] + 1e-8) # 加epsilon防止除零
    #         volatility_batch_np = np.std(daily_returns, axis=1) # 计算每个股票在回溯期内的收益率标准差
    #     else:
    #         volatility_batch_np = np.zeros(stock_num) # 如果只有一个时间点，波动率为0
    # else:
    #     volatility_batch_np = np.zeros(stock_num) # 如果没有特征数据，波动率为0
        
    # volatility_batch = np.expand_dims(volatility_batch_np, axis=1) # 扩展维度以匹配损失函数期望

    return (
        eod_data_batch,
        np.expand_dims(mask_batch, axis=1), # 扩展掩码维度
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1), # 获取基准价格
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1)
        # volatility_batch # 不再返回波动率批次
    )


# 开始训练循环
for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets) # 每个epoch开始时打乱批次顺序
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    # 遍历训练数据
    for j in range(valid_index - lookback_length - steps + 1):
        # 获取一个批次的训练数据 (不再包含volatility_batch)
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        optimizer.zero_grad() # 清空梯度
        prediction = model(data_batch) # 模型前向传播
        # 计算损失 (不再传递volatility_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
        # cur_loss = cur_loss # 这行没有实际作用，可以移除
        cur_loss.backward() # 反向传播
        optimizer.step() # 更新参数

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    # 计算平均训练损失
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    # 在验证集上评估
    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    # 在测试集上评估
    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    # 如果当前验证损失更低，则保存模型性能
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf # 保存此时对应的测试集性能

    # 打印验证集和测试集性能指标
    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')

# 训练结束后打印最佳验证性能及其对应的测试性能
print("Training finished.")
print("Best Validation Performance (based on lowest validation loss):") # 基于最低验证损失的最佳验证性能
print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(best_valid_perf['mse'], best_valid_perf['IC'],
                                                                            best_valid_perf['RIC'], best_valid_perf['prec_10'], best_valid_perf['sharpe5']))
print("Corresponding Test Performance:") # 对应的测试性能
print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(best_test_perf['mse'], best_test_perf['IC'],
                                                                            best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))
