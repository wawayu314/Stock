import random
import numpy as np
import os
import torch as torch
import torch.nn as nn # Added for activation function classes
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockPredict
import pickle
import openpyxl # 新增导入
from openpyxl.utils import get_column_letter # 新增导入
import math # 新增导入，用于计算分割点
import pandas as pd # 新增导入，用于加载股票代码列表
import json # 新增导入，用于加载行业数据


def load_ticker_list(market_name: str):
   
    try:
        # 尝试加载ticker文件
        ticker_path = f'../dataset/{market_name}/{market_name.lower()}_ticker.csv'
        if os.path.exists(ticker_path):
            df = pd.read_csv(ticker_path)
            # 假设第一列是股票代码
            ticker_column = df.columns[0]
            ticker_list = df[ticker_column].tolist()
            print(f"✅ 成功从 {ticker_path} 加载股票列表: {len(ticker_list)} 只股票")
            return ticker_list
        
        # 如果没有ticker文件，尝试从行业数据中提取
        industry_path = f'../dataset/{market_name}/{market_name.lower()}_industry_data.json'
        if os.path.exists(industry_path):
            with open(industry_path, 'r', encoding='utf-8') as f:
                industry_data = json.load(f)
            ticker_list = list(industry_data.keys())
            print(f"✅ 从行业数据文件提取股票列表: {len(ticker_list)} 只股票")
            return ticker_list
            
    except Exception as e:
        print(f"❌ 加载股票列表失败: {e}")
    
    # 如果都失败了，返回空列表
    print(f"⚠️ 警告: 无法加载 {market_name} 的股票列表，稀疏注意力将无法使用")
    return []


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu' # 如果CUDA可用则使用GPU，否则使用CPU

data_path = '../dataset' # 数据路径
market_name = 'NYSE' # 明确设置为SP500市场
relation_name = 'wikidata' # 关系名称 (似乎未使用)
stock_num = 1026 # 股票数量 (此值将在数据加载后被动态覆盖)

# --- 模型超参数调优 (A_SHARE) - "延时慢炖"最终尝试 ---
# 目标: 在极简模型基础上，通过延长训练时间和微调正则化，做提升RIC的最后一次尝试
# 策略: 给予成功的极简模型更长的训练时间，并适度放松约束，看其是否能释放全部潜力
# 1. 延长训练: 大幅增加epochs，进行充分的"慢炖"
# 2. 微调正则化: 略微降低正则化强度，允许模型在更长的训练中学习更精细的模式
# 3. 优化学习节奏: 采用与长训练周期匹配的、带一次重启的平滑学习率曲线
# ------------------------------------

lookback_length = 48 # 增加回看长度，捕捉更稳定的长期模式
epochs = 80 # 大幅增加训练轮次，进行充分的"慢炖"
# valid_index 和 test_index 将在数据加载后动态计算
fea_num = 5 # 特征数量
steps = 1 # 预测步长
learning_rate = 8e-6 # 微调学习率以匹配新容量
alpha = 0.8 # 极大地增加排序损失的权重，主攻IC/RIC指标
scale_factor = 2 # 大幅降低多尺度分析层级，简化模型
activation_str = 'GELU' # 使用GELU激活函数，通常在Transformer架构中表现更好
attention_heads = 12 # 匹配新的embed_dim(72)，并简化注意力机制
attention_dropout = 0.2 # 略微降低正则化
weight_decay = 9.5e-4 # 略微降低正则化

# Determine activation function class based on the string
if activation_str == 'GELU':
    activation_fn_to_pass = nn.GELU
elif activation_str == 'Hardswish': 
    activation_fn_to_pass = nn.Hardswish
elif activation_str == 'ReLU':
    activation_fn_to_pass = nn.ReLU
else:
    # Default to GELU or raise an error if an unsupported activation is specified
    print(f"Warning: Unsupported activation '{activation_str}', defaulting to GELU.")
    activation_fn_to_pass = nn.GELU

dataset_path = '../dataset/' + market_name # 数据集完整路径
if market_name == "SP500":
    # SP500数据集的特殊加载和预处理逻辑
    data = np.load('../dataset/SP500/SP500.npy')
    print(f"原始SP500数据形状: {data.shape} (股票数: {data.shape[0]}, 天数: {data.shape[1]}, 特征数: {data.shape[2]})")
    
    # 使用全部5年数据，不再进行切片
    print(f"使用全部5年数据 (2020-2024)，数据形状: {data.shape}")
    
    price_data = data[:, :, -1] # 价格数据
    mask_data = np.ones((data.shape[0], data.shape[1])) # 掩码数据
    eod_data = data # EOD数据
    gt_data = np.zeros((data.shape[0], data.shape[1])) # 真实收益率数据初始化
    # 计算真实收益率
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   (data[ticket][row - steps][-1] + 1e-8) # 添加小值避免除零
else:
    # 其他市场 (如NASDAQ, NYSE) 的数据加载逻辑
    try:
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print(f"Please ensure {dataset_path}/ contains eod_data.pkl, mask_data.pkl, gt_data.pkl, price_data.pkl")
        exit()

# Verify stock_num if data is loaded (e.g., from eod_data shape)
if eod_data.shape[0] != stock_num:
    print(f"Warning: stock_num ({stock_num}) does not match eod_data.shape[0] ({eod_data.shape[0]}). Adjusting stock_num.")
    stock_num = eod_data.shape[0]

trade_dates = mask_data.shape[1]

# --- 通用设置和动态分割 ---
print(f"市场为 {market_name}，实际加载股票数量更新为: {stock_num}")
print(f"总交易天数: {trade_dates}")

# 动态计算数据集分割点
train_ratio = 0.60
valid_ratio = 0.20
# 测试集比例由 (1 - train_ratio - valid_ratio) 隐式确定

# 确保最小数据集大小
min_valid_days = max(50, lookback_length + steps + 10)  # 验证集最少50天或lookback+steps+10天
min_test_days = max(50, lookback_length + steps + 10)   # 测试集最少50天或lookback+steps+10天
min_train_days = max(100, lookback_length + steps + 20) # 训练集最少100天或lookback+steps+20天

# 计算分割点
valid_index = max(min_train_days, math.floor(trade_dates * train_ratio))
test_index = max(valid_index + min_valid_days, math.floor(trade_dates * (train_ratio + valid_ratio)))

# 确保测试集也有足够的数据
if trade_dates - test_index < min_test_days:
    # 如果测试集太小，向前调整
    test_index = trade_dates - min_test_days
    if test_index <= valid_index:
        # 如果还是不够，重新分配
        available_days = trade_dates - min_train_days
        if available_days >= min_valid_days + min_test_days:
            valid_index = min_train_days
            test_index = trade_dates - min_test_days
        else:
            print(f"错误：数据集太小({trade_dates}天)，无法进行有效的训练/验证/测试分割")
            print(f"最少需要: {min_train_days + min_valid_days + min_test_days}天")
            exit(1)

print(f"根据 {train_ratio:.0%}/{valid_ratio:.0%}/_ 分割比例 (已调整为满足最小数据集要求):")
print(f"训练集天数 (0-indexed, up to valid_index-1): {valid_index} (实际比例: {valid_index/trade_dates:.1%})")
print(f"验证集天数 (valid_index to test_index-1): {test_index - valid_index} (实际比例: {(test_index-valid_index)/trade_dates:.1%})")
print(f"测试集天数 (test_index to end): {trade_dates - test_index} (实际比例: {(trade_dates-test_index)/trade_dates:.1%})")
print(f"验证集起始索引 (valid_index): {valid_index}")
print(f"测试集起始索引 (test_index): {test_index}")

# 最终安全检查
if valid_index >= test_index or test_index >= trade_dates:
    print(f"错误：分割索引无效 - valid_index:{valid_index}, test_index:{test_index}, trade_dates:{trade_dates}")
    exit(1)
    
if test_index - valid_index <= 0:
    print(f"错误：验证集大小为0 - valid_index:{valid_index}, test_index:{test_index}")
    exit(1)
    
if trade_dates - test_index <= 0:
    print(f"错误：测试集大小为0 - test_index:{test_index}, trade_dates:{trade_dates}")
    exit(1)

print("✅ 数据集分割检查通过")

# --- 数据标准化 ---
# 在数据集分割后，使用训练集数据进行标准化
train_eod_data = eod_data[:, :valid_index, :]
data_mean = np.mean(train_eod_data, axis=(0, 1), keepdims=True)
data_std = np.std(train_eod_data, axis=(0, 1), keepdims=True)
data_std[data_std == 0] = 1.0 # 防止除以零
eod_data = (eod_data - data_mean) / data_std
print("✅ EOD数据已使用训练集的均值和标准差进行标准化")
# --- 结束数据标准化 ---

# --- 结束通用逻辑 ---


# Initialize StockMixer model
calculated_concat_time_dim = 0
original_time_steps = lookback_length # Assuming lookback_length is initial_time_steps for calc
calculated_concat_time_dim += original_time_steps
num_conv_scales_to_add = max(0, scale_factor - 1)
for i in range(num_conv_scales_to_add):
    current_stride = 2**(i + 1)
    ts_after_conv = original_time_steps // current_stride
    if ts_after_conv >= 1:
        calculated_concat_time_dim += ts_after_conv
    else:
        break

# 定义注意力模块中FFN的维度乘子
attention_ffn_dim_multiplier = 5 # 微量增加FFN维度，专注提升IC
attention_ffn_dim_to_pass = calculated_concat_time_dim * attention_ffn_dim_multiplier

model = StockPredict(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    scale=scale_factor,
    activation_fn_class=activation_fn_to_pass,
    attention_num_heads=attention_heads,
    attention_hidden_ff_dim=attention_ffn_dim_to_pass,
    attention_dropout_rate=attention_dropout
).to(device)

# --- 设置行业稀疏注意力 ---
print("\n" + "="*60)
print("🎯 设置基于行业的稀疏注意力机制")
print("="*60)

# 加载股票代码列表
ticker_list = load_ticker_list(market_name)

if ticker_list:
    try:
        # 设置模型的稀疏注意力
        model.setup_sparse_attention(market_name, ticker_list)
        print("✅ 稀疏注意力设置成功！")
        
        # 获取稀疏度统计
        sparse_mask = model.stock_attention_mixer.sparse_attention.industry_mask
        if sparse_mask is not None:
            total_connections = sparse_mask.numel()
            active_connections = sparse_mask.sum().item()
            sparsity = 1 - (active_connections / total_connections)
            theoretical_speedup = total_connections / active_connections
            
            print(f"📊 稀疏注意力统计:")
            print(f"   股票数量: {len(ticker_list)}")
            print(f"   全连接情况下的注意力计算: {total_connections:,} 次")
            print(f"   稀疏连接下的注意力计算: {active_connections:,} 次")
            print(f"   稀疏度: {sparsity:.2%}")
            print(f"   理论加速比: {theoretical_speedup:.1f}x")
        
    except Exception as e:
        print(f"❌ 稀疏注意力设置失败: {e}")
        print("⚠️ 模型将使用标准的全连接注意力机制")
else:
    print("⚠️ 未找到股票代码列表，模型将使用标准的全连接注意力机制")

print("="*60)
print("✅ 模型初始化完成")
print("="*60 + "\n")
# --- 结束稀疏注意力设置 ---

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                               betas=(0.9, 0.98), eps=1e-8) # 使用AdamW优化器，更好的正则化和收敛
# 使用Cosine Annealing学习率调度，更好的收敛性能
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=2, eta_min=1e-7, verbose=True
)
# --- 梯度累积配置 ---
gradient_accumulation_steps = 4  # 梯度累积步数，模拟更大的batch size
best_valid_ic = -np.inf # 记录最佳验证IC
best_valid_perf = None # 记录最佳验证集性能
best_test_perf = None # 记录最佳验证IC对应的测试集性能
best_epoch_num = 0 # 记录最佳验证性能对应的轮次数
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
    
    # 梯度累积计数器
    accumulation_steps = 0
    
    # 遍历训练数据
    for j in range(valid_index - lookback_length - steps + 1):
        # 获取一个批次的训练数据 (不再包含volatility_batch)
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        
        prediction = model(data_batch) # 模型前向传播
        # 计算损失 (不再传递volatility_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
        # 损失缩放以适应梯度累积
        cur_loss = cur_loss / gradient_accumulation_steps
        cur_loss.backward() # 反向传播
        
        accumulation_steps += 1
        
        # 每accumulation_steps个batch或最后一个batch时更新参数
        if accumulation_steps == gradient_accumulation_steps or j == (valid_index - lookback_length - steps):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 添加梯度裁剪防止梯度爆炸
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 清空梯度
            accumulation_steps = 0

        tra_loss += cur_loss.item() * gradient_accumulation_steps  # 恢复原始损失值用于记录
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

    # 学习率调度器步进 (CosineAnnealingWarmRestarts每个epoch自动更新)
    scheduler.step()

    # 如果当前验证IC更高，则保存模型性能
    if val_perf['IC'] > best_valid_ic:
        best_valid_ic = val_perf['IC']
        best_valid_perf = val_perf
        best_test_perf = test_perf # 保存此时对应的测试集性能
        best_epoch_num = epoch + 1 # 更新最佳轮次数

    # 打印验证集和测试集性能指标
    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')

# 训练结束后打印最佳验证性能及其对应的测试性能
print("Training finished.")
print(f"Best Validation Performance at Epoch {best_epoch_num} (based on highest validation IC):") # 基于最高验证IC的最佳验证性能
print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(best_valid_perf['mse'], best_valid_perf['IC'],
                                                                            best_valid_perf['RIC'], best_valid_perf['prec_10'], best_valid_perf['sharpe5']))
print("Corresponding Test Performance:") # 对应的测试性能
print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(best_test_perf['mse'], best_test_perf['IC'],
                                                                            best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))

# --- 开始写入Excel的逻辑 ---
if best_valid_perf and best_test_perf and best_epoch_num > 0:
    model_name_for_excel = "StockPredict_SparseAttention"  # 保持原来的模型名称
    excel_file_path = '../training_results.xlsx' # 相对于src目录
    
    # 获取稀疏注意力统计信息
    sparse_attention_info = {
        'enabled': False,
        'sparsity': 0.0,
        'theoretical_speedup': 1.0,
        'industry_groups': 0,
        'stocks_with_industry': 0
    }
    
    try:
        # 尝试获取稀疏注意力统计
        sparse_mask = model.stock_attention_mixer.sparse_attention.industry_mask
        if sparse_mask is not None:
            total_connections = sparse_mask.numel()
            active_connections = sparse_mask.sum().item()
            sparse_attention_info['enabled'] = True
            sparse_attention_info['sparsity'] = 1 - (active_connections / total_connections)
            sparse_attention_info['theoretical_speedup'] = total_connections / active_connections
            
            # 统计行业信息
            industry_mapping = model.stock_attention_mixer.sparse_attention.industry_mapping
            if industry_mapping:
                sparse_attention_info['stocks_with_industry'] = len(industry_mapping)
                # 计算行业组数
                industries = set(industry_mapping.values())
                sparse_attention_info['industry_groups'] = len(industries)
    except Exception as e:
        print(f"⚠️ 无法获取稀疏注意力统计: {e}")
    
    header = ["Model", "Dataset","Best Epoch",
              "Valid MSE", "Valid IC", "Valid RIC", "Valid Prec@10", "Valid SR",
              "Test MSE", "Test IC", "Test RIC", "Test Prec@10", "Test SR",
              "Lookback Length", "Learning Rate", "Alpha", "Scale Factor", "Activation", 
              "Attention Heads", "Attention Dropout", "Attention FFN Multiplier", "Weight Decay", "Total Epochs",
              "Sparse Attention", "Sparsity (%)", "Theoretical Speedup", "Industry Groups", "Stocks with Industry"
              ]
    
    data_row = [model_name_for_excel, market_name,                
                best_epoch_num,
                best_valid_perf.get('mse', float('nan')), 
                best_valid_perf.get('IC', float('nan')),
                best_valid_perf.get('RIC', float('nan')),
                best_valid_perf.get('prec_10', float('nan')),
                best_valid_perf.get('sharpe5', float('nan')),
                best_test_perf.get('mse', float('nan')),
                best_test_perf.get('IC', 'nan'),
                best_test_perf.get('RIC', float('nan')),
                best_test_perf.get('prec_10', float('nan')),
                best_test_perf.get('sharpe5', float('nan')),
                lookback_length, learning_rate, alpha, scale_factor, activation_str,
                attention_heads, attention_dropout, attention_ffn_dim_multiplier, weight_decay, epochs,
                "Yes" if sparse_attention_info['enabled'] else "No",
                f"{sparse_attention_info['sparsity']*100:.1f}" if sparse_attention_info['enabled'] else "N/A",
                f"{sparse_attention_info['theoretical_speedup']:.1f}" if sparse_attention_info['enabled'] else "N/A",
                sparse_attention_info['industry_groups'] if sparse_attention_info['enabled'] else "N/A",
                sparse_attention_info['stocks_with_industry'] if sparse_attention_info['enabled'] else "N/A"
               ]

    try:
        if not os.path.exists(excel_file_path):
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Training Results"
            sheet.append(header)
            # 自动调整列宽
            for col_idx, column_cells in enumerate(sheet.columns):
                length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                sheet.column_dimensions[get_column_letter(col_idx + 1)].width = length + 2
        else:
            workbook = openpyxl.load_workbook(excel_file_path)
            sheet = workbook.active
        
        sheet.append(data_row)
        workbook.save(excel_file_path)
        
        # 打印稀疏注意力信息到结果
        print(f"\n🎯 稀疏注意力训练总结:")
        print(f"   模型名称: {model_name_for_excel}")
        print(f"   稀疏注意力: {'启用' if sparse_attention_info['enabled'] else '未启用'}")
        if sparse_attention_info['enabled']:
            print(f"   稀疏度: {sparse_attention_info['sparsity']*100:.1f}%")
            print(f"   理论加速比: {sparse_attention_info['theoretical_speedup']:.1f}x")
            print(f"   行业组数: {sparse_attention_info['industry_groups']}")
            print(f"   有行业信息的股票: {sparse_attention_info['stocks_with_industry']}")
        
        print(f"Results for {model_name_for_excel} appended to {excel_file_path} (active sheet: {sheet.title})")
    except Exception as e:
        print(f"Error writing {model_name_for_excel} results to Excel: {e}")
else:
    print(f"No best performance data to write to Excel for {model_name_for_excel}.")
# --- 结束写入Excel的逻辑 ---
