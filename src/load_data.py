import numpy as np
import os
from tqdm import tqdm


def load_EOD_data(data_path, market_name, tickers, steps=1):
    # 加载日末 (End-Of-Day, EOD) 数据、掩码、真实收益率和基准价格
    eod_data = [] # 存储EOD特征数据
    masks = [] # 存储掩码数据，标记有效/无效时间点
    ground_truth = [] # 存储真实收益率
    base_price = [] # 存储用于计算收益率的基准价格
    # 遍历每个股票代码
    for index, ticker in enumerate(tqdm(tickers)): # 使用tqdm显示进度条
        # 从CSV文件加载单个股票的EOD数据
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # 特殊处理NASDAQ数据：移除最后一天，因为可能有很多缺失数据
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            # 仅在处理第一个股票时打印数据形状，并初始化Numpy数组
            print('单个EOD数据形状:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32) # 不包含最后一列（通常是价格或日期本身）
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32) # 掩码默认为1（有效）
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32) # 真实收益率默认为0
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32) # 基准价格默认为0
        
        # 遍历单个股票的每一天数据
        for row in range(single_EOD.shape[0]):
            # 检查最后一列（通常是价格）是否为特殊值-1234，表示数据缺失
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0 # 如果数据缺失，则将掩码设为0（无效）
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8: # 确保当前和steps天前的价格都有效
                # 计算真实收益率: (P_t - P_{t-steps}) / P_{t-steps}
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            
            # 遍历每一列特征
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8: # 如果特征值为-1234 (缺失)
                    single_EOD[row][col] = 1.1 # 将缺失特征替换为1.1 (一个任意的非零值，可能需要更好的处理方式)
        
        eod_data[index, :, :] = single_EOD[:, 1:] # 存储除第一列外的所有特征数据 (假设第一列是日期或其他非特征信息)
        base_price[index, :] = single_EOD[:, -1] # 存储最后一列作为基准价格
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    # 加载并处理图关系数据，计算拉普拉斯矩阵或标准化的邻接矩阵
    relation_encoding = np.load(relation_file) # 从.npy文件加载关系编码
    print('关系编码形状:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    # 检查关系编码第三维度（通常是关系类型）的和是否为0，为0则表示节点间无连接
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    # 根据mask_flags创建邻接矩阵，有连接为1，无连接为0
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0) # 计算每个节点的度
    for i in range(len(degree)):
        if degree[i] == 0: # 防止除以零错误
            degree[i] = 1.0 # 如果度为0，则设为1，其倒数也为1
        else:
            degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree) # 计算度的-1/2次方 (D^{-0.5})
    deg_neg_half_power = np.diag(degree) # 构建对角矩阵 D^{-0.5}
    if lap:
        # 计算对称归一化拉普拉斯矩阵: I - D^{-0.5} * A * D^{-0.5}
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        # 计算标准化的邻接矩阵: D^{-0.5} * A * D^{-0.5}
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file):
    # 加载关系编码数据和相应的掩码
    relation_encoding = np.load(relation_file) # 加载关系编码
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    # 检查关系编码第三维度（关系类型）的和是否为0，为0则表示节点间无连接
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    # 创建掩码，无连接处设为-1e9 (一个很小的负数，常用于注意力机制)，有连接处为0
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask


def build_SFM_data(data_path, market_name, tickers):
    # 构建用于SFM (State Frequency Memory) 模型的数据 (此函数似乎用于提取或处理特定类型的数据)
    # 注意：此函数在当前项目中可能未被直接使用，或者用于生成预处理数据。
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('单个EOD数据形状:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32) # 初始化存储数据的数组

        # 遍历每一行数据 (每一天)
        for row in range(single_EOD.shape[0]):
            # 检查最后一列数据是否为-1234 (缺失值)
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                if row < 3: # 如果是前三天的数据缺失
                    # 向后查找第一个非缺失值进行填充
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            break
                else: # 如果不是前三天，使用前三天的均值进行填充
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
            else: # 如果数据有效
                eod_data[index][row] = single_EOD[row][-1] # 直接使用该值
    np.save(market_name + '_sfm_data', eod_data) # 将处理后的数据保存到 .npy 文件