# 步骤 2: 数据集理解和预处理

本节将指导您理解项目所需的数据集格式，并实现数据加载和预处理的逻辑。

## 2.1 数据集来源和格式理解

**任务**: 理解项目使用的数据集 (NASDAQ, NYSE, S&P500) 的原始来源和结构，以及预处理后的 `.pkl` 或 `.npy` 文件格式。

**背景信息**: `README.md` 提及：
- NASDAQ/NYSE 原始数据来自: [Temporal Relational Ranking for Stock Prediction](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)
- S&P500 原始数据来自: [Efficient Integration of Multi-Order Dynamics and Internal Dynamics in Stock Movement Prediction](https://github.com/thanhtrunghuynh93/estimate)
- 预处理后的数据（如 `eod_data.pkl`, `mask_data.pkl`, `gt_data.pkl`, `price_data.pkl` for NASDAQ/NYSE, 或 `SP500.npy` for S&P500）存放在 `dataset` 文件夹中。
- `eod_data`: 包含股票的每日结束时（End-Of-Day）特征数据。
- `mask_data`: 用于标记有效数据点。
- `gt_data` (ground truth): 包含用于计算损失的真实目标值，通常是基于价格变化计算得出的收益率。
- `price_data`: 包含股票的原始价格数据，可能用于计算收益率或作为模型输入的一部分。

**提示词 (针对 LLM)**:
"我正在复现一个股票预测项目。请基于以下信息，帮我理解预处理后的数据文件 (`eod_data.pkl`, `mask_data.pkl`, `gt_data.pkl`, `price_data.pkl`, `SP500.npy`) 可能包含的数据结构和含义：
1.  `eod_data` (End-of-Day features): 可能包含哪些常见的股票技术指标作为特征 (例如，开盘价、收盘价、最高价、最低价、成交量、以及由这些基础数据派生出的其他指标如 MACD, RSI, Bollinger Bands 等)？它的数据维度通常是怎样的 (例如，[股票数量, 时间步长, 特征数量])？
2.  `mask_data`: 这个掩码数据的作用是什么？它通常是什么形状，如何表示有效或无效的数据点？
3.  `gt_data` (Ground Truth): 它是如何从原始价格数据计算得到的？它代表的是绝对价格变化、百分比变化（收益率）还是其他形式？数据维度是怎样的？
4.  `price_data`: 这是否是原始的收盘价序列？它的维度是怎样的？
5.  对于 `SP500.npy`，`README.md` 中提到 `data = data[:, 915:, :]` 和 `price_data = data[:, :, -1]`，这暗示了什么样的数据结构和切片操作？`gt_data` 是如何从这个 `SP500.npy` 计算出来的？`eod_data` 和 `SP500.npy` 中的 `data` 是什么关系？"

**测试方法**:
- **文档回顾**: 仔细阅读 LLM 生成的关于数据格式和含义的解释。
- **手动检查 (如果可以访问数据)**: 如果您有示例数据文件，可以尝试加载它们 (例如，使用 `numpy.load` 或 `pickle.load`) 并检查其形状 (`.shape`) 和一小部分内容，以验证 LLM 的解释是否与实际数据结构一致。

## 2.2 实现数据加载逻辑 (`load_data.py`)

**任务**: 实现一个 Python 脚本 (`load_data.py`)，该脚本能够根据指定的市场名称 (如 NASDAQ, NYSE, SP500) 加载相应的预处理数据文件。

**具体功能点**:
1.  能够处理 `SP500` 市场的特殊加载和预处理逻辑 (如 `train.py` 中所示的 `np.load` 和后续的切片、`gt_data` 计算)。
2.  能够处理其他市场 (如 NASDAQ, NYSE) 的 `.pkl` 文件加载逻辑。
3.  封装加载逻辑到一个或多个函数中，例如 `load_EOD_data` 函数，该函数接收 `data_path`, `market_name`, `stock_num` 等参数，并返回 `eod_data`, `mask_data`, `gt_data`, `price_data`。

**提示词 (针对 LLM)**:
"我需要创建一个 `load_data.py` 文件。请帮我设计其中的函数和逻辑，以实现以下功能：
- 根据 `market_name` (例如 'SP500', 'NASDAQ', 'NYSE') 和 `data_path`，加载相应的股票数据。
- 如果 `market_name` 是 'SP500'：
    - 加载位于 `data_path + '/SP500/SP500.npy'` 的 `.npy` 文件。
    - 对加载的数据进行切片操作 `data = data[:, 915:, :]`。
    - 提取价格数据 `price_data = data[:, :, -1]`。
    - 创建掩码数据 `mask_data`，其形状与 `data` 的前两维相同，并全部填充为 1。
    - 将原始加载并切片后的 `data` 赋值给 `eod_data`。
    - 根据 `price_data` 计算 `gt_data`。计算公式为：`gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / data[ticket][row - steps][-1]`，其中 `steps` 是一个预定义参数（例如 `steps=1`）。需要注意处理边界条件（例如 `row - steps < 0` 的情况）。
- 如果 `market_name` 不是 'SP500' (例如 'NASDAQ' 或 'NYSE'):
    - 从 `data_path + '/' + market_name + '/eod_data.pkl'` 加载 `eod_data`。
    - 从 `data_path + '/' + market_name + '/mask_data.pkl'` 加载 `mask_data`。
    - 从 `data_path + '/' + market_name + '/gt_data.pkl'` 加载 `gt_data`。
    - 从 `data_path + '/' + market_name + '/price_data.pkl'` 加载 `price_data`。
- 所有加载的数据（`eod_data`, `mask_data`, `gt_data`, `price_data`）都应该作为函数的返回值。
请描述这些函数的签名（参数和返回类型）和主要逻辑步骤，不要提供具体的 Python 代码。"

**测试方法**:
- **单元测试**:
    - 准备创建一个名为 `test_load_data.py` 的测试文件。
    - 在该文件中，针对 `load_EOD_data` 函数（或您实现的类似函数）编写单元测试。
    - **测试用例 1 (SP500)**: 
        - 准备一个符合预期结构的 mock `SP500.npy` 文件（或使用一个小型、真实的SP500示例数据）。
        - 调用数据加载函数加载 SP500 数据。
        - 验证返回的 `eod_data`, `mask_data`, `gt_data`, `price_data` 的数据类型 (应为 NumPy 数组) 和形状是否符合预期。
        - 抽样检查 `gt_data` 中的几个值，确保它们是根据描述的公式正确计算的。
        - 检查 `mask_data` 是否全部为 1。
    - **测试用例 2 (NASDAQ/NYSE - Mock PKL)**:
        - 创建一些 mock 的 `.pkl` 文件 (`eod_data.pkl`, `mask_data.pkl`, `gt_data.pkl`, `price_data.pkl`)，其中包含已知结构和内容的小型 NumPy 数组。
        - 调用数据加载函数加载这些 mock PKL 数据。
        - 验证返回的数据的类型和形状是否与 mock 文件中的数据一致。
- **集成测试 (在后续步骤中)**: 当 `train.py` 实现后，观察数据是否能被正确加载并用于模型训练，这将作为更高级别的集成测试。

## 2.3 实现数据批处理逻辑 (`get_batch`)

**任务**: 在训练脚本 (`train.py`) 中，通常会有一个 `get_batch` 函数，用于从加载的完整数据集中提取训练、验证或测试所需的批次数据。

**背景信息 (`train.py` 中的 `get_batch` 函数)**:
- 函数参数: `offset` (批次的起始索引), `lookback_length` (回溯期长度), `steps` (预测步长)。
- 功能: 根据 `offset`, `lookback_length`, 和 `steps`，从全局加载的 `eod_data`, `mask_data`, `price_data`, `gt_data` 中切片出相应的数据批次。
- `mask_batch` 的计算: `mask_batch = mask_data[:, offset: offset + seq_len + steps]`，然后 `mask_batch = np.min(mask_batch, axis=1)`，最后扩展维度。
- 返回: `eod_batch`, `mask_batch`, `price_batch`, `gt_batch`。

**提示词 (针对 LLM)**:
"我需要在我的项目中实现一个 `get_batch` 函数，该函数将用于从已加载的全量数据中提取特定批次。请基于以下描述，为我设计这个函数的逻辑：
- **输入参数**: `offset` (数据在时间序列上的起始点), `lookback_length` (每个样本包含的时间步数), `steps` (预测未来多少步，默认为1), 以及预先加载好的 `eod_data`, `mask_data`, `price_data`, `gt_data` (这些可以作为全局变量或函数参数传入)。
- **逻辑**: 
    1.  确定序列长度 `seq_len = lookback_length`。
    2.  提取 `eod_batch`: 从 `eod_data` 中切片，形状应为 `[stock_num, seq_len, fea_num]`。切片范围是 `eod_data[:, offset : offset + seq_len, :]`。
    3.  计算 `mask_batch`:
        a.  初始切片: `mask_data[:, offset : offset + seq_len + steps]`。
        b.  沿时间轴（axis=1）取最小值: `np.min(mask_batch, axis=1)`。这意味着如果在这段时间窗口内任何一天股票数据无效，则整个序列的该股票的掩码为无效。
        c.  扩展维度: `np.expand_dims(mask_batch, axis=1)`，使其形状变为 `[stock_num, 1]`。
    4.  提取 `price_batch`: 从 `price_data` 中切片，取 `offset + seq_len - 1` 时刻的价格，并扩展维度。形状应为 `[stock_num, 1]`。切片为 `price_data[:, offset + seq_len - 1]`。
    5.  提取 `gt_batch`: 从 `gt_data` 中切片，取 `offset + seq_len + steps - 1` 时刻的真实值，并扩展维度。形状应为 `[stock_num, 1]`。切片为 `gt_data[:, offset + seq_len + steps - 1]`。
- **返回值**: 返回 `eod_batch`, `mask_batch`, `price_batch`, `gt_batch`。
请描述这个函数的主要步骤和数据操作，不要提供具体的 Python 代码。"

**测试方法**:
- **单元测试**:
    - 准备创建一个名为 `test_get_batch.py` 的测试文件 (或者将测试添加到 `test_load_data.py` 或 `test_train_utils.py` 中)。
    - **测试用例**: 
        - 创建小型的、结构已知的 mock `eod_data`, `mask_data`, `price_data`, `gt_data` NumPy 数组。
        - 调用 `get_batch` 函数，使用不同的 `offset`, `lookback_length`, 和 `steps` 值。
        - 验证返回的各个批次数据的形状是否正确。
        - 验证返回的各个批次数据的内容是否是根据输入参数从 mock 数据中正确切片出来的。
        - 特别注意验证 `mask_batch` 的计算逻辑是否符合预期（例如，如果时间窗口内包含无效数据，掩码是否正确生成）。

完成这些数据准备步骤后，项目将拥有可靠的数据输入管道，为后续的模型训练奠定基础。 