# Stock Prediction with Multi-Modal Fusion

基于多尺度时间特征与情绪分析的股价预测系统，融合价格特征、技术指标与新闻情绪分析，支持多市场股票预测。

## 项目概述

本项目实现了一个高级股价预测模型，具有以下特性：

- **多市场支持**: 针对 NYSE、NASDAQ、SP500、A-Share 市场的专门优化
- **多尺度时间特征**: 多尺度卷积捕获不同时间尺度的模式
- **稀疏行业注意力**: 基于行业的稀疏注意力机制，提升计算效率
- **新闻情绪融合**: 基于 FinBERT 的情绪分析，通过门控多模态架构与价格特征融合
- **智能参数配置**: 基于市场特征的自动超参数调优
- **多种融合策略**: 支持拼接融合、门控融合等多种多模态融合方法

## 核心创新

1. **多尺度时间混合器 (Multi-Scale Time Mixer)**: 通过三角上 (TriU) 网络融合原始尺度与多卷积尺度 (stride 2, 4, 8...) 的时间特征

2. **稀疏行业注意力 (Sparse Industry Attention)**: 通过限制在同行业组内的注意力，将 O(n²) 复杂度降至 O(k²×m)

3. **门控多模态融合 (Gated Multimodal Fusion)**: 使用三层 MLP 门网络动态学习价格特征与情绪特征的权重

4. **宏观自适应门控 (Macro-Adaptive Gating)**: 结合 VIX 和国债收益率信号，在市场压力期间自适应控制模态融合

5. **情绪分析辅助 (Sentiment-Augmented Prediction)**: 利用 FinBERT 对财经新闻进行情绪分析，作为预测的辅助特征

## 项目结构

```
Stock/
├── src/
│   ├── model.py                       # 核心模型: MultiScaleTimeMixer, StockPredict
│   ├── train.py                       # 基础训练脚本 (仅价格特征)
│   ├── evaluator.py                   # 性能评估指标 (IC, RIC, Sharpe等)
│   ├── load_data.py                   # 数据加载工具
│   ├── config.py                      # 配置文件
│   │
│   ├── 训练脚本 (Training Scripts)
│   │   ├── gated_multimodal_train.py  # 门控多模态融合训练
│   │   ├── concat_fusion_train.py     # 拼接融合训练
│   │   ├── unimodal_train.py          # 单模态基线训练
│   │   ├── sentiment_lstm_train.py    # 情绪增强LSTM训练
│   │   ├── lstm_train.py              # LSTM基线训练
│   │   ├── linear_train.py            # 线性模型训练
│   │   ├── gcn_train.py               # 图卷积网络训练
│   │   ├── sthan_sr_train.py          # STHAN-SR模型训练
│   │   └── train_ablation.py          # 消融实验训练
│   │
│   ├── 情绪分析 (Sentiment Analysis)
│   │   ├── news_sentiment_preprocessor.py  # 新闻情绪预处理
│   │   └── quick_start_sentiment.py        # 快速开始情绪分析
│   │
│   ├── 数据检查 (Data Checking)
│   │   ├── check_nyse_data.py         # 检查NYSE数据
│   │   ├── check_sentiment_data.py    # 检查情绪数据
│   │   └── quick_check.py             # 快速数据检查
│   │
│   └── 辅助工具 (Utilities)
│       ├── run_all_models.py           # 运行所有模型
│       ├── run_all_news_collection.py # 运行所有新闻收集
│       └── benchmark_flops.py          # 基准测试
│
├── dataset/                            # 数据集目录 (不纳入git)
│   ├── SP500/
│   ├── NASDAQ/
│   ├── NYSE/
│   └── A_SHARE/
│
├── requirements.txt                    # 核心依赖
├── requirements_sentiment.txt          # 情绪分析依赖
└── README.md
```

## 核心模块说明

### 多模态融合训练

| 脚本 | 说明 |
|------|------|
| `gated_multimodal_train.py` | 门控多模态融合，使用 MLP 门网络动态调节价格与情绪特征的融合权重 |
| `concat_fusion_train.py` | 拼接融合，将价格特征与情绪特征直接拼接后输入预测器 |
| `unimodal_train.py` | 单模态训练，仅使用价格特征的基线模型 |
| `sentiment_lstm_train.py` | 情绪增强 LSTM，将情绪特征与价格特征一同输入 LSTM |

### 情绪分析模块

| 脚本 | 说明 |
|------|------|
| `news_sentiment_preprocessor.py` | 使用 FinBERT 对财经新闻进行情感分析，生成每日情感得分 |
| `quick_start_sentiment.py` | 快速开始脚本，用于快速验证情绪分析流程 |

### 基线对比模型

| 脚本 | 模型类型 |
|------|----------|
| `train.py` | 多尺度时间混合器 (本项目核心模型) |
| `lstm_train.py` | 长短期记忆网络 |
| `linear_train.py` | 线性回归模型 |
| `gcn_train.py` | 图卷积网络 |
| `sthan_sr_train.py` | STHAN-SR 模型 |

## 安装

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt

# 情绪分析依赖 (包含 FinBERT)
pip install -r requirements_sentiment.txt
```

### 3. 数据准备

数据集目录结构如下：

```
dataset/
├── NYSE/
│   ├── eod_data.pkl
│   ├── mask_data.pkl
│   ├── gt_data.pkl
│   ├── price_data.pkl
│   ├── nyse_ticker.csv
│   └── nyse_industry_data.json
├── NASDAQ/
│   └── ...
├── SP500/
│   ├── SP500.npy
│   ├── sp500_ticker.csv
│   ├── sp500_industry_data.json
│   └── news_sentiment/          # 新闻情绪数据
│       ├── AAPL_daily_sentiment.csv
│       ├── AAPL_news_detailed.csv
│       └── ...
└── A_SHARE/
    └── ...
```

## 使用方法

### 1. 基础训练 (仅价格特征)

```bash
cd src
python train.py
```

### 2. 门控多模态融合训练 (价格 + 情绪)

```bash
cd src
python gated_multimodal_train.py
```

### 3. 拼接融合训练

```bash
cd src
python concat_fusion_train.py
```

### 4. 单模态基线对比

```bash
cd src
python unimodal_train.py
```

### 5. 情绪分析数据生成

```bash
cd src
python news_sentiment_preprocessor.py
```

### 6. 切换市场

修改训练脚本中的 `MARKET_NAME` 变量：

```python
MARKET_NAME = 'SP500'  # 选项: 'NYSE', 'NASDAQ', 'SP500', 'A_SHARE'
```

### 7. 自定义参数

系统会自动根据市场选择最优参数：

| 参数 | NYSE | NASDAQ | SP500 | A_SHARE |
|------|------|--------|-------|---------|
| lookback_length | 32 | 64 | 32 | 36 |
| epochs | 60 | 60 | 60 | 150 |
| learning_rate | 5e-5 | 1e-4 | 2e-5 | 8e-5 |
| alpha (排名损失) | 0.75 | 0.8 | 0.5 | 0.2 |
| scale_factor | 3 | 3 | 4 | 1 |

### 8. 运行所有模型对比

```bash
cd src
python run_all_models.py
```

## 性能指标

- **IC (信息系数)**: 预测与实际收益的相关性
- **RIC (排名信息系数)**: 排名预测质量
- **Precision@10**: 前10%股票的预测准确率
- **夏普比率**: 风险调整收益指标

## 模型架构

### 多尺度时间混合器

- 使用1D卷积在多个时间尺度上提取特征
- 通过三角上 (TriU) 网络融合不同尺度
- 使用因果注意力保持时间顺序

### 行业稀疏注意力

- 加载基于SIC的行业分类
- 限制注意力在同行业组内
- 根据行业分布实现2-5倍计算加速

### 门控多模态融合

- **价格编码器**: 对价格/技术特征使用LSTM
- **情绪编码器**: 对新闻情绪特征使用LSTM
- **门网络**: 3层MLP预测模态权重
- **宏观控制器**: VIX + 国债收益率信号调制门控

## 引用

如果本项目对你的研究有帮助，请引用：

```
@misc{stock_prediction_2025,
  title = {Stock Prediction with Multi-Modal Fusion},
  author = {Yu Huang},
  year = {2025}
}
```

## 许可证

MIT License

---

**风险提示**: 本项目仅用于教育和研究目的，不构成投资建议。股票投资有风险，请谨慎操作。
