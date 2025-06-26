# 股票预测模型批量训练系统

这个系统可以帮您一次性运行所有模型，并自动管理数据集设置和训练流程。

## 🚀 快速开始

### 1. 基本使用
```bash
# 方法1: 从项目根目录运行
python src/run_all_models.py

# 方法2: 进入src目录后运行（推荐）
cd src
python run_all_models.py
```

### 2. 配置数据集
在 `src/config.py` 中修改：
```python
MARKET_NAME = 'NASDAQ'  # 或 'SP500'
```

### 3. 选择要运行的模型
在 `src/config.py` 中修改：
```python
MODELS_TO_RUN = {
    'StockPredict': True,   # StockMixer模型
    'GCN': True,           # 图卷积网络
    'LSTM': True,          # LSTM模型
    'LinearModel': True,   # 线性基准模型
    'STHAN-SR': False      # 跳过这个模型
}
```

## 📁 文件结构

```
项目根目录/
├── src/
│   ├── run_all_models.py     # 主控制脚本
│   ├── config.py             # 配置文件
│   ├── README_batch_training.md  # 使用说明
│   ├── train.py              # StockPredict训练脚本
│   ├── gcn_train.py          # GCN训练脚本
│   ├── lstm_train.py         # LSTM训练脚本
│   ├── linear_train.py       # LinearModel训练脚本
│   └── sthan_sr_train.py     # STHAN-SR训练脚本
├── dataset/
│   ├── NASDAQ/               # NASDAQ数据文件
│   └── SP500/                # SP500数据文件
├── logs/                     # 训练日志（自动创建）
├── backup_original_files/    # 文件备份（自动创建）
└── training_results.xlsx     # 结果汇总表（自动创建）
```

## 📋 功能特性

### ✅ 自动化功能
- **数据集切换**: 一键修改所有脚本的`market_name`设置
- **批量训练**: 顺序或并行运行多个模型
- **结果汇总**: 自动生成训练报告和Excel结果文件
- **日志管理**: 为每个模型单独保存详细日志

### ✅ 安全功能
- **文件备份**: 修改脚本前自动备份原始文件
- **环境检查**: 运行前检查Python环境和数据文件
- **错误处理**: 单个模型失败不影响其他模型训练

### ✅ 便利功能
- **预设配置**: 提供快速测试、完整训练等预设
- **进度监控**: 实时显示训练进度和耗时
- **结果预览**: 训练完成后显示关键性能指标

## 🔧 配置选项

### 运行模式
```python
RUN_MODE = 'sequential'  # 顺序运行（推荐）
RUN_MODE = 'parallel'    # 并行运行（需要足够GPU内存）
```

### 预设配置
```python
# 在src/config.py中取消注释来使用预设
load_preset("quick_test")      # 快速测试（只运行LinearModel）
load_preset("full_nasdaq")     # 完整NASDAQ测试
load_preset("full_sp500")      # 完整SP500测试
load_preset("compare_datasets") # 数据集对比
```

## 📊 输出文件

### 训练结果
- `../training_results.xlsx`: 所有模型的性能汇总表
- `../logs/batch_training_*.log`: 批量训练的主日志
- `../logs/*_training.log`: 每个模型的详细训练日志

### 备份文件
- `../backup_original_files/`: 修改前的原始脚本备份

## 🎯 使用场景

### 场景1: 对比不同数据集
```bash
# 先运行NASDAQ
cd src
# 编辑config.py: MARKET_NAME = 'NASDAQ'
python run_all_models.py

# 再运行SP500
# 编辑config.py: MARKET_NAME = 'SP500'
python run_all_models.py
```

### 场景2: 快速测试新设置
```python
# 在src/config.py中使用快速测试预设
load_preset("quick_test")
# 只运行LinearModel，快速验证环境和数据
```

### 场景3: 完整基准测试
```python
# 运行所有模型进行完整对比
MODELS_TO_RUN = {
    'StockPredict': True,
    'GCN': True, 
    'LSTM': True,
    'LinearModel': True,
    'STHAN-SR': True
}
```

## ⚠️ 注意事项

### 工作目录
- **推荐**: 在`src`目录下运行脚本
- **路径**: 所有相对路径都基于`src`目录计算

### GPU内存
- **顺序运行**: 每次只训练一个模型，GPU内存需求较低
- **并行运行**: 同时训练多个模型，需要大量GPU内存

### 训练时间
- 所有模型训练可能需要几小时到十几小时
- 建议先用`quick_test`预设验证环境

### 数据要求
- **NASDAQ**: 需要`../dataset/NASDAQ/`下的pkl文件
- **SP500**: 需要`../dataset/SP500/SP500.npy`文件

## 🔍 故障排除

### 常见问题
1. **"未找到config.py"**: 确保在src目录下运行，或使用完整路径
2. **"数据文件不存在"**: 检查项目根目录的dataset目录结构
3. **"GPU内存不足"**: 改用`RUN_MODE = 'sequential'`
4. **"某个模型训练失败"**: 查看对应的详细日志文件

### 检查命令
```bash
# 检查环境
cd src
python -c "import torch, numpy, pandas, openpyxl; print('环境OK')"

# 检查数据文件
ls ../dataset/NASDAQ/  # 应该有pkl文件
ls ../dataset/SP500/   # 应该有npy文件

# 检查工作目录
pwd  # 应该显示.../src
```

## 📈 结果分析

训练完成后，查看项目根目录的 `training_results.xlsx` 文件：
- **MSE**: 均方误差（越小越好）
- **IC**: 信息系数（越大越好）
- **RIC**: 排序信息系数（越大越好）
- **Prec@10**: Top10精确度（越大越好）
- **SR**: 夏普比率（越大越好）

## 🛠️ 自定义扩展

### 添加新模型
1. 在`run_all_models.py`的`TRAINING_SCRIPTS`中添加脚本路径
2. 在`config.py`的`MODELS_TO_RUN`中添加开关
3. 确保新脚本有`market_name`变量并且支持路径引用

### 修改超参数
- 直接在各个训练脚本中修改
- 或者扩展`config.py`添加超参数配置

## 💡 使用技巧

### 快速切换配置
```bash
# 在src目录下创建多个配置文件
cp config.py config_nasdaq.py
cp config.py config_sp500.py

# 使用时复制对应配置
cp config_nasdaq.py config.py
python run_all_models.py
```

### 查看实时日志
```bash
# 在另一个终端窗口查看日志
tail -f ../logs/batch_training_*.log
```

### 并行模式监控
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi
```

---

**🎉 祝您训练顺利！如有问题，请查看日志文件进行调试。** 