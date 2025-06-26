# -*- coding: utf-8 -*-
"""
批量训练配置文件
在这里修改所有设置，然后运行 run_all_models.py
"""

# =============== 数据集配置 ===============
MARKET_NAME = 'A_SHARE'  # 可选: 'NASDAQ', 'SP500', 'NYSE', 'A_SHARE'

# =============== 训练模型选择 ===============
# 设置为 True 表示运行该模型，False 表示跳过
# 为了快速验证，默认只运行主模型 StockPredict
MODELS_TO_RUN = {
    'StockPredict': False,   # 主要的StockMixer模型
    'GCN': True,           # 图卷积网络
    'LSTM': True,          # 长短期记忆网络  
    'LinearModel': True,   # 线性基准模型
    'STHAN-SR': True       # 时空混合注意力网络
}

# =============== 运行方式配置 ===============
RUN_MODE = 'parallel'    # 'sequential'(顺序运行) 或 'parallel'(并行运行)
                          # 推荐: 如果GPU内存够大用parallel，否则用sequential

# =============== 日志配置 ===============
ENABLE_LOGGING = True     # 是否启用详细日志
LOG_DIR = '../logs'      # 日志目录（相对于src目录）
SAVE_MODEL_LOGS = True   # 是否为每个模型单独保存日志

# =============== 高级配置 ===============
# 超时设置（秒）
MAX_TRAINING_TIME_PER_MODEL = 3600  # 每个模型最大训练时间1小时
ENABLE_TIMEOUT = False               # 是否启用超时控制

# GPU设置
CUDA_VISIBLE_DEVICES = None  # 例如: "0,1" 或 None (使用所有GPU)

# 验证设置  
CHECK_REQUIREMENTS = True    # 运行前检查依赖包
BACKUP_ORIGINAL_FILES = True # 修改文件前是否备份

# =============== 快速预设配置 ===============
def load_preset(preset_name):
    """加载预设配置"""
    global MARKET_NAME, MODELS_TO_RUN, RUN_MODE
    
    if preset_name == "quick_test":
        # 快速测试：只运行线性模型
        MARKET_NAME = 'NASDAQ'
        MODELS_TO_RUN = {
            'StockPredict': False,
            'GCN': False,
            'LSTM': False,
            'LinearModel': True,
            'STHAN-SR': False
        }
        RUN_MODE = 'sequential'
        
    elif preset_name == "full_nasdaq":
        # 完整NASDAQ测试
        MARKET_NAME = 'NASDAQ'
        MODELS_TO_RUN = {
            'StockPredict': True,
            'GCN': True,
            'LSTM': True,
            'LinearModel': True,
            'STHAN-SR': True
        }
        RUN_MODE = 'sequential'
        
    elif preset_name == "full_sp500":
        # 完整SP500测试
        MARKET_NAME = 'SP500'
        MODELS_TO_RUN = {
            'StockPredict': True,
            'GCN': True,
            'LSTM': True,
            'LinearModel': True,
            'STHAN-SR': True
        }
        RUN_MODE = 'sequential'
        
    elif preset_name == "full_ashare":
        # 完整A_SHARE测试
        MARKET_NAME = 'A_SHARE'
        MODELS_TO_RUN = {
            'StockPredict': True,
            'GCN': True,
            'LSTM': True,
            'LinearModel': True,
            'STHAN-SR': True
        }
        RUN_MODE = 'sequential'

    elif preset_name == "compare_datasets":
        # 数据集对比：主要模型
        MODELS_TO_RUN = {
            'StockPredict': True,
            'GCN': True,
            'LSTM': False,
            'LinearModel': True,
            'STHAN-SR': False
        }
        RUN_MODE = 'sequential'

# =============== 使用预设配置 ===============
# 取消注释下面的行来使用预设配置
# load_preset("quick_test")      # 快速测试
# load_preset("full_nasdaq")     # 完整NASDAQ测试
# load_preset("full_sp500")      # 完整SP500测试
# load_preset("full_ashare")     # 完整A_SHARE测试
# load_preset("compare_datasets") # 数据集对比

print(f"📋 当前配置: 数据集={MARKET_NAME}, 运行模式={RUN_MODE}")
print(f"🎯 要运行的模型: {[k for k, v in MODELS_TO_RUN.items() if v]}") 