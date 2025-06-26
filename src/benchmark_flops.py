# ----------------------------------------------------------------------------------
# 脚本说明:
# 该脚本用于精确计算和比较全连接注意力与稀疏注意力模型的理论计算量 (FLOPs)。
# 它不测量实际运行时间，而是通过静态分析模型结构来提供一个公平、可复现的效率指标。
# 这对于在论文中展示您提出的稀疏注意力机制的内在优势至关重要。
#
# 前置要求:
# 请确保已安装 'fvcore' 库:
# pip install fvcore
# ----------------------------------------------------------------------------------

import numpy as np
import os
import torch
import torch.nn as nn
from model import StockPredict
import pandas as pd
import json
from fvcore.nn import FlopCountAnalysis, flop_count_table

# ====================================================================================
# --- 计算量基准测试配置 ---
# ====================================================================================
# 核心参数: 数据放大因子。更大的值能更好地模拟大规模场景，并凸显复杂度差异。
AMPLIFICATION_FACTOR = 1 

# 固定使用一个市场的数据作为基准
MARKET_NAME = 'NYSE'

# 模型超参数应与主要训练脚本保持一致，以确保可比性
LOOKBACK_LENGTH = 32
FEA_NUM = 5
SCALE_FACTOR = 3
ACTIVATION_STR = 'GELU'
ATTENTION_HEADS = 8
ATTENTION_DROPOUT = 0.1
# ====================================================================================


def load_ticker_list(market_name: str):
    """加载股票代码列表，用于设置稀疏注意力。"""
    try:
        ticker_path = f'../dataset/{market_name}/{market_name.lower()}_ticker.csv'
        if os.path.exists(ticker_path):
            df = pd.read_csv(ticker_path)
            ticker_column = df.columns[0]
            return df[ticker_column].tolist()
        
        industry_path = f'../dataset/{market_name}/{market_name.lower()}_industry_data.json'
        if os.path.exists(industry_path):
            with open(industry_path, 'r', encoding='utf-8') as f:
                industry_data = json.load(f)
            return list(industry_data.keys())
            
    except Exception as e:
        print(f"❌ 加载股票列表失败: {e}")
    return []

def build_model(use_sparse, stock_num, device, ticker_list):
    """构建并返回一个配置好的StockPredict模型。"""
    if ACTIVATION_STR == 'GELU': activation_fn_to_pass = nn.GELU
    elif ACTIVATION_STR == 'Hardswish': activation_fn_to_pass = nn.Hardswish
    else: activation_fn_to_pass = nn.ReLU

    calculated_concat_time_dim = LOOKBACK_LENGTH
    num_conv_scales_to_add = max(0, SCALE_FACTOR - 1)
    for i in range(num_conv_scales_to_add):
        ts_after_conv = LOOKBACK_LENGTH // (2**(i + 1))
        if ts_after_conv >= 1: calculated_concat_time_dim += ts_after_conv
        else: break
    attention_ffn_dim_to_pass = calculated_concat_time_dim * 6

    model = StockPredict(
        stocks=stock_num,
        time_steps=LOOKBACK_LENGTH,
        channels=FEA_NUM,
        scale=SCALE_FACTOR,
        activation_fn_class=activation_fn_to_pass,
        attention_num_heads=ATTENTION_HEADS,
        attention_hidden_ff_dim=attention_ffn_dim_to_pass,
        attention_dropout_rate=ATTENTION_DROPOUT,
        use_sparse_attention=use_sparse,
        use_multiscale_fusion=True,
        use_triu_network=True
    ).to(device)

    if use_sparse:
        if not ticker_list:
            raise ValueError("需要提供股票代码列表 (ticker_list) 来构建稀疏注意力模型。")
        model.setup_sparse_attention(MARKET_NAME, ticker_list)
    
    model.eval() # 设置为评估模式
    return model

def main():
    """主执行函数"""
    torch.random.manual_seed(12345678)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 准备数据维度和股票列表 ---
    print(f"加载市场 '{MARKET_NAME}' 的元数据...")
    ticker_list = load_ticker_list(MARKET_NAME)
    if not ticker_list:
        print("无法加载股票列表，基准测试中止。")
        return
        
    # 我们只需要原始股票数来计算放大后的数量
    # 使用pandas读取csv获取股票数比加载整个pkl更高效
    ticker_path = f'../dataset/{MARKET_NAME}/{MARKET_NAME.lower()}_ticker.csv'
    original_stock_num = len(pd.read_csv(ticker_path))

    print(f"原始股票数量: {original_stock_num}")

    # --- 人工放大问题规模 ---
    if AMPLIFICATION_FACTOR > 1:
        print(f"🔥 人工放大因子为 {AMPLIFICATION_FACTOR}x...")
        stock_num = original_stock_num * AMPLIFICATION_FACTOR
        ticker_list = ticker_list * AMPLIFICATION_FACTOR
    else:
        stock_num = original_stock_num
    
    print(f"分析场景: {stock_num} 只股票")

    # --- 创建一个用于分析的虚拟输入张量 ---
    # 注意: 根据当前 model.py 的设计，输入应为 3D 张量: (stock_num, time_steps, channels)
    dummy_input = torch.randn(stock_num, LOOKBACK_LENGTH, FEA_NUM).to(device)

    # --- 分析全连接模型 ---
    print("\n" + "="*70)
    print("📊 正在分析: 全连接注意力模型 (Full Attention)")
    print("="*70)
    full_model = build_model(False, stock_num, device, ticker_list)
    flops_full = FlopCountAnalysis(full_model, dummy_input)
    total_flops_full = flops_full.total()
    total_params_full = sum(p.numel() for p in full_model.parameters())
    print(flop_count_table(flops_full, max_depth=4))

    # --- 分析稀疏注意力模型 (理论计算) ---
    print("\n" + "="*70)
    print("📊 正在分析: 稀疏注意力模型 (理论最优实现)")
    print("="*70)
    
    # 为了获取稀疏掩码信息，我们需要构建一次稀疏模型
    sparse_model_for_mask = build_model(True, stock_num, device, ticker_list)
    active_connections = sparse_model_for_mask.stock_attention_mixer.sparse_attention.industry_mask.sum().item()
    
    # --- 手动计算 Attention Matmul 的FLOPs ---
    # 详尽解释:
    # 1. 注意力机制的核心计算在于 Q@K.T 和 Scores@V 这两个矩阵乘法。
    # 2. 由于 fvcore 在此模型上严重低估了 matmul 的计算量，我们将手动计算这部分，
    #    并以此作为最核心、最公平的效率对比指标。
    
    # 获取计算所需维度
    attn_module = full_model.stock_attention_mixer.sparse_attention
    embed_dim = attn_module.embed_dim
    
    # 全连接注意力的Matmul FLOPs:
    # 两个主要Matmul的复杂度均为 O(2 * S^2 * D)，其中S=序列长度, D=嵌入维度
    # 总计 = 4 * stock_num^2 * embed_dim
    dense_attn_matmul_flops = 4 * (stock_num ** 2) * embed_dim
    
    # 理论上的稀疏注意力Matmul FLOPs:
    # 真实稀疏实现的复杂度为 O(4 * active_connections * D)
    sparse_attn_matmul_flops = 4 * active_connections * embed_dim
    
    print("稀疏模型理论计算完成。详情见下方总结。")


    # --- 打印最终对比结果 ---
    print("\n" + "="*70)
    print("📊 计算量基准测试最终对比 (聚焦于核心注意力计算)")
    print("="*70)
    print(f"实验配置:")
    print(f"  - 市场: {MARKET_NAME}")
    print(f"  - 股票数: {stock_num} ({AMPLIFICATION_FACTOR}x)")
    print(f"  - 注意力嵌入维度: {embed_dim}")
    print("-" * 70)
    print("说明: 由于fvcore严重低估了此模型的总计算量，我们直接对比模型中")
    print("      计算最密集的部分：注意力矩阵乘法（Q@K.T 和 Scores@V）。")
    print("-" * 70)


    # 格式化输出
    header = "| 注意力类型         | 核心计算量 (GFLOPs) |"
    sep =    "|--------------------|-----------------------|"
    row_full = f"| 全连接 (Dense)     | {dense_attn_matmul_flops/1e9:<21.2f} |"
    row_sparse=f"| 稀疏 (Sparse)      | {sparse_attn_matmul_flops/1e9:<21.2f} |"

    print(header)
    print(sep)
    print(row_full)
    print(row_sparse)
    print("-" * 70)

    if dense_attn_matmul_flops > 0 and sparse_attn_matmul_flops > 0:
        reduction = 1 - (sparse_attn_matmul_flops / dense_attn_matmul_flops)
        print(f"✅ 结论: 稀疏注意力将模型核心计算瓶颈的理论开销减少了 {reduction:.2%}")
    print("="*70)


if __name__ == "__main__":
    main() 