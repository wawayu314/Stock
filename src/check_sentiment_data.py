#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪数据检查和归一化工具
检查情绪数据的数值分布，并提供归一化功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import pickle
from pathlib import Path

def check_sentiment_data_distribution(market_name='NYSE'):
    """检查情绪数据的数值分布"""
    
    print(f"🔍 检查 {market_name} 市场的情绪数据分布...")
    
    # 加载情绪数据
    sentiment_file = f'../dataset/processed_sentiment/{market_name.lower()}_processed_sentiment.csv'
    
    if not os.path.exists(sentiment_file):
        print(f"❌ 情绪数据文件不存在: {sentiment_file}")
        return None
    
    # 读取情绪数据
    print(f"📖 加载情绪数据: {sentiment_file}")
    df = pd.read_csv(sentiment_file)
    
    print(f"📊 情绪数据概览:")
    print(f"   - 总记录数: {len(df):,}")
    print(f"   - 列数: {len(df.columns)}")
    print(f"   - 股票数量: {df['ticker'].nunique()}")
    
    # 获取数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    
    print(f"   - 数值列: {len(numeric_columns)} 个")
    print(f"   - 非数值列: {len(non_numeric_columns)} 个")
    print(f"   - 数值列名称: {list(numeric_columns)}")
    print(f"   - 非数值列名称: {list(non_numeric_columns)}")
    
    if len(numeric_columns) == 0:
        print("⚠️ 没有找到数值列，无法进行数值分析")
        return df
    
    # 检查数值分布
    print(f"\n📈 数值列统计信息:")
    stats = df[numeric_columns].describe()
    print(stats)
    
    # 检查是否有异常值
    print(f"\n🔍 异常值检查:")
    for col in numeric_columns:
        values = df[col].dropna()
        if len(values) == 0:
            continue
            
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()
        
        # 检查是否有NaN或无穷大
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        
        print(f"   {col}:")
        print(f"     范围: [{min_val:.6f}, {max_val:.6f}]")
        print(f"     均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
        print(f"     NaN数量: {nan_count}, 无穷大数量: {inf_count}")
        
        # 检查是否可能已经归一化
        if min_val >= -5 and max_val <= 5 and abs(mean_val) < 1:
            print(f"     ✅ 可能已归一化 (标准化)")
        elif min_val >= 0 and max_val <= 1:
            print(f"     ✅ 可能已归一化 (Min-Max)")
        else:
            print(f"     ⚠️ 可能未归一化 (范围过大)")
    
    return df

def normalize_sentiment_data(df, method='standard', exclude_columns=['ticker', 'date']):
    """归一化情绪数据"""
    
    print(f"🔧 开始归一化情绪数据，方法: {method}")
    
    # 获取需要归一化的列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
    
    if len(columns_to_normalize) == 0:
        print("⚠️ 没有找到需要归一化的数值列")
        return df
    
    print(f"📊 将归一化以下列: {columns_to_normalize}")
    
    # 创建副本
    df_normalized = df.copy()
    
    # 选择归一化方法
    if method == 'standard':
        # 标准化 (Z-score)
        scaler = StandardScaler()
        print("📐 使用标准化 (Z-score): (x - mean) / std")
    elif method == 'minmax':
        # Min-Max归一化
        scaler = MinMaxScaler()
        print("📐 使用Min-Max归一化: (x - min) / (max - min)")
    elif method == 'robust':
        # 鲁棒归一化
        scaler = RobustScaler()
        print("📐 使用鲁棒归一化: (x - median) / IQR")
    else:
        print(f"❌ 不支持的归一化方法: {method}")
        return df
    
    # 执行归一化
    try:
        # 处理NaN值
        original_data = df_normalized[columns_to_normalize].copy()
        
        # 填充NaN值为0（临时处理）
        data_for_scaling = original_data.fillna(0)
        
        # 执行归一化
        normalized_data = scaler.fit_transform(data_for_scaling)
        
        # 将结果放回DataFrame
        df_normalized[columns_to_normalize] = normalized_data
        
        # 恢复原始的NaN位置
        for col in columns_to_normalize:
            nan_mask = original_data[col].isna()
            df_normalized.loc[nan_mask, col] = 0  # 将NaN设为0
        
        print("✅ 归一化完成")
        
        # 显示归一化后的统计信息
        print("\n📊 归一化后的统计信息:")
        normalized_stats = df_normalized[columns_to_normalize].describe()
        print(normalized_stats)
        
        return df_normalized, scaler
        
    except Exception as e:
        print(f"❌ 归一化失败: {e}")
        return df, None

def save_normalized_sentiment_data(df, scaler, market_name='NYSE', method='standard'):
    """保存归一化后的情绪数据"""
    
    # 创建保存目录
    output_dir = Path('../dataset/processed_sentiment/')
    output_dir.mkdir(exist_ok=True)
    
    # 保存归一化后的数据
    normalized_file = output_dir / f'{market_name.lower()}_normalized_sentiment.csv'
    df.to_csv(normalized_file, index=False)
    print(f"💾 已保存归一化数据: {normalized_file}")
    
    # 保存归一化器
    scaler_file = output_dir / f'{market_name.lower()}_sentiment_scaler_{method}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 已保存归一化器: {scaler_file}")
    
    return normalized_file, scaler_file

def compare_before_after_normalization(df_original, df_normalized, columns_to_compare=None):
    """比较归一化前后的数据分布"""
    
    if columns_to_compare is None:
        numeric_columns = df_original.select_dtypes(include=[np.number]).columns
        columns_to_compare = [col for col in numeric_columns if col not in ['ticker', 'date']]
    
    print(f"📊 比较归一化前后的数据分布...")
    
    for col in columns_to_compare[:5]:  # 只比较前5列，避免输出过多
        if col in df_original.columns and col in df_normalized.columns:
            print(f"\n📈 列: {col}")
            
            # 原始数据统计
            orig_data = df_original[col].dropna()
            norm_data = df_normalized[col].dropna()
            
            if len(orig_data) > 0 and len(norm_data) > 0:
                print(f"   原始数据: 范围[{orig_data.min():.6f}, {orig_data.max():.6f}], 均值={orig_data.mean():.6f}, 标准差={orig_data.std():.6f}")
                print(f"   归一化后: 范围[{norm_data.min():.6f}, {norm_data.max():.6f}], 均值={norm_data.mean():.6f}, 标准差={norm_data.std():.6f}")

def main():
    """主函数"""
    
    print("🚀 情绪数据检查和归一化工具")
    print("=" * 50)
    
    # 处理所有三个市场
    markets = ['NYSE', 'NASDAQ', 'SP500']
    
    for market_name in markets:
        print(f"\n{'='*20} 处理 {market_name} 市场 {'='*20}")
        
        # 1. 检查原始数据分布
        print(f"\n1️⃣ 检查 {market_name} 原始情绪数据分布")
        df_original = check_sentiment_data_distribution(market_name)
        
        if df_original is None:
            print(f"❌ 无法加载 {market_name} 情绪数据，跳过此市场")
            continue
        
        # 2. 归一化数据
        print(f"\n2️⃣ 归一化 {market_name} 情绪数据")
        
        # 尝试不同的归一化方法
        methods = ['standard', 'robust', 'minmax']
        
        success = False
        for method in methods:
            print(f"\n🔧 尝试 {method} 归一化...")
            
            try:
                df_normalized, scaler = normalize_sentiment_data(df_original, method=method)
                
                if scaler is not None:
                    # 3. 比较归一化前后
                    print(f"\n3️⃣ 比较 {method} 归一化前后")
                    compare_before_after_normalization(df_original, df_normalized)
                    
                    # 4. 保存归一化数据
                    print(f"\n4️⃣ 保存 {method} 归一化数据")
                    normalized_file, scaler_file = save_normalized_sentiment_data(
                        df_normalized, scaler, market_name, method
                    )
                    
                    print(f"✅ {market_name} {method} 归一化完成")
                    success = True
                    break  # 使用第一个成功的方法
                else:
                    print(f"❌ {market_name} {method} 归一化失败")
                    
            except Exception as e:
                print(f"❌ {market_name} {method} 归一化出错: {e}")
                continue
        
        if not success:
            print(f"❌ {market_name} 所有归一化方法都失败")
        
        print(f"\n{'='*50}")
    
    print("\n🎉 所有市场的情绪数据检查和归一化完成！")
    print("💡 建议使用 standard (标准化) 方法，因为它对异常值较为鲁棒")
    print("📁 归一化后的文件保存在: ../dataset/processed_sentiment/")
    print("   - {market}_normalized_sentiment.csv (归一化数据)")
    print("   - {market}_sentiment_scaler_standard.pkl (归一化器)")

if __name__ == "__main__":
    main() 